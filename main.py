import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import time
from model import LPDecoder_ogb as LPDecoder
from model import GCN_mgaev3 as GCN
from model import SAGE_mgaev2 as SAGE
from model import GIN_mgaev2 as GIN
from model import MLP
from model import Shared_kernel as Skernel
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
from utils import edgemask_um, edgemask_dm, do_edge_split_nc, calculate_metrics, resplit, loss_to_log, show_auc, \
    show_auprc
import os.path as osp
# import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def random_edge_mask(args, edge_index, device, num_nodes):
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * args.mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index_train).t()
    return adj, edge_index_train, edge_index_mask


def train(model, decoder, mlp, skernel, link_data, edge_index, node_data, optimizer, labels, mask, args):
    model.train()  # encoder
    decoder.train()  # decoder
    mlp.train()  # classifier
    skernel.train()  # share-kernel

    total_loss = total_examples = 0
    if args.mask_type == 'um':
        adj, _, pos_train_edge = edgemask_um(args.mask_ratio, edge_index, link_data.x.device, link_data.x.shape[0])
    else:
        adj, _, pos_train_edge = edgemask_dm(args.mask_ratio, edge_index, link_data.x.device, link_data.x.shape[0])
    adj = adj.to(link_data.x.device)

    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size,shuffle=True):
        optimizer.zero_grad()

        h = model(link_data.x, adj)
        edge = pos_train_edge[perm].t()

        pos_out = decoder(h, edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, link_data.x.shape[0], edge.size(), dtype=torch.long,device=link_data.x.device)
        neg_out = decoder(h, edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        x_list = []
        for data in node_data:  # node and edge
            h = model(data.x, data.full_adj_t)
            feature = [feature.detach() for feature in h]
            feature_list = extract_feature_list_layer2(feature)  # feature_list[0]: last; feature_list[1]: last two
            x_list.append(feature_list[1])  # last two embedding combine

        con_feature = skernel(x_list)
        out = mlp(con_feature)
        clf_loss = get_clf_loss(out, labels, mask)

        loss = pos_loss + neg_loss + args.alpha * clf_loss  # total loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(skernel.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


# classify loss
def get_clf_loss(out, labels, mask):
    pos = torch.eq(labels[mask], 1).float()
    neg = torch.eq(labels[mask], 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + alpha_neg * neg
    return F.binary_cross_entropy_with_logits(out[mask], labels[mask], weights, reduction='mean')


def extract_feature_list_layer2(feature_list):
    xx_list = []
    xx_list.append(feature_list[-1])
    tmp_feat = torch.cat(feature_list, dim=-1)
    xx_list.append(tmp_feat)
    return xx_list


@torch.no_grad()
def test_classify_mlp(feature, mlp, labels, args, mask):
    mlp.eval()
    out = mlp(feature)
    out = out[mask]
    out = torch.squeeze(out)
    true_lab = labels[mask].cpu()
    pred_lab = np.zeros(true_lab.shape[0])
    pred_lab[out.cpu() > 0.5] = 1
    y_score = out.cpu().detach().numpy()
    acc, auprc, f1, auc = calculate_metrics(true_lab, pred_lab, y_score)
    return f1, acc, auc, auprc, true_lab, y_score

def prediction(node_data, model, skernel, mlp, cv):
    model.eval()
    skernel.eval()
    mlp.eval()
    
    x_list = []
    for data in node_data:
        feature = model(data.x, data.full_adj_t)
        feature = [feature_.detach() for feature_ in feature]
        feature_list = extract_feature_list_layer2(feature)
        x_list.append(feature_list[1])  # combine last two
    con_feature = skernel(x_list)
    out = mlp(con_feature)
    y_score = out.cpu().detach().numpy()
    
    pred_save_path = './pred/' + str(cv+1) + '_final_pred.pkl'
    torch.save(y_score, pred_save_path)


def main():
    parser = argparse.ArgumentParser(description='S2-GAE (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', type=str, default='GCN')
    parser.add_argument('--link_dataset', type=str, default='concentrate')
    parser.add_argument('--node_dataset', type=str, default=['CPDB_v34', 'string', 'irefindex'])
    parser.add_argument('--use_valedges_as_input', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--decode_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--decode_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500) # 500
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--mask_type', type=str, default='dm', help='dm | um')  # whether to use mask features
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load link dataset
    if args.link_dataset in {'concentrate', 'concentrate_v4'}:
        path = './dataset/' + args.link_dataset + '_dataset.pkl'
        link_data = torch.load(path)
        labels = link_data.y
    else:
        raise ValueError(args.link_dataset)

    if link_data.is_undirected():
        edge_index = link_data.edge_index
    else:
        print('### Input graph {} is directed'.format(args.link_dataset))
        edge_index = to_undirected(link_data.edge_index)

    link_data.full_adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                        sparse_sizes=(link_data.x.shape[0], link_data.x.shape[0])).t()

    edge_index, test_edge, test_edge_neg = do_edge_split_nc(edge_index, link_data.x.shape[0])
    labels = labels.to(device)
    link_data = link_data.to(device)

    # load node datasets
    node_data = []
    for dataset in args.node_dataset:
        if dataset in {'CPDB_v34', 'CPDB', 'string', 'irefindex', 'humannet'}:
            path = './dataset/' + dataset + '_dataset.pkl'
            data = torch.load(path)
            if data.is_undirected():
                n_edge_index = data.edge_index
            else:
                n_edge_index = to_undirected(data.edge_index)

            data.full_adj_t = SparseTensor(row=n_edge_index[0], col=n_edge_index[1],
                                           sparse_sizes=(data.x.shape[0], data.x.shape[0])).t()
            data = data.to(device)
            node_data.append(data)

    train_mask, val_mask, test_mask = resplit(link_data, args.cv, device)

    # save model path
    save_path_model = 'weight/s2gaemlp-' + args.use_sage + '_{}_{}'.format(args.link_dataset,
                                                                           args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels) + '_model.pth'
    save_path_decoder = 'weight/s2gaemlp-' + args.use_sage + '_{}_{}'.format(args.link_dataset,
                                                                            args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels) + '_pred.pth'
    save_path_mlp = 'weight/s2gaemlp-' + args.use_sage + '_{}_{}'.format(args.link_dataset,
                                                                        args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels) + '_mlp.pth'
    save_path_skernel = 'weight/s2gaemlp-' + args.use_sage + '_{}_{}'.format(args.link_dataset,
                                                                            args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels) + '_skernel.pth'

    out2_dict = {0: 'last', 1: 'combine'}
    result_dict = out2_dict
    mlp_result_final_acc = np.zeros(shape=[args.cv, len(out2_dict)])
    mlp_result_final_auc = np.zeros(shape=[args.cv, len(out2_dict)])
    mlp_result_final_auprc = np.zeros(shape=[args.cv, len(out2_dict)])

    if args.use_sage == 'SAGE':
        model = SAGE(link_data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.use_sage == 'GIN':
        model = GIN(link_data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    else:
        model = GCN(link_data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    decoder = LPDecoder(args.hidden_channels, args.decode_channels, 1, args.num_layers,
                        args.decode_layers, args.dropout).to(device)

    mlp = MLP(in_channels=args.hidden_channels * 2, drop_rate=0.1, hidden_dim=25).to(device)
    skernel = Skernel(ppi_num=len(args.node_dataset)).to(device)

    print('Start training with mask ratio={} # optimization edges={} / {}'.format(args.mask_ratio,
                                                                                  int(args.mask_ratio *
                                                                                      edge_index.shape[0]),
                                                                                  edge_index.shape[0]))

    # plot roc-curve and PR-curve
    visual_args_true = []
    visual_args_score = []

    for cv in range(args.cv):
        model.reset_parameters()
        decoder.reset_parameters()
        mlp.reset_parameters()
        skernel.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(decoder.parameters()) + list(mlp.parameters()) + list(skernel.parameters()),
            lr=args.lr, weight_decay=0.0001)  # 0.0001 is best right now

        log_file = './log/multi_ppi_cv_' + str(cv) + '.txt'
        log_loss = []  # record loss
        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):
            # 训练模型
            t1 = time.time()
            loss = train(model, decoder, mlp, skernel, link_data, edge_index, node_data, optimizer, labels,
                         train_mask[cv], args)
            log_loss.append((epoch, loss))  # record loss
            t2 = time.time()

            x_list = []
            for data in node_data:
                feature = model(data.x, data.full_adj_t)
                feature = [feature_.detach() for feature_ in feature]
                feature_list = extract_feature_list_layer2(feature)
                x_list.append(feature_list[1])  # combine last two

            con_feature = skernel(x_list)
            f1, acc, auc, auprc_val, _, _ = test_classify_mlp(con_feature, mlp, labels, args, val_mask[cv])

            if auprc_val >= best_valid:
                best_valid = auprc_val
                best_epoch = epoch
                torch.save(model.state_dict(), save_path_model)
                torch.save(decoder.state_dict(), save_path_decoder)
                torch.save(mlp.state_dict(), save_path_mlp)
                torch.save(skernel.state_dict(), save_path_skernel)
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == 50:
                print('Early stop at {}'.format(epoch))
                break
        loss_to_log(log_file, log_loss, best_epoch)  # record loss

        print('##### Testing on {}/{}'.format(cv, args.cv))

        model.load_state_dict(torch.load(save_path_model))
        decoder.load_state_dict(torch.load(save_path_decoder))
        mlp.load_state_dict(torch.load(save_path_mlp))
        skernel.load_state_dict(torch.load(save_path_skernel))

        x_list = []
        for data in node_data:
            feature = model(data.x, data.full_adj_t)
            feature = [feature_.detach() for feature_ in feature]
            feature_list = extract_feature_list_layer2(feature)
            x_list.append(feature_list[1])
        con_feature = skernel(x_list)

        f1, acc, auc, auprc, y_true, y_score = test_classify_mlp(con_feature, mlp, labels, args, test_mask[cv])
        visual_args_true.append(y_true)  # for plot
        visual_args_score.append(y_score)   # for plot

        mlp_result_final_acc[cv, 1] = acc
        mlp_result_final_auc[cv, 1] = auc
        mlp_result_final_auprc[cv, 1] = auprc

        print('**** MLP test acc on cv {}/{} for {} is F1={} acc={} auc={} auprc={}'
              .format(cv + 1, args.cv, result_dict[1], f1, acc, auc, auprc))

        # prediction
        prediction(node_data, model, skernel, mlp, cv)

    print('\n------- Print final result for MLP')

    temp_result_acc = mlp_result_final_acc[:, 1]
    temp_result_auc = mlp_result_final_auc[:, 1]
    temp_result_auprc = mlp_result_final_auprc[:, 1]
    print('#### Final mlp test result on {} is acc-mean={} std={}'.format(result_dict[1], np.mean(temp_result_acc),
                                                                          np.std(temp_result_acc)))
    print('#### Final mlp test result on {} is auc-mean={} std={}'.format(result_dict[1], np.mean(temp_result_auc),
                                                                          np.std(temp_result_auc)))
    print('#### Final mlp test result on {} is auprc-mean={} std={}'.format(result_dict[1], np.mean(temp_result_auprc),
                                                                            np.std(temp_result_auprc)))

    # plot roc curve and PR-curve
    show_auc(visual_args_true, visual_args_score, args.cv)
    show_auprc(visual_args_true, visual_args_score, args.cv)

    # save roc curve and PR-curve
    curve_path = './MTCDG_curve.json'
    data_to_save = {'y_true': visual_args_true, 'y_score': visual_args_score}
    torch.save(data_to_save, curve_path)


if __name__ == "__main__":
    main()
