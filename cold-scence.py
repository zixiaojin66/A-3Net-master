# 十折交叉训练， 一次mask73行
import argparse
import csv
import datetime
import shutil

import networkx as nx
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, DataLoader

from Net import *
from smiles2vector import load_drug_smile, convert2graph
from utils import *

raw_file = 'data_ICS/raw_frequency_750.mat'
SMILES_file = 'data_ICS/drug_SMILES_750.csv'
blind_mask_mat_file = './data_ICS/blind_mask_mat_750.mat'
side_effect_label = "data_ICS/side_effect_label_750.mat"
dataset = 'drug_sideEffect'
input_dim = 109


def loss_fun(output, label, lam, eps):
    x0 = torch.where(label == 0)
    x1 = torch.where(label != 0)
    loss = torch.sum((output[x1] - label[x1]) ** 2) + lam * torch.sum((output[x0] - eps) ** 2)
    return loss


def generateMat():
    """
    将矩阵按比例mask, 将被mask的部分分为10份，生成10份mask位置矩阵，保存在./data_ICS/processed/blind_mask_mat.mat
    :return:
    """
    # 每次加载都把之前的数据删除
    filenames = os.listdir('data_ICS/processed')
    for s in filenames:
        os.remove('data_ICS/processed/' + s)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # mask, get mask Mat
    index = np.arange(0, len(raw), 1)
    np.random.shuffle(index)
    x = []
    n = int(np.ceil(len(index) / 10))
    for i in range(10):
        if i == 9:
            x.append(index.tolist())
        x.append(index[0:n].tolist())
        index = index[n:]

    dic = {}
    for i in range(10):
        mask = np.ones(raw.shape)
        mask[x[i]] = 0
        dic['mask' + str(i)] = mask
    scipy.io.savemat(blind_mask_mat_file, dic)


def split_data(tenfold=False):
    """
    读取./data_ICS/processed/blind_mask_mat.mat，根据原始频率矩阵生成10份被mask的频率矩阵并yield
    :return:
    """
    raw_frequency = scipy.io.loadmat(raw_file)
    print('******************')
    blind_mask_mat = scipy.io.loadmat(blind_mask_mat_file)
    drug_dict, drug_smile = load_drug_smile(SMILES_file)
    print(len(drug_dict))

    for idx in range(10):
        raw = raw_frequency['R']
        mask = blind_mask_mat['mask' + str(idx)]
        drug_dict, drug_smile = load_drug_smile(SMILES_file)

        index = np.asarray(np.where(mask[:, 0].flatten() == 0)[0]).tolist()

        frequencyMat = np.delete(raw, index, axis=0)
        print(len(frequencyMat))
        test_smiles = []
        test_label = []
        index.reverse()
        for i in index:
            smi = drug_smile.pop(i)
            test_smiles.append(smi)
            test_label.append(raw[i])
        train_smiles = drug_smile
        test_smiles.reverse()
        test_label.reverse()
        test_label = np.asarray(test_label)

        train_simle_graph = convert2graph(train_smiles)
        test_simle_graph = convert2graph(test_smiles)

        train_data = myDataset(root='data_ICS', dataset=dataset + '_blind_train' + str(idx), drug_simles=train_smiles,
                               frequencyMat=frequencyMat, simle_graph=train_simle_graph)
        test_data = myDataset(root='data_ICS', dataset=dataset + '_blind_test' + str(idx), drug_simles=test_smiles,
                              frequencyMat=test_label, simle_graph=test_simle_graph)
        yield idx, frequencyMat, mask

        if not tenfold and idx == 0:
            break


# training function at each epoch
def train(model, device, train_loader, optimizer, lamb, epoch, log_interval, sideEffectsGraph, raw, id, DF, not_FC, eps):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    avg_loss = []
    for batch_idx, data in enumerate(train_loader):

        sideEffectsGraph = sideEffectsGraph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out, _, _ = model(data, sideEffectsGraph, DF, not_FC)

        raw_label = data.y

        pred = out.to(device)

        loss = loss_fun(pred.flatten(), raw_label.flatten(), lamb, eps)

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if (batch_idx + 1) % log_interval == 0:
            print('{} Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(id, epoch,
                                                                              (batch_idx + 1) * len(data.y),
                                                                              len(train_loader.dataset),
                                                                              100. * (batch_idx + 1) / len(
                                                                                  train_loader),



                                                                            loss.item()))
    return sum(avg_loss) / len(avg_loss)


def predict(model, device, loader, sideEffectsGraph, DF, not_FC):
    model.eval()
    torch.cuda.manual_seed(42)
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for batch_idx, data in enumerate(loader):
            raw_label = torch.FloatTensor(data.y)
            data = data.to(device)
            out, _, _ = model(data, sideEffectsGraph, DF, not_FC)

            location = torch.where(raw_label != 0)
            pred = out[location]
            label = raw_label[location]

            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def evaluate(model, device, loader, sideEffectsGraph, DF, not_FC, result_folder, id):
    total_preds = torch.Tensor()
    total_label = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            # 查找被mask的数据
            label = data.y
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()

            total_preds = torch.cat((total_preds, pred), 0)
            total_label = torch.cat((total_label, label), 0)
            # test batch size must be 1
            pred = pred.numpy().flatten()
            label = (label.numpy().flatten() != 0).astype(int)

            singleDrug_auc.append(roc_auc_score(label, pred))
            singleDrug_aupr.append(average_precision_score(label, pred))
    if id == 1:
        pred_result = pd.read_csv(result_folder + '/blind_pred.csv', header=0, index_col=None).values
        raw_result = pd.read_csv(result_folder + '/blind_raw.csv', header=0, index_col=None).values
    else:
        pred_result = pd.read_csv(result_folder + '/blind_pred.csv', header=None, index_col=None).values
        raw_result = pd.read_csv(result_folder + '/blind_raw.csv', header=None, index_col=None).values
    print(pred_result.shape)


    pred_result = pd.DataFrame(np.vstack((pred_result, total_preds.numpy())))
    raw_result = pd.DataFrame(np.vstack((raw_result, total_label.numpy())))
    pred_result.to_csv(result_folder + '/blind_pred.csv', header=False, index=False)
    raw_result.to_csv(result_folder + '/blind_raw.csv', header=False, index=False)


    drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
    drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
    total_preds = total_preds.numpy()
    total_label = total_label.numpy()

    pos = total_preds[np.where(total_label)]
    pos_label = np.ones(len(pos))

    neg = total_preds[np.where(total_label == 0)]
    neg_label = np.zeros(len(neg))

    y = np.hstack((pos, neg))
    y_true = np.hstack((pos_label, neg_label))
    auc_all = roc_auc_score(y_true, y)
    aupr_all = average_precision_score(y_true, y)

    # others

    Te = {}
    Te_all = {}
    Te_pairs = np.where(total_label)
    Te_pairs = np.array(Te_pairs).transpose()

    for pair in Te_pairs:
        drug_id = pair[0]
        SE_id = pair[1]
        if drug_id not in Te:
            Te[drug_id] = [SE_id]
        else:
            Te[drug_id].append(SE_id)
    shape = total_label.shape
    for i in range(shape[0]):
        Te_all[i] = [i for i in range(shape[1])]

    positions = [1, 5, 10, 15]
    map_value, auc_value, ndcg, prec, rec = evaluate_others(total_preds, Te_all, Te, positions)

    p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
    r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
    return auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15


def main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name, frequencyMat,
         id, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca):
    print('\n=======================================================================================')
    print('\n第 {} 次训练：\n'.format(id))
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('Batch size: ', train_batch)
    print('Lambda: ', lamb)
    print('weight_decay: ', weight_decay)
    print('KNN: ', knn)
    print('metric: ', metric)
    print('tenfold: ', tenfold)
    print('DF: ', DF)
    print('not_FC: ', not_FC)
    print('output_dim: ', output_dim)
    print('Eps: ', eps)
    print('PCA: ', pca)

    model_st = modeling.__name__
    train_losses = []
    test_MSE = []
    test_pearsons = []
    test_rMSE = []
    test_spearman = []
    test_MAE = []
    print('\nrunning on ', model_st + '_' + dataset)
    processed_raw = raw_file

    if not os.path.isfile(processed_raw):
        print('Missing raw FrequencyMat, exit!!!')
        exit(1)

    # 生成副作用的graph信息
    frequencyMat = frequencyMat.T
    if pca:
        pca_ = PCA(n_components=256)
        similarity_pca = pca_.fit_transform(frequencyMat)
        print('PCA 信息保留比例： ')
        print(sum(pca_.explained_variance_ratio_))
        A = kneighbors_graph(similarity_pca, knn, mode='connectivity', metric=metric, include_self=False)
    else:
        A = kneighbors_graph(frequencyMat, knn, mode='connectivity', metric=metric, include_self=False)
    G = nx.from_numpy_matrix(A.todense())
    edges = []
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])

    edges = np.array(edges).T
    edges = torch.tensor(edges, dtype=torch.long)

    node_label = scipy.io.loadmat(side_effect_label)['node_label']
    feat = torch.tensor(node_label, dtype=torch.float)
    sideEffectsGraph = Data(x=feat, edge_index=edges)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # make data_WS Pytorch mini-batch processing ready
    train_data = myDataset(root='data_ICS', dataset=dataset + '_blind_train' + str(id - 1))
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    test_data = myDataset(root='data_ICS', dataset=dataset + '_blind_test' + str(id - 1))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print('CPU/GPU: ', torch.cuda.is_available())

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    model = modeling(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_file_name = str(id) + 'Blind_MF_' + model_st + '_epoch=' + str(num_epoch) + '.model'
    result_log = result_folder + '/' + model_st + '_result.csv'
    loss_fig_name = str(id) + model_st + '_loss'
    pearson_fig_name = str(id) + model_st + '_pearson'
    MSE_fig_name = str(id) + model_st + '_MSE'

    for epoch in range(num_epoch):
        train_loss = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, lamb=lamb,
                           epoch=epoch + 1, log_interval=log_interval, sideEffectsGraph=sideEffectsGraph, raw=raw,
                           id=id, DF=DF, not_FC=not_FC, eps=eps)
        train_losses.append(train_loss)

        # test_labels, test_preds = predict(model=model, device=device, loader=test_loader,
        #                                   sideEffectsGraph=sideEffectsGraph, raw=raw, mask=mask, DF=DF, not_FC=not_FC)
        # ret_test = [mse(test_labels, test_preds), pearson(test_labels, test_preds), rmse(test_labels, test_preds),
        #             spearman(test_labels, test_preds), MAE(test_labels, test_preds)]
        #
        # test_pearsons.append(ret_test[1])
        # test_MSE.append(ret_test[0])
        # test_rMSE.append(ret_test[2])
        # test_spearman.append(ret_test[3])
        # test_MAE.append(ret_test[4])
    torch.save(model, "model_cold.pt")
    test_labels, test_preds = predict(model=model, device=device, loader=test_loader,
                                      sideEffectsGraph=sideEffectsGraph, DF=DF, not_FC=not_FC)
    ret_test = [mse(test_labels, test_preds), pearson(test_labels, test_preds), rmse(test_labels, test_preds),
                spearman(test_labels, test_preds), MAE(test_labels, test_preds)]
    test_pearsons, test_rMSE, test_spearman, test_MAE = ret_test[1], ret_test[2], ret_test[3], ret_test[4]
    auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15 = evaluate(model=model,
                                                                                                         device=device,
                                                                                                         loader=test_loader,
                                                                                                         sideEffectsGraph=sideEffectsGraph,
                                                                                                         DF=DF,
                                                                                                         not_FC=not_FC,
                                                                                                         result_folder=result_folder,
                                                                                                         id=id)
    if save_model:
        checkpointsFolder = result_folder + '/checkpoints/'
        isCheckpointExist = os.path.exists(checkpointsFolder)
        if not isCheckpointExist:
            os.makedirs(checkpointsFolder)
        torch.save(model.state_dict(), checkpointsFolder + model_file_name)

    result = [test_pearsons, test_rMSE, test_spearman, test_MAE, auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg,
              p1, p5, p10, p15, r1, r5, r10, r15]
    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)

    print('Test:\nPearson: {:.5f}\trMSE: {:.5f}\tSpearman: {:.5f}\tMAE: {:.5f}'.format(result[0], result[1], result[2],
                                                                                       result[3]))
    print('\tall AUC: {:.5f}\tall AUPR: {:.5f}\tdrug AUC: {:.5f}\tdrug AUPR: {:.5f}'.format(result[4], result[5],
                                                                                            result[6], result[7]))
    print('\tMAP: {:.5f}\tnDCG@10: {:.5f}'.format(map_value, ndcg))
    print('\tP@1: {:.5f}\tP@5: {:.5f}\tP@10: {:.5f}\tP@15: {:.5f}'.format(p1, p5, p10, p15))
    print('\tR@1: {:.5f}\tR@5: {:.5f}\tR@10: {:.5f}\tR@15: {:.5f}'.format(r1, r5, r10, r15))
    # train loss
    my_draw_loss(train_losses, loss_fig_name, result_folder)
    # # test pearson
    # draw_pearson(test_pearsons, pearson_fig_name, result_folder)
    # # test mse
    # my_draw_mse(test_MSE, test_rMSE, MSE_fig_name, result_folder)


if __name__ == '__main__':

    total_start = datetime.datetime.now()
    # 参数定义
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=3,
                        help='0: GCN, 1: GIN, 2: GAT2, 3: GAT3, 4: GAT, 5: GCN')
    parser.add_argument('--metric', type=int, required=False, default=0, help='0: cosine, 1: jaccard, 2: euclidean')
    parser.add_argument('--train_batch', type=int, required=False, default=10, help='Batch size training set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.001, help='weight_decay')
    parser.add_argument('--lamb', type=float, required=False, default=0.03, help='LAMBDA')
    parser.add_argument('--epoch', type=int, required=False, default=3000, help='Number of epoch')
    parser.add_argument('--knn', type=int, required=False, default=5, help='Number of KNN')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda:0', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=256, help='output dim, <= 109')
    parser.add_argument('--eps', type=float, required=False, default=0.5, help='regard 0 as eps when training')

    parser.add_argument('--tenfold', action='store_true', default=False, help='use 10 folds Cross-validation ')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model and features')
    parser.add_argument('--DF', action='store_true', default=False, help='use DF decoder')
    parser.add_argument('--not_FC', action='store_true', default=False, help='not use Linear layers')
    parser.add_argument('--PCA', action='store_true', default=False, help='use PCA')
    args = parser.parse_args()

    modeling = [GCN, GIN, GAT, GAT3, GAT1, RGCN][args.model]
    metric = ['cosine', 'jaccard', 'euclidean'][args.metric]
    train_batch = args.train_batch
    lr = args.lr
    knn = args.knn
    num_epoch = args.epoch
    weight_decay = args.wd
    lamb = args.lamb
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    tenfold = args.tenfold
    save_model = args.save_model
    DF = args.DF
    not_FC = args.not_FC
    output_dim = args.dim
    eps = args.eps
    pca = args.PCA

    processed_mask_mat = blind_mask_mat_file
    if not os.path.isfile(processed_mask_mat):
        print('Missing data_WS files, generating......')
        generateMat()

    result_folder = './result_ICS/'

    if tenfold:
        result_folder += '10ICS_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)
    else:
        result_folder += '1ICS_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)

    isExist = os.path.exists(result_folder)
    if not isExist:
        os.makedirs(result_folder)
    else:
        # 清空原文件 添加表头
        shutil.rmtree(result_folder)
        os.makedirs(result_folder)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']
    pred_result = result_folder + '/blind_pred.csv'
    pred_ = pd.DataFrame(columns=[i for i in range(raw.shape[1])])
    pred_.to_csv(pred_result, header=True, index=False)
    raw_result = result_folder + '/blind_raw.csv'
    raw_ = pd.DataFrame(columns=[i for i in range(raw.shape[1])])
    raw_.to_csv(raw_result, header=True, index=False)

    result_log = result_folder + '/' + modeling.__name__ + '_result.csv'
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    with open(result_log, 'w', newline='') as f:
        fieldnames = ['pearson', 'rMSE', 'spearman', 'MAE', 'auc_all', 'aupr_all', 'drugAUC', 'drugAUPR', 'MAP', 'nDCG',
                      'P1', 'P5', 'P10', 'P15', 'R1', 'R5', 'R10', 'R15']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for (id, frequencyMat, mask) in split_data(tenfold):
        start = datetime.datetime.now()
        main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name,
             frequencyMat, id + 1, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca)
        end = datetime.datetime.now()
        print('本次运行时间：{}\t'.format(end - start))

    data = pd.read_csv(result_log)
    L = len(data.rMSE)
    avg = [sum(data.pearson) / L, sum(data.rMSE) / L, sum(data.spearman) / L, sum(data.MAE) / L, sum(data.auc_all) / L,
           sum(data.aupr_all) / L, sum(data.drugAUC) / L, sum(data.drugAUPR) / L, sum(data.MAP) / L, sum(data.nDCG) / L,
           sum(data.P1) / L, sum(data.P5) / L, sum(data.P10) / L, sum(data.P15) / L, sum(data.R1) / L, sum(data.R5) / L,
           sum(data.R10) / L, sum(data.R15) / L]
    print('\n\tavg pearson: {:.4f}\tavg rMSE: {:.4f}\tavg spearman: {:.4f}\tavg MAE: {:.4f}'.format(avg[0], avg[1],
                                                                                                    avg[2], avg[3]))
    print('\tavg all AUC: {:.4f}\tavg all AUPR: {:.4f}\tavg drug AUC: {:.4f}\tavg drug AUPR: {:.4f}'.format(avg[4],
                                                                                                            avg[5],
                                                                                                            avg[6],
                                                                                                            avg[7]))
    print('\tavg MAP: {:.4f}\tavg nDCG@10: {:.4f}'.format(avg[8], avg[9]))
    print('\tavg P@1: {:.4f}\tavg P@5: {:.4f}\tavg P@10: {:.4f}\tavg P@15: {:.4f}'.format(avg[10], avg[11], avg[12],
                                                                                          avg[13]))
    print('\tavg R@1: {:.4f}\tavg R@5: {:.4f}\tavg R@10: {:.4f}\tavg R@15: {:.4f}'.format(avg[14], avg[15], avg[16],
                                                                                          avg[17]))
    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['avg'])
        writer.writerow(avg)
    total_end = datetime.datetime.now()
    print('总体运行时间：{}\t'.format(total_end - total_start))
