import os

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
from vector import load_drug_smile, convert2graph
from utils import *

raw_file = 'data/raw_frequency_750.mat'
SMILES_file = 'data/drug_SMILES_750.csv'
mask_mat_file = 'data/mask_mat_750.mat'
side_effect_label = 'data/side_effect_label_750.mat'
input_dim = 109


def loss_fun(output, label, lam, eps):
    x0 = torch.where(label == 0)
    x1 = torch.where(label != 0)
    loss = torch.sum((output[x1] - label[x1]) ** 2) + lam * torch.sum((output[x0] - eps) ** 2)
    return loss

def generateMat(k=10):
    """
    将矩阵按比例mask, 将被mask的部分分为10份，生成10份mask位置矩阵，保存在./data_WS/processed/mask_mat.mat
    :return:
    """
    # 每次加载都把之前的数据删除
    filenames = os.listdir('warm-scence_data')
    print(filenames)
    for s in filenames:
        os.remove('warm-scence_data/' + s)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # mask, get mask Mat
    index_pair = np.where(raw != 0)
    index_arr = np.arange(0, index_pair[0].shape[0], 1)
    np.random.shuffle(index_arr)
    x = []
    n = math.ceil(index_pair[0].shape[0] / k)
    for i in range(k):
        if i == k - 1:
            x.append(index_arr[0:].tolist())
        else:
            x.append(index_arr[0:n].tolist())
            index_arr = index_arr[n:]

    dic = {}
    for i in range(k):
        mask = np.ones(raw.shape)
        mask[index_pair[0][x[i]], index_pair[1][x[i]]] = 0
        dic['mask' + str(i)] = mask
    scipy.io.savemat(mask_mat_file, dic)


def split_data(tenfold=False):
    """
    读取 data/mask_mat.mat，根据原始频率矩阵生成10份被mask的频率矩阵并yield
    :return:
    """
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']
    mask_mat = scipy.io.loadmat(mask_mat_file)
    drug_dict, drug_smile = load_drug_smile(SMILES_file)
    print(len(drug_dict))

    simle_graph = convert2graph(drug_smile)
    dataset = 'drug_sideEffect'

    for i in range(10):
        mask = mask_mat['mask' + str(i)]
        frequencyMat = raw * mask
        # np.set_printoptions(precision=4)    # 保留四位小数
        # np.savetxt('./data_WS/processed/frequencyMat.csv', frequencyMat, fmt='%.2f')
        data = myDataset(root='data_WS', dataset=dataset + '_data' + str(i), drug_simles=drug_smile,
                         frequencyMat=frequencyMat,
                         simle_graph=simle_graph)
        yield i, frequencyMat, mask

        if not tenfold and i == 0:
            break


# training function at each epoch
def train(model, device, train_loader, optimizer, lamb, epoch, log_interval, sideEffectsGraph, raw, id, DF, not_FC, eps):
    """

    :param model:
    :param device:
    :param train_loader: 数据加载器
    :param optimizer: 优化器
    :param epoch: 训练数
    :param log_interval: 记录间隔
    :param sideEffectsGraph: 副作用图信息，
    :param raw: 原始数据
    :param id: 第id次训练(第id折）
    :return: 本次训练的平均损失
    """
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    singleDrug_auc = []
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        # 查找被mask的数据
        index = [x[0] for x in data.index]
        label = data.y
        sideEffectsGraph = sideEffectsGraph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out, x, x_e = model(data, sideEffectsGraph, DF, not_FC)

        pred = out.to(device)
        train_label = torch.FloatTensor(label)
        loss = loss_fun(pred.flatten(), train_label.flatten().to(device), lamb, eps)
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


def predict(model, device, loader, sideEffectsGraph, raw, DF, not_FC):
    """
    :param model:
    :param device:
    :param loader: 数据加载器
    :param sideEffectsGraph: 副作用图信息，
    :param raw: 原始数据
    :return: 所有的被mask的原始值，所有的被mask的预测值，都是1维
    """
    # 声明为张量
    total_preds = torch.Tensor()
    total_reals = torch.Tensor()
    model.eval()
    torch.cuda.manual_seed(42)
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            # 查找被mask的数据
            index = [x[0] for x in data.index]

            label = data.y
            raw_label = torch.FloatTensor(raw[index])
            index_pair = torch.where(raw_label != label)
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)

            pred = output.cpu()[index_pair]
            real = raw_label[index_pair]

            # torch.cat()：将两个tensor拼接，按维数0拼接（往下拼）或按维数1拼接（往右拼）
            total_preds = torch.cat((total_preds, pred), 0)
            total_reals = torch.cat((total_reals, real), 0)

    return total_reals.numpy().flatten(), total_preds.numpy().flatten()


def getAllResultMatrix(model, device, loader, sideEffectsGraph, mask, result_folder, DF, not_FC):
    """
    保存预测结果
    """
    # 声明为张量
    pred_result = pd.read_csv(result_folder + '/pred_result.csv', header=None, index_col=None).values
    # pred_result = np.loadtxt(result_folder + '/pred_result.csv')
    # print(pred_result.shape)
    pred = torch.Tensor()
    model.eval()
    torch.cuda.manual_seed(42)
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    # 顺序加载数据
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            # print(output.shape, type(output))
            # exit(1)
            pred = torch.cat((pred, output.cpu()), 0)
        # 保存此次预测的所有位置
        # np.set_printoptions(precision=4)
        pred = pred.numpy()
        # print(pred.shape)
        mask = (mask == 0).astype(int)
        pred_result = pred_result + pred * mask
        # print(pred_result.shape)

        pred_result = pd.DataFrame(pred_result)
        pred_result.to_csv(result_folder + '/pred_result.csv', header=False, index=False, float_format='%.4f')


def evaluate(model, device, loader, sideEffectsGraph, mask, raw, DF, not_FC):
    total_preds = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            # 查找被mask的数据
            index = [x[0] for x in data.index]
            train_label = data.y.numpy().flatten()
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()
            # torch.cat()：将两个tensor拼接，按维数0拼接（往下拼）或按维数1拼接（往右拼）
            total_preds = torch.cat((total_preds, pred), 0)

            pred = pred.numpy().flatten()
            train_label = (train_label != 0).astype(int)
            if sum(mask[index].flatten()) == len(mask[index].flatten()):
                continue
            posi = pred[np.where(mask[index].flatten() == 0)[0]]
            nege = pred[np.where((mask[index].flatten() - train_label))[0]]
            y = np.hstack((posi, nege))
            y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            singleDrug_auc.append(roc_auc_score(y_true, y))
            singleDrug_aupr.append(average_precision_score(y_true, y))

    drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
    drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
    print('num of singleDrug_auc: ', len(singleDrug_auc))
    # print('drugAUPR: ', drugAUPR)
    total_preds = total_preds.numpy()

    pos = total_preds[np.where(mask == 0)]
    pos_label = np.ones(len(pos))

    neg = total_preds[np.where(raw == 0)]
    neg_label = np.zeros(len(neg))

    y = np.hstack((pos, neg))
    y_true = np.hstack((pos_label, neg_label))
    auc_all = roc_auc_score(y_true, y)
    aupr_all = average_precision_score(y_true, y)
    # others
    Tr_neg = {}
    Te = {}
    train_data = raw * mask
    Te_pairs = np.where(mask == 0)
    Tr_neg_pairs = np.where(train_data == 0)
    Te_pairs = np.array(Te_pairs).transpose()
    Tr_neg_pairs = np.array(Tr_neg_pairs).transpose()
    for te_pair in Te_pairs:
        drug_id = te_pair[0]
        SE_id = te_pair[1]
        if drug_id not in Te:
            Te[drug_id] = [SE_id]
        else:
            Te[drug_id].append(SE_id)

    for te_pair in Tr_neg_pairs:
        drug_id = te_pair[0]
        SE_id = te_pair[1]
        if drug_id not in Tr_neg:
            Tr_neg[drug_id] = [SE_id]
        else:
            Tr_neg[drug_id].append(SE_id)

    positions = [1, 5, 10, 15]
    map_value, auc_value, ndcg, prec, rec = evaluate_others(total_preds, Tr_neg, Te, positions)

    p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
    r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
    return auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15
def evaluatee(model, device, loader, sideEffectsGraph, mask, raw, DF, not_FC):
    total_preds = torch.Tensor()
    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()
            # torch.cat()：将两个tensor拼接，按维数0拼接（往下拼）或按维数1拼接（往右拼）
            total_preds = torch.cat((total_preds, pred), 0)
    total_preds = total_preds.numpy()
    pos = total_preds[np.where(mask == 0)]
    pos_label = np.ones(len(pos))
    neg = total_preds[np.where(raw == 0)]
    neg_label = np.zeros(len(neg))

    y = np.hstack((pos, neg))
    y_true = np.hstack((pos_label, neg_label))
    auc_all = roc_auc_score(y_true, y)
    print('小飞棍来咯: ', auc_all)
    return auc_all

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
    dataset = 'drug_sideEffect'
    train_losses = []
    # test_MSE = []
    # test_pearsons = []
    # test_rMSE = []
    # test_spearman = []
    # test_MAE = []
    print('\nrunning on ', model_st + '_' + dataset)
    processed_raw = raw_file

    if not os.path.isfile(processed_raw):
        print('Missing FrequencyMat, exit!!!')
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

    # load  side_effect_label mat ，用node_label做点信息 994*243
    node_label = scipy.io.loadmat(side_effect_label)['node_label']
    feat = torch.tensor(node_label, dtype=torch.float)
    sideEffectsGraph = Data(x=feat, edge_index=edges)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # make data_WS Pytorch mini-batch processing ready
    train_data = myDataset(root='data_WS', dataset='drug_sideEffect_data' + str(id - 1))
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    test_loader = DataLoader(train_data, batch_size=1, shuffle=False)

    print('CPU/GPU: ', torch.cuda.is_available())

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    model = modeling(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_file_name = str(id) + 'MF_' + model_st + '_epoch=' + str(num_epoch) + '.model'
    result_log = result_folder + '/' + model_st + '_result.csv'
    loss_fig_name = str(id) + model_st + '_loss'
    pearson_fig_name = str(id) + model_st + '_pearson'
    MSE_fig_name = str(id) + model_st + '_MSE'
    rMSE_fig_name = str(id) + model_st + '_rMSE'
    auc_all=[]
    for epoch in range(num_epoch):
        train_loss = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, lamb=lamb,
                           epoch=epoch + 1, log_interval=log_interval, sideEffectsGraph=sideEffectsGraph, raw=raw,
                           id=id, DF=DF, not_FC=not_FC, eps=eps)

        train_losses.append(train_loss)

    torch.save(model.state_dict(), "new.pth")
    if save_model:
        checkpointsFolder = result_folder + '/checkpoints/'
        isCheckpointExist = os.path.exists(checkpointsFolder)
        if not isCheckpointExist:
            os.makedirs(checkpointsFolder)
        torch.save(model.state_dict(), checkpointsFolder + model_file_name)

    test_labels, test_preds = predict(model=model, device=device, loader=test_loader,
                                      sideEffectsGraph=sideEffectsGraph, raw=raw, DF=DF, not_FC=not_FC)

    ret_test = [mse(test_labels, test_preds), pearson(test_labels, test_preds), rmse(test_labels, test_preds),
                spearman(test_labels, test_preds), MAE(test_labels, test_preds)]
    test_pearsons, test_rMSE, test_spearman, test_MAE = ret_test[1], ret_test[2], ret_test[3], ret_test[4]
    auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15 = evaluate(model=model,
                                                                                                         device=device,
                                                                                                         loader=test_loader,
                                                                                                         sideEffectsGraph=sideEffectsGraph,
                                                                                                         mask=mask,
                                                                                                         raw=raw, DF=DF,
                                                                                                         not_FC=not_FC)

    # 写入预测效果
    result = [test_pearsons, test_rMSE, test_spearman, test_MAE, auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg,
              p1, p5, p10, p15, r1, r5, r10, r15]
    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)
    # 写入预测值
    getAllResultMatrix(model=model, device=device, loader=test_loader, sideEffectsGraph=sideEffectsGraph, mask=mask,
                       result_folder=result_folder, DF=DF, not_FC=not_FC)

    print('Test:\nPearson: {:.5f}\trMSE: {:.5f}\tSpearman: {:.5f}\tMAE: {:.5f}'.format(result[0], result[1], result[2],
                                                                                       result[3]))
    print('\tall AUC: {:.5f}\tall AUPR: {:.5f}\tdrug AUC: {:.5f}\tdrug AUPR: {:.5f}'.format(result[4], result[5],
                                                                                            result[6], result[7]))
    print('\tMAP: {:.5f}\tnDCG@10: {:.5f}'.format(map_value, ndcg))
    print('\tP@1: {:.5f}\tP@5: {:.5f}\tP@10: {:.5f}\tP@15: {:.5f}'.format(p1, p5, p10, p15))
    print('\tR@1: {:.5f}\tR@5: {:.5f}\tR@10: {:.5f}\tR@15: {:.5f}'.format(r1, r5, r10, r15))
    # train loss
    my_draw_loss(train_losses, loss_fig_name, result_folder)
    # test pearson
    # draw_pearson(test_pearsons, pearson_fig_name, result_folder)
    # # test mse
    # my_draw_mse(test_MSE, test_rMSE, MSE_fig_name, result_folder)


if __name__ == '__main__':

    # 参数定义
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0,
                        help='0:A_3Net')
    parser.add_argument('--metric', type=int, required=False, default=0, help='0: cosine, 1: jaccard, 2: euclidean')
    parser.add_argument('--train_batch', type=int, required=False, default=10, help='Batch size training set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.001, help='weight_decay')
    parser.add_argument('--lamb', type=float, required=False, default=0.03, help='LAMBDA')
    parser.add_argument('--epoch', type=int, required=False, default=3000, help='Number of epoch')
    parser.add_argument('--knn', type=int, required=False, default=10, help='Number of KNN')
    parser.add_argument('--log_interval', type=int, required=False, default=40, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda:0', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=200, help='features dimensions of drugs and side effects')
    parser.add_argument('--eps', type=float, required=False, default=0.5, help='regard 0 as eps when training')

    parser.add_argument('--tenfold', action='store_true', default=False, help='use 10 folds Cross-validation ')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model and features')
    parser.add_argument('--DF', action='store_true', default=False, help='use DF decoder')
    parser.add_argument('--not_FC', action='store_true', default=False, help='not use Linear layers')
    parser.add_argument('--PCA', action='store_true', default=False, help='use PCA')
    # 属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中，使用即可
    args = parser.parse_args()

    modeling = [A3_Net][args.model]
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

    # 加载预处理数据
    dataset = 'drug_sideEffect'


    processed_mask_mat = mask_mat_file
    if not os.path.isfile(processed_mask_mat):
        print('Missing data_WS files, generating......')
        generateMat()

    ######################################################################################
    result_folder = './result_WS/'

    if tenfold:
        result_folder += '10WS_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)
    else:
        result_folder += '1WS_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)

    isExist = os.path.exists(result_folder)
    if not isExist:
        os.makedirs(result_folder)
    else:
        shutil.rmtree(result_folder)
        os.makedirs(result_folder)
    ######################################################################################

    result_log = result_folder + '/' + modeling.__name__ + '_result.csv'
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    with open(result_log, 'w', newline='') as f:
        fieldnames = ['pearson', 'rMSE', 'spearman', 'MAE', 'auc_all', 'aupr_all', 'drugAUC', 'drugAUPR', 'MAP', 'nDCG',
                      'P1', 'P5', 'P10', 'P15', 'R1', 'R5', 'R10', 'R15']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # 保存十折交叉后的所有预测值,即整个非零值部分
    pred_result = np.zeros(raw.shape)
    # np.savetxt(result_folder + '/pred_result.csv', pred_result, fmt='%.4f')
    pred_result = pd.DataFrame(pred_result)
    pred_result.to_csv(result_folder + '/pred_result.csv', header=False, index=False)

    start = datetime.datetime.now()
    # (id, frequencyMat) = next(split_data())
    for (id, frequencyMat, mask) in split_data(tenfold):

        start_ = datetime.datetime.now()
        main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name,
             frequencyMat, id + 1, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca)

        end_ = datetime.datetime.now()
        print('本次运行时间：{}\t'.format(end_ - start_))
    end = datetime.datetime.now()

    # 写入均值
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

    print('运行时间：{}\t'.format(end - start))
