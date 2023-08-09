import torch
from Net import *
model=torch.load('new.pth')
import scipy
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
import networkx as nx
import torch
import numpy as np
import os
import torch
from scipy import stats
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from tqdm import trange
import matplotlib.pyplot as plt
current_path = os.path.dirname(os.path.abspath(__file__))
mapping={0:'未发现',1:'非常罕见',2:'罕见',3:'不频繁',4:'频繁',5:'非常频繁'}
f_p=os.path.join(current_path, 'frequencyMat.csv')
frequencyMat=np.loadtxt(f_p,delimiter=',',dtype='int')
side_effect_label = os.path.join(current_path,'side_effect_label_750.mat')
input_dim = 109
cuda_name='cuda:0'
DF=False
not_FC=False
knn=5
pca=False
metric='cosine'

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

class myDataset(InMemoryDataset):
    def __init__(self, root='/data_WS', dataset='drug_sideEffect_data',
                 drug_simles=None, frequencyMat=None,
                 transform=None, pre_transform=None, simle_graph=None, saliency_map=False):

        # root is required for save preprocessed data_WS, default is '/data_WS'
        super(myDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset
        self.dataset = dataset
        # self.similarity = similarity
        # self.raw = raw
        # self.frequencyMat = frequencyMat
        self.saliency_map = saliency_map

        if os.path.isfile(self.processed_paths[0]):
            print('Pre_processed data found: {}, loading...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre_processing...'.format(self.processed_paths[0]))
            self.process(drug_simles, frequencyMat, simle_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 返回一个包含没有处理的数据的名字的list
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):  # 返回一个包含所有处理过的数据名字的list
        return [self.dataset + '.pt']

    def download(self):  # 下载数据集函数，不需要的话直接填充pass
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # feature - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data_WS
    def process(self, drug_silmes, frequencyMat, simle_graph):
        assert (len(drug_silmes) == len(frequencyMat)), "The two lists must be the same L!"
        data_list = []
        data_len = len(drug_silmes)
        print(data_len)
        data_len = trange(data_len)
        data_len.set_description("Processing ")
        for i in data_len:
            # data_len.set_description("Processing ")
            # print('Convert SIMLES to graph: {}/{}'.format(i + 1, data_len))
            smiles = drug_silmes[i]
            labels = frequencyMat[i]
            # Convert SMILES to molecular representation using rdkit
            c_size, features, edge_index, edge_type = simle_graph[smiles]
            # print(type(edge_index), edge_index,i)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index),
                                y=torch.FloatTensor([labels]))
            GCNData.__setitem__('edge_type', torch.IntTensor(edge_type * 2 ).flatten())
            # 记录此特征矩阵x的行开始的坐标，为0；
            # 利用DataLoader读取时，返回一个(1 * batch_size)维度的tensor，代表共batch_size个x,每个x的行从x_index[i]开始
            GCNData.__setitem__('x_index', torch.LongTensor([0]))

            # 记录此SMILES对应在所有SMILES的坐标，用于计算loss时查找对应的frequencyMat的位置
            # 利用DataLoader读取时，返回一个(batch_size * 1)的二维列表
            GCNData.index = [i]  # 输出为二维列表

            # 记录每张smile_graph的原子的个数，即特征矩阵x的行数；
            # 利用DataLoader读取时，返回一个(1 * batch_size)维度的tensor，代表共batch_size个x,每个x有c_size[i]的原子
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list.append(GCNData)
            # print(data_list)
        # print(data_list[0])
        # 判断数据对象是否应该保存
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # 保存到磁盘前进行转化
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        # 将数据对象的python列表整理为内部存储格式，torch_geometric.data_WS.InMemoryDataset
        data, slices = self.collate(data_list)
        # save preprocessed data_WS
        torch.save((data, slices), self.processed_paths[0])
        pass
from torch_geometric.data import  DataLoader
id=1

train_data = myDataset(root='data_WS', dataset='drug_sideEffect_data' + str(id - 1))
test_loader = DataLoader(train_data, batch_size=1, shuffle=False)
#预测函数
def predict(model ,loader,device, sideEffectsGraph, DF, not_FC):
    total_preds = torch.Tensor()
    model.eval()
    torch.cuda.manual_seed(42)
    print('Make prediction for {} samples...'.format(1))
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()
            total_preds = torch.cat((total_preds, pred), 0)

    return total_preds
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

a=predict(model ,test_loader,device, sideEffectsGraph, DF, not_FC)
a=list(np.array(a))
num_0=[]
b=[]
c=[]
d=[]
e=[]
f=[]
frequencyMat=frequencyMat.T
import statsmodels.nonparametric.api as smnp
def kde_test(data, kernel, bw, gridsize, cut):
    """
    :param data:样本数据
    :param kernel:核函数
    :param bw:带宽
    :param gridsize:绘制拟合曲线中的离散点数；可理解为精度，会改变kde曲线的圆滑程度
    :param cut: 源代码说明——Draw the estimate to cut * bw from the extreme data points.
    :return: kde估计曲线的x、y坐标
    """
    fft = kernel == "gau"
    kde = smnp.KDEUnivariate(data)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut)
    return kde.support, kde.density



for i in range(750):
    for j in range(994):
        if frequencyMat[i+1][j]==0:
            num_0.append(a[i][j])
for i in range(750):
    for j in range(994):
        if frequencyMat[i+1][j]==1:
            b.append(a[i][j])
for i in range(750):
    for j in range(994):
        if frequencyMat[i+1][j]==2:
            c.append(a[i][j])
for i in range(750):
    for j in range(994):
        if frequencyMat[i+1][j]==3:
            d.append(a[i][j])
for i in range(750):
    for j in range(994):
        if frequencyMat[i+1][j]==4:
            e.append(a[i][j])
for i in range(750):
    for j in range(994):
        if frequencyMat[i+1][j]==5:
            f.append(a[i][j])
#得到对应标签的所有值
#画出核密度估计曲线
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 使用scipy库中的gaussian_kde函数分别计算a和b的核密度估计曲线
kde_a = stats.gaussian_kde(num_0)
kde_b = stats.gaussian_kde(b)
kde_c = stats.gaussian_kde(c)
kde_d= stats.gaussian_kde(d)
kde_e= stats.gaussian_kde(e)
kde_f= stats.gaussian_kde(f)
# 定义一个函数来计算两条核密度估计曲线的差值
def diff(x):
    return kde_a(x) - kde_b(x)
def diff_2(x):
    return kde_b(x) - kde_c(x)
def diff_3(x):
    return kde_c(x) - kde_d(x)
def diff_4(x):
    return kde_d(x) - kde_e(x)
def diff_5(x):
    return kde_e(x) - kde_f(x)
# 使用scipy库中的optimize模块中的root_scalar函数计算两条曲线的交点
result = optimize.root_scalar(diff, bracket=[0, 2])
result2 = optimize.root_scalar(diff_2, bracket=[-10,10])
result3 = optimize.root_scalar(diff_3, bracket=[1, 3])
result4 = optimize.root_scalar(diff_4, bracket=[-10, 10])
result5 = optimize.root_scalar(diff_5, bracket=[3, 5])
# 打印交点值
y=kde_a(result.root)[0]
y2=kde_b(result2.root)[0]
y3=kde_c(result3.root)[0]
y4=kde_d(result4.root)[0]
y5=kde_e(result5.root)[0]
import matplotlib.pyplot as plt

# 给定点的坐标
point = (result.root, y)
point2 = (result2.root, y2)
point3= (result3.root, y3)
point4 = (result4.root, y4)
point5 = (result5.root, y5)
# 绘制图像
fig, ax = plt.subplots()
color_list = ['k', 'gray', 'b', 'g', 'r','y']
sns.kdeplot(num_0,
            kernel='gau',
            bw="scott",
            label="frequency=0",
            color=color_list[0],
            linewidth=2
            )
sns.kdeplot(b,
            kernel='gau',
            bw="scott",
            label="frequency=1",
            color=color_list[1],
            linewidth=2
            )
sns.kdeplot(c,
            kernel='gau',
            bw="scott",
            label="frequency=2",
            color=color_list[2],
            linewidth=2
            )
sns.kdeplot(d,
            kernel='gau',
            bw="scott",
            label="frequency=3",
            color=color_list[3],
            linewidth=2
            )
sns.kdeplot(e,
            kernel='gau',
            bw="scott",
            label="frequency=4",
            color=color_list[4],
            linewidth=2
            )
sns.kdeplot(f,
            kernel='gau',
            bw="scott",
            label="frequency=5",
            color=color_list[5],
            linewidth=2
            )


# 画出从给定点出发的垂线
ax.plot([point[0], point[0]], [point[1], 0], 'k--')
# 标出点的x轴坐标
ax.annotate(f'({round(point[0],2)}, {round(point[1],2)})', xy=point, xytext=(point[0]+0.5, point[1]), arrowprops=dict(arrowstyle='->'))

# 画出从给定点出发的垂线
ax.plot([point2[0], point2[0]], [point2[1], 0], 'k--')
# 标出点的x轴坐标
ax.annotate(f'({round(point2[0],2)}, {round(point2[1],2)})', xy=point2, xytext=(point2[0]+0.5, point2[1]), arrowprops=dict(arrowstyle='->'))

# 画出从给定点出发的垂线
ax.plot([point3[0], point3[0]], [point3[1], 0], 'k--')
# 标出点的x轴坐标
ax.annotate(f'({round(point3[0],2)}, {round(point3[1],2)})', xy=point3, xytext=(point3[0]+0.5, point3[1]), arrowprops=dict(arrowstyle='->'))

# 画出从给定点出发的垂线
ax.plot([point4[0], point4[0]], [point4[1], 0], 'k--')
# 标出点的x轴坐标
ax.annotate(f'({round(point4[0],2)}, {round(point4[1],2)})', xy=point4, xytext=(point4[0]+0.5, point4[1]), arrowprops=dict(arrowstyle='->'))

# 画出从给定点出发的垂线
ax.plot([point5[0], point5[0]], [point5[1], 0], 'k--')
# 标出点的x轴坐标
ax.annotate(f'({round(point5[0],2)}, {round(point5[1],2)})', xy=point5, xytext=(point5[0]+0.5, point5[1]), arrowprops=dict(arrowstyle='->'))

# 设置x轴的范围和标签
ax.set_xlim(-1, 6)
ax.set_xticks(range(-1, 6))


# 添加x，y轴标签和标题
plt.xlabel("Frequency")
plt.ylabel("Density")
plt.title("Kernel Density Estimation for $A3Net$")
plt.legend()
# 显示图像
plt.show()