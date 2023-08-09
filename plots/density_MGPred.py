import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df1=pd.read_excel(r"C:\Users\31849\Desktop\truth.xlsx")
df2=pd.read_excel(r"C:\Users\31849\Desktop\pred.xlsx")
data_label=list(df1.iloc[:,0])
data_pred=list(df2.iloc[:,0])

a=data_pred
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]
frequencyMat=data_label
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




for i in range(745500):
        if frequencyMat[i]==1:
            b.append(a[i])
for i in range(745500):
        if frequencyMat[i]==2:
            c.append(a[i])
for i in range(745500):
        if frequencyMat[i]==3:
            d.append(a[i])
for i in range(745500):
        if frequencyMat[i]==4:
            e.append(a[i])
for i in range(745500):
        if frequencyMat[i]==5:
            f.append(a[i])
for i in range(745500):
        if frequencyMat[i]==0:
            g.append(a[i])
#得到对应标签的所有值
#画出核密度估计曲线
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 使用scipy库中的gaussian_kde函数分别计算a和b的核密度估计曲线
kde_b = stats.gaussian_kde(b)
kde_c = stats.gaussian_kde(c)
kde_d= stats.gaussian_kde(d)
kde_e= stats.gaussian_kde(e)
kde_f= stats.gaussian_kde(f)
kde_g=stats.gaussian_kde(g)
# 定义一个函数来计算两条核密度估计曲线的差值

def diff_2(x):
    return kde_b(x) - kde_c(x)
def diff_3(x):
    return kde_c(x) - kde_d(x)
def diff_4(x):
    return kde_d(x) - kde_e(x)
def diff_5(x):
    return kde_e(x) - kde_f(x)
def diff_6(x):
    return kde_e(x) - kde_g(x)
# 使用scipy库中的optimize模块中的root_scalar函数计算两条曲线的交点

result2 = optimize.root_scalar(diff_2, bracket=[1,2.5])
result3 = optimize.root_scalar(diff_3, bracket=[1, 5])
result4 = optimize.root_scalar(diff_4, bracket=[1, 5])
result5 = optimize.root_scalar(diff_5, bracket=[1, 5])
result6 = optimize.root_scalar(diff_6,bracket=[3,4])
# 打印交点值

y2=kde_b(result2.root)[0]
y3=kde_c(result3.root)[0]
y4=kde_d(result4.root)[0]
y5=kde_e(result5.root)[0]
y6=kde_g(result6.root)[0]

import matplotlib.pyplot as plt

# 给定点的坐标

point2 = (result2.root, y2)
point3= (result3.root, y3)
point4 = (result4.root, y4)
point5 = (result5.root, y5)
point6 = (result6.root, y6)
# 绘制图像
fig, ax = plt.subplots()
color_list = ['k', 'gray', 'b', 'g', 'r','y']
sns.kdeplot(g,
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

# 画出从给定点出发的垂线
ax.plot([point6[0], point6[0]], [point6[1], 0], 'k--')
# 标出点的x轴坐标
ax.annotate(f'({round(point6[0],2)}, {round(point6[1],2)})', xy=point6, xytext=(point6[0]+0.5, point6[1]), arrowprops=dict(arrowstyle='->'))

# 设置x轴的范围和标签
ax.set_xlim(-1, 6)
ax.set_xticks(range(-1, 6))


# 添加x，y轴标签和标题
plt.xlabel("Frequency")
plt.ylabel("Density")
plt.title("Kernel Density Estimation for MGPred")
plt.legend()
# 显示图像
plt.show()