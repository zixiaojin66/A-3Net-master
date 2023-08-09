import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
#为呈现金字塔型预测真实频率对比图，我们将真实频率取负
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_excel(r"C:\Users\31849\Desktop\compare.xlsx")
df_male = df.groupby(by='leixing').get_group('Pred')
list_male = df_male['Frequency'].values.tolist()  # 将ndarray转换为list
df_female = df.groupby(by='leixing').get_group('True_f')
list_female = df_female['Frequency'].values.tolist()  # 将ndarray转换为list
df_age = df.groupby('Side-Effect').sum()
count = df_age.shape[0]
y = np.arange(1, 11)
labels = []
for i in range(count):
    age = df_age.index[i]
    labels.append(age)
print(labels)
fig = plt.figure()
ax = fig.add_subplot(111)

# 创建 colormap
cmap = cm.get_cmap('RdBu')

# 设置数据范围
vmin = -5
vmax = 5

# 绘制人口金字塔图，并设置颜色
patches = []
used_colors = set()  # 用于跟踪已经添加到标签的颜色
for i in range(count):
    color = cmap((list_male[i] - vmin) / (vmax - vmin))
    colorr = cmap(abs(list_female[i] - vmin) / (vmax - vmin))
    ax.barh(y[i], list_male[i], tick_label=labels[i], color=color)
    ax.barh(y[i], list_female[i], tick_label=labels[i], color=colorr)

    # 添加颜色标签
    if color not in used_colors:
        patch1 = mpatches.Patch(color=color, label='Pred:{}'.format(list_male[i]))
        patches.append(patch1)
        used_colors.add(color)
    if colorr not in used_colors:
        patch2 = mpatches.Patch(color=colorr, label='True:{}'.format(list_male[i]))
        patches.append(patch2)
        used_colors.add(colorr)

ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper left')


ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['5', '4', '3', '2', '1', '0', '1', '2', '3', '4', '5'])
ax.set_xlabel('Frequency')
ax.set_yticks([1,2,3,4,5,6,7,8,9,10])
ax.set_yticklabels(labels,fontsize=9)
ax.set_title('True              Sumatriptan               Pred')
# 设置标签字体大小
plt.show()