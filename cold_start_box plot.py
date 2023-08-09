import pandas as pd
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_excel(r'cold_pred.xlsx')
data = np.array(df).T
print(data)

# 读取truth数据
truth = pd.read_excel('cold_truth.xlsx')
truth = np.array(truth)

# 将预测数据按照真实类别分组
pred_all = [[] for _ in range(6)]
for i in range(994):
    if truth[i] == 0.0:
        pred_all[0].append(data[0][i])
    elif truth[i] == 1.0:
        pred_all[1].append(data[0][i])
    elif truth[i] == 2.0:
        pred_all[2].append(data[0][i])
    elif truth[i] == 3.0:
        pred_all[3].append(data[0][i])
    elif truth[i] == 4.0:
        pred_all[4].append(data[0][i])
    elif truth[i] == 5.0:
        pred_all[5].append(data[0][i])

## AQI（图一）##
plt.subplot(1,6,1)      #将四个箱线图画在一行中

plt.grid(linestyle="--", alpha=0.3)    #绘制图像底部虚线.   (linestyle=ls,  markerfacecolor=mfc)
plt.boxplot(
            pred_all[0],
            patch_artist=True,
            showmeans=True,    #显示均值点
            whis=8,
            widths=0.2,      #箱体宽度
            boxprops={'color': 'black', 'facecolor': '#FFDF00'},   #设置箱体属性
            flierprops={'marker': 'o', 'mfc': 'red', 'color': 'black'},    #设置异常值属性
            meanprops={'marker': '+', 'mfc': 'black'},      #设置均值点属性
            medianprops={'ls': '--', 'color': 'orange'},      #设置中位数属性
            whiskerprops={'ls': '--', 'mfc': 'red', 'color': 'black'},  #设置触须属性
                   )

plt.title('Freq=0')    #子标题
plt.xticks([])   #关闭x轴坐标显示



## PM 2.5（图二）##
plt.subplot(1,6,2)

plt.grid(linestyle="--", alpha=0.3)
plt.boxplot(
            pred_all[1],
            patch_artist=True,
            showmeans=True,
            whis=8,
            widths=0.2,
            boxprops={'color': 'black', 'facecolor': '#2C4096'},
            flierprops={'marker': 'o', 'mfc': 'red', 'color': 'black'},
            meanprops={'marker': '+', 'mfc': 'black'},
            medianprops={'ls': '--', 'color': 'orange'},
            whiskerprops={'ls': '--', 'mfc': 'red', 'color': 'black'}
            )
plt.title('Freq=1')
plt.xticks([])



## PM 10（图三）##
plt.subplot(1,6,3)

plt.grid(linestyle="--", alpha=0.3)
plt.boxplot(pred_all[2],
            patch_artist=True,
            showmeans=True,
            whis=8,
            widths=0.2,
            boxprops={'color': 'black', 'facecolor': '#019000'},
            flierprops={'marker': 'o', 'mfc': 'red', 'color': 'black'},
            meanprops={'marker': '+', 'mfc': 'black'},
            medianprops={'ls': '--', 'color': 'orange'},
            whiskerprops={'ls': '--', 'mfc': 'red', 'color': 'black'}
            )
plt.title('Freq=2')
plt.xticks([])



## So2（图四）##
plt.subplot(1,6,4)
plt.grid(linestyle="--", alpha=0.3)
plt.boxplot( pred_all[3],
            patch_artist=True,
            showmeans=True,
            whis=8,
            widths=0.2,
            boxprops={'color': 'black', 'facecolor': '#D22C2C'},
            flierprops={'marker': 'o', 'mfc': 'red', 'color': 'black'},
            meanprops={'marker': '+', 'mfc': 'black'},
            medianprops={'ls': '--', 'color': 'orange'},
            whiskerprops={'ls': '--', 'mfc': 'red', 'color': 'black'}
            )
plt.title('Freq=3')
plt.xticks([])
## So2（图四）##
plt.subplot(1,6,5)
plt.grid(linestyle="--", alpha=0.3)
plt.boxplot( pred_all[4],
            patch_artist=True,
            showmeans=True,
            whis=8,
            widths=0.2,
            boxprops={'color': 'black', 'facecolor': '#2C4096'},
            flierprops={'marker': 'o', 'mfc': 'red', 'color': 'black'},
            meanprops={'marker': '+', 'mfc': 'black'},
            medianprops={'ls': '--', 'color': 'orange'},
            whiskerprops={'ls': '--', 'mfc': 'red', 'color': 'black'}
            )
plt.title('Freq=4')
plt.xticks([])
## So2（图四）##
plt.subplot(1,6,6)
plt.grid(linestyle="--", alpha=0.3)
plt.boxplot( pred_all[5],
            patch_artist=True,
            showmeans=True,
            whis=8,
            widths=0.2,
            boxprops={'color': 'black', 'facecolor': '#D22C2C'},
            flierprops={'marker': 'o', 'mfc': 'red', 'color': 'black'},
            meanprops={'marker': '+', 'mfc': 'black'},
            medianprops={'ls': '--', 'color': 'orange'},
            whiskerprops={'ls': '--', 'mfc': 'red', 'color': 'black'}
            )
plt.title('Freq=5')
plt.xticks([])
plt.suptitle('Gadoteridol')
       #总标题
plt.show()