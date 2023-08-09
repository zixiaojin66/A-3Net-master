import matplotlib.pyplot as plt
import numpy as np

data1= np.load(file=r'data1.npy')
data2= np.load(file=r'data2.npy')
data3= np.load(file=r'data3.npy')
data4= np.load(file=r'data4.npy')
data5= np.load(file=r'data5.npy')
data6= np.load(file=r'data6.npy')
data1=np.mean(data1,axis=0)
data2=np.mean(data2,axis=0)
data3=np.mean(data3,axis=0)
data4=np.mean(data4,axis=0)
data5=np.mean(data5,axis=0)
data6=np.mean(data6,axis=0)
Att=[]
for i in range(1059):
    a = []
    for j in range(1059):
        x1=data1[i][j]
        x2 = data2[i][j]
        x3 = data3[i][j]
        x4 = data4[i][j]
        x5 = data5[i][j]
        x6 = data6[i][j]
        mean=(x1+x2+x3+x4+x5+x6)/6
        a.append(mean*10000-9)
    Att.append(a)

import seaborn as sns
sns.heatmap(data=Att)
plt.show()