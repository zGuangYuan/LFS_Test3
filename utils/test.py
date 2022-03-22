import pandas as pd
#import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import model
import matplotlib
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import roc_auc_score,f1_score,roc_curve,auc,classification_report
from PIL import Image,ImageDraw,ImageFont
from numpy.random import seed
from tensorflow.keras.models import Sequential,Model,load_model

seed(42)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
'''全局变量'''
# 训练集的图像路径
Bmode_imgs_path_root = '../testDataset_part/Bmode/'
Swe_imgs_path_root = '../testDataset_part/swe/'
radiomics_csv_path = '../testDataset_part/Radiomics/radiomics_part.csv'

print(tf.config.experimental.list_logical_devices())
print(tf.test.is_gpu_available())






test = pd.read_csv(radiomics_csv_path,header=0)



array_test = test.values


test_attr = array_test[:, 2:466]

test_label = array_test[:, 466].astype('int')



test_attr = test_attr.astype(np.float64) # 把数据抓换成float类型
test_attr = StandardScaler().fit_transform(test_attr) #数据进行标准化处理,转换成ndarray类型


m,n = test_attr.shape
width,height = 224,224





def loadTestBmodeImg(dstDir=Bmode_imgs_path_root):
    digitsFile = [dstDir+'/'+fn for fn in os.listdir(dstDir) if fn.endswith('.jpg')]

    #print(digitsFile)
    # 重新进行排序
    # 按第一列的序号ID1进行排序
    digitsFile.sort(key=lambda x: int(x.split('//')[1].split('-')[0]))

    #digitsFile.sort(key=lambda fn:int(os.path.basename(fn)[:5]))
    digitsData = []
    for fn in digitsFile:
        with Image.open(fn) as im:
            data = [im.getpixel((w,h)) for w in range(width) for h in range(height)]
            digitsData.append(data)
    return(digitsData)





def loadTestSWEImg(dstDir=Swe_imgs_path_root):
    digitsFile = [dstDir+'/'+fn for fn in os.listdir(dstDir) if fn.endswith('.jpg')]
    # 排序
    digitsFile.sort(key=lambda x: int(x.split('//')[1].split('-')[0]))
    #digitsFile.sort(key=lambda fn:int(os.path.basename(fn)[:5]))
    digitsData = []
    for fn in digitsFile:
        with Image.open(fn) as im:
            data = [im.getpixel((w,h)) for w in range(width) for h in range(height)]
            digitsData.append(data)
    return(digitsData)


testBmodeData = loadTestBmodeImg()


testSWEData = loadTestSWEImg()




test_B= np.array(testBmodeData)
test_B = test_B.reshape(test_B.shape[0],224,224,3)/255





test_S= np.array(testSWEData)
test_S = test_S.reshape(test_S.shape[0],224,224,3)/255



test_label_encoded=to_categorical(test_label)




filepath = '../model/rbs_weights.hdf5'



best_model = load_model(filepath)


score = best_model.evaluate([test_attr,test_B,test_S],test_label_encoded)
predict = best_model.predict([test_attr,test_B,test_S])

predict_classes = np.argmax(predict,axis=1)
fpr,tpr,thresholds = roc_curve(test_label,predict[:,1])



print('loss:%.3f,Accuracy:%.3f' % (score[0], score[1]))
print("AUC For Model :{:.3f}".format(auc(fpr, tpr)))
print(classification_report(test_label,predict_classes,digits=3,target_names=["0","1"]),'\n')



def roc_curve_and_score(y_ture, pred_proba):
    fpr, tpr, _ = roc_curve(y_ture.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_ture.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc


plt.figure(figsize=(10,4))
fpr, tpr, roc_auc = roc_curve_and_score(test_label, predict[:,1])
plt.subplot(1,2,1)
plt.plot(fpr, tpr, color='green', lw=2,label=' AUC={0:.3f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title("ROC")
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')



## 计算混淆矩阵并可视化
plt.subplot(1,2,2)
plt.title("confusion_matrix")

print(test_label)
print(predict_classes)


conf_mat = confusion_matrix(test_label,predict_classes)
df_cm = pd.DataFrame(conf_mat,index=[0,1],columns=[0,1])

heatmap = sns.heatmap(df_cm,annot=True,fmt="d",cmap="YlGnBu")
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha="right")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha="right")
plt.ylabel("Ture label")
plt.xlabel("Predicted label")
plt.show()