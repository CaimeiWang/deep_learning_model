#加载需要用到的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import *
from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score



pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)
pd.set_option('display.max_colwidth', 100000)
#数据处理
data=pd.read_csv('train.csv')
data=data.fillna(0)  #把nan都填0，可以改其它值****
#data=data.dropna(axis=0,how='any') #也可以去掉有空的行****
data=data.drop(columns='id', inplace=False)
flag=data['flag']
label=[]
for i in flag:
    label.append([i])
label=np.array(label)
data=data.drop(columns='flag', inplace=False)
data=data.values
data=scale(data,axis=0)   #按行标准化
pca=PCA(n_components=30)  #加载PCA算法，设置降维后主成分数目为30 #30可以调整****
pca_data=pca.fit_transform(data)  #对样本进行降维
pca_data=np.concatenate((label,pca_data),axis=1)
shuffle_num= np.random.permutation(len(pca_data)) #打乱数据集顺序
pca_data=np.array(pca_data)[shuffle_num]
# #
# #将特征和标签对应，并把标签制成独热型
train=[]
train_label=[]
val=[]
val_label=[]
test=[]
test_label=[]
test_label1=[]
for i in range(pca_data.shape[0]): #按8:1:1划分数据集
    if i<(8*pca_data.shape[0]//10):
        train.append(pca_data[i][1:len(pca_data[i])])
        if int(pca_data[i][0])==0:
            train_label.append([1,0])
        elif int(pca_data[i][0])==1:
            train_label.append([0,1])
    elif i in range(8*pca_data.shape[0]//10,9*pca_data.shape[0]//10):
        val.append(pca_data[i][1:len(pca_data[i])])
        if int(pca_data[i][0])==0:
            val_label.append([1,0])
        elif int(pca_data[i][0])==1:
            val_label.append([0,1])
    else:
        test.append(pca_data[i][1:len(pca_data[i])])
        if int(pca_data[i][0])==0:
            test_label.append([1,0])
            test_label1.append(0)
        elif int(pca_data[i][0])==1:
            test_label.append([0,1])
            test_label1.append(1)
#
train=np.array(train)
train_label=np.array(train_label)
val=np.array(val)
val_label=np.array(val_label)
test=np.array(test)
test_label=np.array(test_label)
test_label1=np.array(test_label1)

# # # np.save('test.npy',test)
# # # np.save('test_label.npy',test_label)
# # # np.save('test_label1.npy',test_label1)
# # #构建网络
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(30,))) #32是神经元个数，此数可以调整***
model.add(Dropout(0.5))   # 0.5是神经元舍弃比，次数可以调整****
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))  #一个dense是一层，层数可以增加和减少****
# model.add(Dropout(0.5))
model.add(Dense(2, activation ='softmax'))
model.summary() #显示网络

#编译网络结构
model.compile(loss='categorical_crossentropy',
              optimizer=sgd(),
              metrics=['accuracy'])
######这个数据集的主要问题就是过拟合，可以减少主成分数目，减少神经元数目，减少网络层数，增加dropout，减少批处理尺寸
#训练网咯
epochs=100 #迭代次数，可以调整****
batch_size=32 #批处理尺寸，可以调整****
history = model.fit(train,train_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, #0，不显示，1，进度条，2，文字
                    validation_data=(val,val_label))


def show_train_history(train_history, train_metrics, validation_metrics):
    plt.plot(train_history.history[train_metrics],label='train')
    plt.plot(train_history.history[validation_metrics],label='validation')
    #plt.title('Train History')
    plt.ylabel(train_metrics)
    plt.xlabel('Epoch')
    plt.legend()


# 显示训练过程
def plot(history):
    plt.figure(figsize=(6.5,6))
    plt.subplot(2,1,1)
    show_train_history(history,'loss','val_loss')
    plt.subplot(2,1,2)
    show_train_history(history, 'accuracy', 'val_accuracy')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

plot(history)
score = model.evaluate(test,test_label,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
#
def plot_ROC(model, x_test, y_test):  # 绘制ROC和AUC，来判断模型的好坏
    y_pro = model.predict_proba(x_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, y_pro[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    plt.plot(false_positive_rate, recall, 'b', label='AUC=%0.2f'%roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('false_positive_rate')
    plt.show()


#测试集
test_data=pd.read_csv('test.csv')
id=test_data['id']
test_data=test_data.fillna(0)
test_data=test_data.drop(columns='id', inplace=False)
test_data=test_data.drop(columns='flag', inplace=False)
test_data=test_data.values
test_data=scale(test_data,axis=0)   #按行标准化
test_pca=PCA(n_components=30)  #加载PCA算法，设置降维后主成分数目为30
pca_test=test_pca.fit_transform(test_data)  #对样本进行降维
y_pro=model.predict_proba(pca_test)
y_pre=y_pro[:,1]
frame={'id':id,
       'predict':list(y_pre)}
df=pd.DataFrame(frame)
df.to_csv('predict.csv')


#保存模,
#model.save_weights('deep_learning_model1.h5')