# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:53:01 2017

@author: Administrator
"""
'''###################################################################################
#其算法的描述为：                                                                         
#1）计算测试数据与各个训练数据之间的距离；                                                 
#2）按照距离的递增关系进行排序；                                                          
#3）选取距离最小的K个点；                                                                 
#4）确定前K个点所在类别的出现频率；                                                        
#5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。                                 
######################################################################################
#KNN算法优点：
#1.简单、有效。
#2.重新训练的代价较低（类别体系的变化和训练集的变化，在Web环境和电子商务应用中是很常见的）。
#3.计算时间和空间线性于训练集的规模（在一些场合不算太大）。
#4.由于KNN方法主要靠周围有限的邻近的样本，而不是靠判别类域的方法来确定所属类别的，因此对于
   类域的交叉或重叠较多的待分样本集来说，KNN方法较其他方法更为适合。
#5.该算法比较适用于样本容量比较大的类域的自动分类，而那些样本容量较小的类域采用这种算法比较
   容易产生误分。
#####################################################################################
#KNN算法注意点：
#1、knn算法的训练集数据必须要相对公平，各个类型的数据数量应该是平均的，否则当A数据由1000个 
    B数据由100个，到时无论如何A数据的样本还是占优的。
#2、knn算法如果纯粹凭借分类的多少做判断，还是可以继续优化的，比如近的数据的权重可以设大，
    最后根据所有的类型权重和进行比较，而不是单纯的凭借数量。
#3、knn算法的缺点是计算量大，这个从程序中也应该看得出来，里面每个测试数据都要计算到所有的训
    练集数据之间的欧式距离，时间复杂度就已经为O(n*n)，如果真实数据的n非常大，这个算法的开销
    的确态度，所以KNN不适合大规模数据量的分类

'''
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import itertools

'''
此部分为数据的预处理
'''
iris=datasets.load_iris()#导入sklearn内置的花的数据集，共150个数据记录，每条数据包括x（四个属性）与y
iris_x=iris.data#x的4个分量分别为花的萼片长、萼片宽、花瓣的长，花瓣的宽一共4个属性
iris_y=iris.target#y为花的分类，共3种：Iris setosa, Iris virginica and Iris versicolor

indices=np.random.permutation(len(iris_x))#获得一个打乱的index（0-149）的列表例如[32,24,5.....]，为了后面选取训练数据的随机性

iris_x_train=iris_x[indices[:-15]]#选取随机数列的前140个为训练数据
iris_y_trian=iris_y[indices[:-15]]#相对应得标签

iris_x_test=iris_x[indices[-15:]]#选取随机数列的后10个为测试数据
iris_y_test=iris_y[indices[-15:]]


'''
此部分为数据可视化部分
'''
data=iris
features = data['data']
feature_names = data['feature_names']
target = data['target']
labels = data['target_names'][data['target']]
feature_names_2 = []
#排列组合  
feature_names_2= list(itertools.combinations(feature_names,2))
print(len(feature_names_2))
for i in feature_names_2:
    print(i)
    
plt.figure(1)
for i,k in enumerate(feature_names_2):
    index1 = feature_names.index(k[0])
    index2 = feature_names.index(k[1])
    plt.subplot(2,3,1+i)
    for t,marker,c in zip(range(3),">ox","rgb"):        
        plt.scatter(features[target==t,index1],features[target==t,index2],marker=marker,c=c)
        plt.xlabel(k[0])
        plt.ylabel(k[1])
        plt.xticks([])
        plt.yticks([])#给y轴标记坐标
        #plt.autoscale()#自动调整最佳比例
        #plt.tight_layout()#紧凑显示图片，居中显示      
plt.show()



'''
此部分为定义了KNN的类
'''
class K_nearest_Neighbor(object):#首先定义一个处理KNN的类
    '''a KNN classifier with L2 distance'''
    def __init__(self):
        pass
    
    def train(self,X,y):
        '''
        训练分类器，对于K-近邻来说，这只是记住训练数据 
        输入：x维度为（num_train,D）
              y[i]是x[i]的标签
        '''
        self.X_train=X
        self.y_train=y
    
    def predict(self,X,k=1,num_loops=0):
        '''
        利用分类器预测测试数据的标签，num_loops0，1，2代表着用不同的方法实现距离的计算
        '''
        if num_loops==0:
            dists=self.compute_distance_no_loops(X)
        elif num_loops==1:
            dists=self.compute_distance_one_loops(X)
        elif num_loops==2:
            dists=self.compute_distance_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops'%num_loops)
        return self.predict_lables(dists,k=k)
            
    def compute_distance_no_loops(self,X):
        '''
        这是利用矩阵的基本性质进行计算距离
        '''
        num_test=X.shape[0]
        num_train=self.X_train.shape[0]
        dists=np.zeros((num_test,num_train))
        dists=np.multiply(np.dot(X,self.X_train.T),-2)#这个是-2*x1*x2，相当于平方项的中间项
        sq1=np.sum(np.square(X),axis=1,keepdims=True)#这个为x1^2
        sq2=np.sum(np.square(self.X_train),axis=1)#这个为x2^2
        dists=np.add(dists,sq1)
        dists=np.add(dists,sq2)
        dists=np.sqrt(dists)#获得每一个测试样本与训练样本之间的L2距离
        return dists
    
    def compute_distance_one_loops(self,X):
        '''
        利用了一个循环来实现距离
        '''
        num_test=X.shape[0]
        num_train=self.X_train.shape[0]
        dists=np.zeros((num_test,num_train))
        for i in range(num_test):
            dists[i,:]=np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis=1))
        return dists
            
    def compute_distance_two_loops(self,X):
        '''
        利用了两个循环来实现距离
        '''
        num_test=X.shape[0]
        num_train=self.X_train.shape[0]
        dists=np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j]=np.sqrt(np.sum(np.square(self.X_trian[j,:]-X[i,:])))
        return dists
    
    def predict_lables(self,dists,k=1):
        num_test=dists.shape[0]
        y_pred=np.zeros(num_test)
        for i in range(num_test):
            closest_y=[]
            closest_y=self.y_train[np.argsort(dists[i,:])[:k]]#选取与测试样本距离最近的K个训练样本
            y_pred[i]=np.argmax(np.bincount(closest_y))#根据K个训练样本的label值对测试样本的结果进行投票
        return y_pred#返回投票值也就是预测的label值
    
'''
主程序
'''       
knn=K_nearest_Neighbor()
knn.train( iris_x_train,iris_y_trian)
print('预测样本种类：',knn.predict(iris_x_test))
print('正确样本种类：',' '.join(str(iris_y_test)))
acc=[1 if knn.predict(iris_x_test)[i]==iris_y_test[i] else 0 for i in range(15)]
acc_ture=sum(acc)/15.*100
print('准确率为{}%'.format(acc_ture))

