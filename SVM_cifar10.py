'''
include three classes:
class get_data_and_pre_dataset:   get_train_data:  this function is to get train_data from file
                                  get_test_data:   this function is to get test_data from file
                                  Visualdata:      this function is to Visualize data


class SVM:  train:           this function is to train data and get the loss and updata W(parameter)
            compute_loss:    this function is to compute loss according to the train data
            test:            this function is to predict labels according to the input data
        
class train_test :    set_train_valid_test:get the strain_data, valid_data , test_data
                                     train:train the model
                                     test:test the input data
'''



import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

'''
'''
class SVM:
    def __init__(self):
        self.W=None

    def train(self,X_train,Y_train,iter_nums,batch_num,learning_rate,alter,reg):
        '''
        
        :param X_train: 训练数据
        :param Y_train: 训练数据的labels值
        :param iter_nums: 迭代次数
        :param batch_num: 设定的batch集中数据的多少
        :param learning_rate:学习率
        :param alter: 冗余量
        :param reg: 正则化参数
        :return:
        '''
        X_train_num=X_train.shape[0]
        X_train_dim=X_train.shape[1]
        class_num=np.max(Y_train)+1
        loss_history=[]
        print('\nlearning_rate:{},   reg:{},   alter:{}'.format(learning_rate,reg,alter))
        if self.W==None:
            self.W=np.random.randn(class_num,X_train_dim)
        for iter_num in range(iter_nums):
            sample_id=np.random.choice(X_train_num,batch_num,replace=False)
            X_train_batch=X_train[sample_id]
            Y_train_batch=Y_train[sample_id]
            loss,gred=self.compute_loss(X_train_batch,Y_train_batch,alter,reg)
            if iter_num %500==0:
               print('     iter_num:{:4d}, loss:{}'.format(iter_num,loss))
            loss_history.append(loss)
            self.W-=learning_rate*gred
        return loss_history



    def compute_loss(self,X_train_batch,Y_train_batch,alter,reg):
        '''

        :param X_train_batch: batch集的图片数据
        :param Y_train_batch: batch集的图片类别
        :param alter: 冗余量
        :param reg: 正则化参数
        :return: 损失值与梯度
        '''
        score=X_train_batch.dot(self.W.T)
        X_train_batch_num=X_train_batch.shape[0]
        correct_score=score[range(X_train_batch_num),Y_train_batch]
        margins=score-correct_score[:,np.newaxis]+alter
        margins=np.maximum(0,margins)
        margins[range(X_train_batch_num),Y_train_batch]=0
        loss=np.sum(margins)/X_train_batch_num+0.5*reg*np.sum(self.W*self.W)
        ground_true=np.zeros(score.shape)
        ground_true[margins>0]=1
        ground_true[range(X_train_batch_num),Y_train_batch]-=np.sum(ground_true,axis=1)
        gred=ground_true.T.dot(X_train_batch)/X_train_batch_num+0.5*reg*self.W
        return loss,gred

    def predict(self,X_test):
        '''

        :param X_test: 输入的测试数据
        :return:
        '''
        score=X_test.dot(self.W.T)
        result=np.zeros(X_test.shape[0])
        result=np.argmax(score,axis=1)
        return result


class get_data_and_pre_dataset:

    def get_train_data(self,ROOT):
        '''

        :param ROOT:文件的根目录
        :return:
        '''
        X_train_data=[]
        Y_train_data=[]
        for i in range(1,6):
            print('read train_data from data_batch_{}'.format(i))
            filename=os.path.join(ROOT,'data_batch_{}'.format(i))
            with open(filename,'rb')as f:
                datadict=pickle.load(f,encoding='bytes')
                x_data=datadict[b'data']
                y_data=datadict[b'labels']
                X_train_data.append(x_data)
                Y_train_data.append(y_data)
        X_train_data=np.concatenate(X_train_data)
        Y_train_data=np.concatenate(Y_train_data)
        print('finish reading batch_data from file')
        return X_train_data,Y_train_data

    def get_test_data(self,ROOT):
        '''

        :param ROOT:文件的根目录
        :return:
        '''
        print('\n'+'read test_data from test_batch')
        filename=os.path.join(ROOT,'test_batch')
        with open(filename,'rb')as f :
            datadict=pickle.load(f,encoding='bytes')
            X_test_data=datadict[b'data']
            Y_test_data=datadict[b'labels']
        print('finish reading test_data from test_batch')
        return X_test_data,Y_test_data

    def Visualdata(self,X,Y):
        '''

        :param X:输入图片的像素数据
        :param Y:输入图片的类别
        :return:
        '''
        class_num=np.max(Y)+1
        plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        class_name=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        sample_num=10
        for idx,name in enumerate(class_name):
            class_id=np.flatnonzero(Y==idx)
            class_sample=np.random.choice(class_id,sample_num,replace=False)
            x_plot_data=X[class_sample,:].reshape(sample_num,3,32,32).transpose(0,2,3,1).astype('float')
            for i in range(8):
                plt.subplot(sample_num,class_num,i*sample_num+idx+1)
                plt.imshow(x_plot_data[i].astype('uint8'))
                plt.axis('off')
                if i ==0:
                    plt.title(name)
        print('\n'+'when you turn off the firgue,the program will continue ')
        plt.show()


class train_test:
    def set_train_valid_test(self,X_train,Y_train,X_test,Y_test):
        '''

        :param X_train:从文件中读取训练图片像素数据
        :param Y_train: 从文件中读取训练图片类别
        :param X_test: 从文件中读取的测试图片像素数据
        :param Y_test: 从文件中读取的测试图片的类别
        :return:
        '''
        train_num=49000
        valid_num=1000
        x_train=X_train[:train_num,:]
        x_valid=X_train[train_num:train_num+valid_num,:]
        y_train=Y_train[:train_num]
        y_valid=Y_train[train_num:train_num+valid_num]

        image_mean=np.mean(X_train,axis=0)
        x_train=x_train-image_mean
        x_test=X_test-image_mean
        x_valid=x_valid-image_mean

        x_train=np.hstack([x_train,np.ones((x_train.shape[0],1))])
        x_test=np.hstack([x_test,np.ones((x_test.shape[0],1))])
        x_valid=np.hstack([x_valid,np.ones((x_valid.shape[0],1))])

        return x_train,y_train,x_valid,y_valid,x_test,Y_test


    def train(self,X_train,Y_train,X_valid,Y_valid):
        learning_rate=[1e-7, 5e-5]
        reg=[5, 1]
        best_acc=0
        best_parameter=None
        for i in learning_rate:
            for j in reg:
                train_svm=SVM()
                train_svm.train(X_train,Y_train,1500,200,i,1,j)
                y_pred=train_svm.predict(X_valid)
                accuracy=np.mean(y_pred==Y_valid)
                if accuracy>best_acc:
                    best_acc=accuracy
                    best_parameter=(i,j)
        print('-----------------training result------------------')
        print('best parameter：  training_rate:{},  reg:{}'.format(best_parameter[0],best_parameter[1]))
        print('best accuracy：{}'.format(best_acc))



        return best_parameter

    def test(self,X_train,Y_train,best_parameter,X_test,Y_test):
        test_svm=SVM()
        test_svm.train(X_train,Y_train,1500,200,best_parameter[0],1,best_parameter[1])
        y_pred = test_svm.predict(X_test)
        accuracy = np.mean(y_pred == Y_test)
        print('-----------------test result------------------')
        print('Accuracy achieved during cross-validation: {}'.format(np.mean(accuracy)))

step1=get_data_and_pre_dataset()
X_train,Y_train=step1.get_train_data('./cifar-10-batches-py')
X_test,Y_test=step1.get_test_data('./cifar-10-batches-py')
step1.Visualdata(X_train,Y_train)

step2=train_test()
print('\n-----------------training process------------------')
x_train,y_train,x_valid,y_valid,x_test,y_test=step2.set_train_valid_test(X_train,Y_train,X_test,Y_test)
best_parameter=step2.train(x_train,y_train,x_valid,y_valid)
print('\n-----------------test process------------------')
step2.test(x_train,y_train,best_parameter,x_test,y_test)