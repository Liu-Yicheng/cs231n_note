import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

'''
include three classes:
class get_data_and_pre_dataset:   get_train_data:  this function is to get train_data from file
                                  get_test_data:   this function is to get test_data from file
                                  Visualdata:      this function is to Visualize data


class softmax:  train:           this function is to train data and get the loss and updata W(parameter)
                compute_loss:    this function is to compute loss according to the train data
                test:            this function is to predict labels according to the input data

class train_test :    set_train_valid_test:get the strain_data, valid_data , test_data
                                     train:train the model
                                     test:test the input data
'''
class  softmax:
    def __init__(self):
        self.W=None

    def train(self,X_train,Y_train,iter_nums,batch_num,learning_rate,reg):
        X_train_num,X_train_dim=X_train.shape
        class_num=np.max(Y_train)+1
        loss_history=[]
        if self.W==None:
            self.W=0.01*np.random.randn(X_train_dim,class_num)
        print('\nlearning_rate:{}  reg:{}'.format(learning_rate, reg))
        for iter_num in range(iter_nums):

            batch_id=np.random.choice(X_train_num,batch_num,replace=False)
            X_batch=X_train[batch_id,:]
            Y_batch=Y_train[batch_id]
            loss,dw=self.softmax_loss_vectorized(X_batch, Y_batch, reg)
            self.W-=(learning_rate-iter_num*6e-9)*dw

            loss_history.append(loss)
            if  iter_num%500==0:
                print("{}:  {}".format(iter_num,loss))


    def predict(self,X_test):
        score = X_test.dot(self.W)
        result = np.zeros(X_test.shape[0])
        result = np.argmax(score, axis=1)
        return result


    def softmax_loss_vectorized(self, X, y, reg):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        # Initialize the loss and gradient to zero.



        dW = np.zeros_like(self.W)    # D by C
        num_train, dim = X.shape

        f = X.dot(self.W)    # N by C
        f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1
        prob = np.exp(f-f_max) / np.sum(np.exp(f-f_max), axis=1, keepdims=True)

        y_trueClass = np.zeros_like(prob)
        y_trueClass[range(num_train), y] = 1.0    # N by C
        loss = -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(self.W * self.W)#向量化直接操作即可
        dW = -np.dot(X.T, y_trueClass - prob) / num_train + reg * self.W

        return loss, dW



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
        learning_rate=[5e-5]
        reg=[4,4.2,4.5,4.7]
        best_acc=0
        best_parameter=None
        for i in learning_rate:
            for j in reg:
                train_softmax=softmax()
                train_softmax.train(X_train,Y_train,2500,200,i,j)
                y_pred=train_softmax.predict(X_valid)
                accuracy=np.mean(y_pred==Y_valid)
                print(accuracy)
                if accuracy>best_acc:
                    best_acc=accuracy
                    best_parameter=(i,j)
        print('-----------------training result------------------')
        print('best parameter：  training_rate:{},  reg:{}'.format(best_parameter[0],best_parameter[1]))
        print('best accuracy：{}'.format(best_acc))



        return best_parameter

    def test(self,X_train,Y_train,best_parameter,X_test,Y_test):
        test_softmax=softmax()
        test_softmax.train(X_train,Y_train,1500,200,best_parameter[0],1,best_parameter[1])
        y_pred = test_softmax.predict(X_test)
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
