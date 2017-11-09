import os
import pickle
import numpy as np
import matplotlib.pylab as plt

class Data(object):
    def __init__(self):
        pass

    def get_train_data(self,ROOT):
        X_data=[]
        Y_data=[]
        for i in range(1,6):
            print('read train_data from data_batch_{}'.format(i))
            filename=os.path.join(ROOT,'data_batch_{}'.format(i))
            with open(filename,'rb') as f:
                datadict=pickle.load(f,encoding='bytes')
                x_data=datadict[b'data']
                y_data=datadict[b'labels']
                X_data.append(x_data)
                Y_data.append(y_data)

        X_train_data=np.concatenate(X_data)
        Y_train_data=np.concatenate(Y_data)
        print('finish reading batch_data from file')
        return X_train_data,Y_train_data

    def get_test_data(self,ROOT):
        print('\n' + 'read test_data from test_batch')
        filename=os.path.join(ROOT,'test_batch')
        with open(filename,'rb')as f:
            datadict=pickle.load(f,encoding='bytes')
            X_test_data=datadict[b'data']
            Y_test_data=datadict[b'labels']
        print('finish reading test_data from test_batch')
        return X_test_data,Y_test_data


    def Visualdata(self,x_data,Y_data):
        class_num=np.max(Y_data)+1
        X_data=x_data.reshape(x_data.shape[0],3,32,32).transpose(0,2,3,1).astype('float')
        plt.rcParams['figure.figsize'] = (15.0, 10.0)  # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        class_names=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        sample_num=10
        for class_iter,class_name in enumerate(class_names):
            class_id=np.flatnonzero(Y_data==class_iter)
            sample_id=np.random.choice(class_id,sample_num,replace=False)
            X_sample_data=X_data[sample_id,:]
            for pic_num in range(sample_num):
                plt.subplot(sample_num,class_num,pic_num*sample_num+class_iter+1)
                plt.imshow(X_sample_data[pic_num].astype('uint8'))
                plt.axis('off')
                if pic_num==0:
                    plt.title(class_name)
        print('\n' + 'when you turn off the firgue,the program will continue ')
        plt.show()

class two_layers_network():
    """
        A two-layer fully-connected neural network. The net has an input dimension of
        N, a hidden layer dimension of H, and performs classification over C classes.
        We train the network with a softmax loss function and L2 regularization on the
        weight matrices. The network uses a ReLU nonlinearity after the first fully
        connected layer.

        In other words, the network has the following architecture:

        input - fully connected layer - ReLU - fully connected layer - softmax

        The outputs of the second fully-connected layer are the scores for each class.
        """
    def __init__(self,input_size,hidden_layer,output):
        self.W1=0.0001*np.random.randn(input_size,hidden_layer)
        self.W2=0.0001*np.random.randn(hidden_layer,output)
        self.b1=np.zeros(hidden_layer)
        self.b2=np.zeros(output)

    def train(self,X_train,Y_train,train_num,batch_num,iter_nums,learning_rate ,learning_rate_decay):
        loss_history=[]
        x_train_data=X_train[:train_num,:]
        x_valid_data=X_train[train_num:-1,:]
        y_train_data=Y_train[:train_num]
        y_valid_data=Y_train[train_num:-1]
        for iter_num in range(iter_nums):
            sample_id=np.random.choice(x_train_data.shape[0],batch_num,replace=False)
            x_batch_data=x_train_data[sample_id,:]
            y_batch_data=y_train_data[sample_id]
            loss,dw1,dw2,db1,db2=self.loss(x_batch_data,y_batch_data)
            loss_history.append(loss)
            a = db2.reshape(-1)
            db2 = a
            a = db1.reshape(-1)
            db1 = a
            db1.reshape(-1)

            self.W1-=learning_rate*dw1
            self.W2-=learning_rate*dw2
            self.b1+=-learning_rate*db1
            self.b2+=-learning_rate*db2
            if iter_num%500==0:

                learning_rate *= learning_rate_decay
                pred=self.predict(x_valid_data)
                accuracy=np.sum(pred==y_valid_data)/x_valid_data.shape[0]
                print('iter_num:{} ,    loss:{:.4f},   accuracy of valid_data:{:.4f}\n'.format(iter_num, loss,accuracy))


    def loss(self,x_batch_data,y_batch_data):

        H1 = np.maximum(0, np.dot(x_batch_data,self.W1)+ self.b1)
        score=H1.dot(self.W2)+self.b2
        per_max=np.max(score,axis=1,keepdims=True)
        margins=score-per_max
        margins=np.exp(margins)/np.sum(np.exp(margins),axis=1,keepdims=True)
        correct_class=np.zeros_like(margins)
        correct_class[range(x_batch_data.shape[0]),y_batch_data]=1
        loss=-np.sum(correct_class*np.log(margins))/(x_batch_data.shape[0])+0.5*(1e-5)*np.sum(self.W1*self.W1)+0.5*(1e-5)*np.sum(self.W2*self.W2)
        margin_new=(margins-correct_class)/x_batch_data.shape[0]
        dw2=np.dot(H1.T,margin_new)+(1e-5)*self.W2
        db2=np.sum(margin_new,axis=0,keepdims=True)

        dh1=np.dot(margin_new,self.W2.T)
        dh1[H1 <= 0] = 0
        dw1=np.dot(x_batch_data.T,dh1)+(1e-5)*self.W1
        db1=np.sum(dh1,axis=0,keepdims=True)
        return loss,dw1,dw2,db1,db2


    def predict(self,X):
        y_pred = None
        h1 = np.maximum(0, (np.dot(X, self.W1) + self.b1))
        scores = np.dot(h1, self.W2) + self.b2
        y_pred = np.argmax(scores, axis=1)
        return y_pred

def test(X_train,Y_train,X_test,Y_test):
    test_TwoLayerNet = two_layers_network(X_train.shape[1], 100, 10)
    test_TwoLayerNet.train(x_train_data,y_train_data,48000,500,7000,1e-3,0.97)
    y_pred = test_TwoLayerNet.predict(X_test)
    accuracy = np.mean(y_pred == Y_test)
    print('-----------------test result------------------')
    print('Accuracy achieved during cross-validation: {}'.format(np.mean(accuracy)))


step1=Data()
x_train_data,y_train_data=step1.get_train_data('./cifar-10-batches-py')
step1.Visualdata(x_train_data,y_train_data)
x_test,y_test=step1.get_test_data('./cifar-10-batches-py')
'''
step2=two_layers_network(x_train_data.shape[1],100,10)
print('\n-----------------training process------------------')
step2.train(x_train_data,y_train_data,48000,500,7000,1e-3,0.97)
'''
test(x_train_data,y_train_data,x_test,y_test)

