# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:00:48 2018

@author: xfxf
"""
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD

model = Sequential() #模型初始化
model.add(Dense(64, activation='relu', input_dim=20)) #添加输入层20，第一隐藏层64 第一隐藏层用tanh作为激活函数
model.add(Dropout(0.5)) #使用Dropout防止过拟合
model.add(Dense(64, activation='relu')) #添加第二隐藏层64,第二隐藏层用tanh作为激活函数
model.add(Dropout(0.5)) #使用Dropout防止过拟合
model.add(Dense(10, activation='softmax')) #添加输出层1,出层用sigmoid作为激活函数

sgd = SGD(lr=1,decay=1e-6,momentum=0.9,nesterov=True) #定义求解算法
model.compile(loss='mean_squared_error',optimizer=sgd) #编译成模型，定义损失函数

model.fit(x_train,y_train,nb_epoch=20,batch_size=16) #训练模型
score = model.evaluate(x_test,y_test,batch_size=16) #测试模型


import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

#导入数据
from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder='D:\\work\\sky_drive\\mnist'
mnist = input_data.read_data_sets(MNIST_data_folder,one_hot=True)
print(mnist.train.next_batch(1))
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w)+b)

y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
    
correct_pridiction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pridiction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


#另一种导入数据的方法
'''
import numpy as np
class mnist_data(object):
    def load_npz(self,path):
        f = np.load(path)
        for i in f:
           print i        
        x_train = f['trainInps']
        y_train = f['trainTargs']
        x_test = f['testInps']
        y_test = f['testTargs']        
        f.close()
        return (x_train, y_train), (x_test, y_test)
a = mnist_data()
(x_train, y_train), (x_test, y_test) = a.load_npz('D:/AI/torch/data/mnist.npz')
print ("train rows:%d,test rows:%d"% (x_train.shape[0], x_test.shape[0]))
print("x_train shape",x_train.shape)
print("y_train shape",y_train.shape )'''





