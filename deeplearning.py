# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 08:32:07 2019

@author: xfxf
"""

# In[0]:
import os
os.system('cls')
import tensorflow as tf
import numpy as np

#导入数据
from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder='D:\\work\\sky_drive\\deepleaning\\minist_testdemo\\mnist\\MNIST_DATA'
MNIST_data_folder='D:\\work\\0_lea\\28.燎原计划\\期末考试\\入学摸底测试——深度学习讲评\\二、深度学习实战数据集MNIST'
mnist = input_data.read_data_sets(MNIST_data_folder,one_hot=True)
#print(mnist.train.next_batch(1))
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

#定义参数
numClasses = 10 
inputSize = 784 
numHiddenUnits = 250 
numHiddenUnitsLayer2 = 150
trainingIterations = 1000 
batchSize = 100 
learning_rate = 0.1

#设置网络
X = tf.placeholder(tf.float32, shape = [None, inputSize])
y = tf.placeholder(tf.float32, shape = [None, numClasses])
W1 = tf.Variable(tf.random_normal([inputSize, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])
W2 = tf.Variable(tf.random_normal([numHiddenUnits, numHiddenUnitsLayer2], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1), [numHiddenUnitsLayer2])
W3 = tf.Variable(tf.random_normal([numHiddenUnitsLayer2, numClasses], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1), [numClasses])

hiddenLayerOutput = tf.matmul(X, W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
hiddenLayer2Output = tf.matmul(hiddenLayerOutput, W2) + B2
hiddenLayer2Output = tf.nn.relu(hiddenLayer2Output)
finalOutput = tf.matmul(hiddenLayer2Output, W3) + B3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = finalOutput))
opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(finalOutput,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#运行训练
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    _, trainingLoss = sess.run([opt, loss], feed_dict={X: batchInput, y: batchLabels})
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
        print ("step %d, training accuracy %g"%(i, train_accuracy))

#预测结果，评估模型
testInputs = mnist.test.images
testLabels = mnist.test.labels
acc = accuracy.eval(session=sess, feed_dict = {X: testInputs, y: testLabels})
print("testing accuracy: {}".format(acc))

#输出结果


# In[1]:
import os
os.system('cls')
reset
clear

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #导入读取数据的input_data
#在同一个文件夹下解压数据集到MNIST_data文件夹，然后用input_data函数读取，标签采用one_hot编码
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 

batch_size = 100 #每批数据大小
learning_rate = 0.01 #学习率
num_batchs = 10001 #训练轮数

#定义模型
#默认读取的数据是每一张图片由长度为784的向量组成，而非28×28的单通道数据。

X_holder = tf.placeholder(tf.float32)#定义输入占位符，输入数据是[batch_size,784]
y_holder = tf.placeholder(tf.float32)#定义输出占位符, 输出是[batch_size,10]

Weights1 = tf.Variable(tf.truncated_normal([784, 250],stddev=0.01))#权重通过tf.truncated_normal进行正态初始化，标准差设置为0.01 
biases1 = tf.Variable(tf.zeros([1,250]))#偏置初始化，0初始化
W_plus_b1 = tf.nn.relu(tf.matmul(X_holder, Weights1) + biases1)#计算第一层全连接，激活函数采用relu,计算后数据大小变为 [batch_size,250]

Weights2 = tf.Variable(tf.truncated_normal([250, 150],stddev=0.01))#权重通过tf.truncated_normal进行正态初始化，标准差设置为0.01
biases2 = tf.Variable(tf.zeros([1,150]))#偏置初始化，0初始化
W_plus_b2 = tf.nn.relu(tf.matmul(W_plus_b1, Weights2) + biases2)#计算第二层全连接，激活函数采用relu,计算后数据大小变为[batch_size,150]

Weights3 = tf.Variable(tf.truncated_normal([150, 10], stddev=0.01)) #权重通过tf.truncated_normal进行正态初始化，标准差设置为0.01
biases3 = tf.Variable(tf.zeros([1,10]))#偏置初始化，0初始化
predict_y = tf.nn.softmax(tf.matmul(W_plus_b2, Weights3) + biases3)#通过softmax激活函数计算输出层,输出的数据大小为[batch_size,10]

loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))#定义损失函数为交叉熵
optimizer = tf.train.GradientDescentOptimizer(learning_rate)#优化器为梯度下降，使用之前定义的学习率
train = optimizer.minimize(loss)#定义优化过程为极小化损失函数

session = tf.Session() #创建会话
init = tf.global_variables_initializer()#初始化变量
session.run(init)

#训练过程
for i in range(num_batchs):
    train_images, train_labels = mnist.train.next_batch(batch_size)#通过next_batch方法导入批次数据
    session.run(train, feed_dict={X_holder:train_images, y_holder:train_labels}) #给占位符输入数据
    #每隔一定批次计算准确率    
    if i % 1000 == 0:
        #定义预测准确的规则，因为输出层经过softmax激活函数输出一个1×10向量，依次表示是数字0-9的概率，所以需要argmax返回概率最大标签
        correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))#定义预测一致的规则
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#定义计算准确率规则
        train_accuracy = session.run(accuracy, feed_dict={X_holder:train_images, y_holder:train_labels})#计算训练过程中一个batch的准确率
        print('step:%d train accuracy:%.4f ' %(i, train_accuracy))#打印准确率
#计算测试集的预测标签并输出
test_images, test_labels = mnist.test.next_batch(10000)#读取测试集，测试集共10000个数据。
y_pred = tf.argmax(predict_y, 1) #返回最大概率的数字
test_pred = session.run(y_pred, feed_dict={X_holder:test_images, y_holder:test_labels})#返回预测标签

#将预测标签写入文件
with open('肖锋_dl_result1.csv', 'w') as f:
    f.write('predict labels'+'\n')#写入表头
    #写入数据
    for labels in test_pred:
        f.write(str(labels) + '\n')


# In[2]:
import os
os.system('cls')
reset
clear

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #导入读取数据的input_data
#在同一个文件夹下解压数据集到MNIST_data文件夹，然后用input_data函数读取，标签采用one_hot编码
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 

batch_size = 200
learning_rate = 0.001
num_batchs = 1001

#定义模型
#默认读取的数据是每一张图片由长度为784的向量组成，而非28×28的单通道数据。

X_holder = tf.placeholder(tf.float32)#定义输入占位符
y_holder = tf.placeholder(tf.float32)#定义输出占位符
X_images = tf.reshape(X_holder, [-1, 28, 28, 1])#数据变成28×28单通道，数据大小此时为[batch_size,28,28,1]

#第一个卷积层
conv1_Weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))#定义卷积核，四个参数分别表示卷积核长，卷积核宽，步长，数量
conv1_biases = tf.Variable(tf.constant(0.1, shape=[32]))#定义偏置
#计算卷积，padding策略为'SAME',此时数据大小为[batch_size,28,28,32]
conv1_conv2d = tf.nn.conv2d(X_images, conv1_Weights, strides=[1, 1, 1, 1], padding='SAME') + conv1_biases 
conv1_activated = tf.nn.relu(conv1_conv2d)#定义激活函数并计算
#定义池化层并计算，此时数据大小为[batch_size,14,14,32]
conv1_pooled = tf.nn.max_pool(conv1_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#第二个卷积层
conv2_Weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))#定义卷积核，四个参数分别表示卷积核长，卷积核宽，步长，数量
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))#定义偏置
#计算卷积，padding策略为'SAME',此时数据大小为[batch_size,14,14,64]
conv2_conv2d = tf.nn.conv2d(conv1_pooled, conv2_Weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_biases
conv2_activated = tf.nn.relu(conv2_conv2d)#定义激活函数并计算
#定义池化层并计算，此时数据大小为[batch_size,7,7,64]
conv2_pooled = tf.nn.max_pool(conv2_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#第一个全连接层
connect1_flat = tf.reshape(conv2_pooled, [-1, 7 * 7 * 64])#定义flat层，卷积层和全连接层之间需要一个flat层压平数据
#权重通过tf.truncated_normal进行正态初始化，标准差设置为0.01
connect1_Weights = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1)) 
connect1_biases = tf.Variable(tf.constant(0.1, shape=[1024]))#偏置初始化，常量初始化
connect1_Wx_plus_b = tf.add(tf.matmul(connect1_flat, connect1_Weights), connect1_biases)
connect1_activated = tf.nn.relu(connect1_Wx_plus_b)#计算第一层全连接，激活函数采用relu,计算后数据大小变为 [batch_size,1024]
#输出层
connect2_Weights = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))#权重通过tf.truncated_normal进行正态初始化，标准差设置为0.01
connect2_biases = tf.Variable(tf.constant(0.1, shape=[10]))#偏置初始化，常量初始化
connect2_Wx_plus_b = tf.add(tf.matmul(connect1_activated, connect2_Weights), connect2_biases)
predict_y = tf.nn.softmax(connect2_Wx_plus_b)#通过softmax激活函数计算输出层,输出的数据大小为[batch_size,10]

loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))#损失函数
optimizer = tf.train.AdamOptimizer(learning_rate)#优化器采用Adam，学习率采用之前定义的
train = optimizer.minimize(loss)#训练目标为极小化损失函数

init = tf.global_variables_initializer()#初始化变量
session = tf.Session()#创建会话
session.run(init)

#训练过程
for i in range(num_batchs):
    train_images, train_labels = mnist.train.next_batch(batch_size)#通过next_batch方法导入批次数据
    session.run(train, feed_dict={X_holder:train_images, y_holder:train_labels})#给占位符输入数据    
    #每隔一定批次计算准确率
    if i % 100 == 0:
         #定义预测准确的规则，因为输出层经过softmax激活函数输出一个1×10向量，依次表示是数字0-9的概率，所以需要argmax返回概率最大的标签
        correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))#定义预测一致的规则
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#定义准确率计算规则
        train_accuracy = session.run(accuracy, feed_dict={X_holder:train_images, y_holder:train_labels})#计算batch准确率
        print('step:%d train accuracy:%.4f ' %(i, train_accuracy))#打印准确率

#计算测试集的预测标签并输出
#由于卷积更复杂，一次性计算10000个测试集的预测值并写入会out of memeory，因此分批次计算并写入。如果GPU内存够大可以一次性计算并写入。
#将预测标签写入文件
with open('肖锋_dl_result2.csv', 'w') as f:
    f.write('predict labels'+'\n')#写入表头
    #写入数据
    for i in range(10):
        test_images, test_labels = mnist.test.next_batch(1000)#读取测试集，测试集共10000个数据。
        y_pred = tf.argmax(predict_y, 1) #返回最大概率的数字
        test_pred = session.run(y_pred, feed_dict={X_holder:test_images, y_holder:test_labels})#返回预测标签
        for labels in test_pred:
            f.write(str(labels) + '\n')


# In[3]:
import os
os.system('cls')
reset
clear

import tensorflow as tf
from tensorflow import keras

# # 读取数据
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('训练集图片数量及维度：', train_images.shape, 
      '测试集图片数量及维度：', test_images.shape)

# # 数据预处理

# 改变输入数据的形状，CNN的输入是4维的张量，第一维是样本大小，第二维和第三维是长度和宽度，第四维是像素通道
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')
# 像素的灰度值在0-255之间，为了使模型的训练效果更好，将数值归一化映射到0-1之间
train_images = train_images / 255
test_images = test_images / 255


# # 构建LeNet-5网络
model = keras.Sequential([
    # 卷积层1:32个5*5的卷积核，步长为1，输入大小为28*28*1，全0填充，激活函数为tanh
    keras.layers.Conv2D(filters=32, kernel_size=(5,5),
                        strides=1, padding="same", input_shape=(28, 28, 1), 
                        activation='tanh', name='conv2d_1'),
    # 池化层1：采样区域为2*2，步长为2
    keras.layers.MaxPool2D(pool_size=(2,2), strides=2, name="maxpool_1"),
    # 卷积层2:64个5*5的卷积核，步长为1，全0填充，激活函数为tanh
    keras.layers.Conv2D(filters=64, kernel_size=(5,5),
                        strides=1, padding="same", activation='tanh', name='conv2d_2'),
    # 池化层2：采样区域为2*2，步长为2
    keras.layers.MaxPool2D(pool_size=(2,2), strides=2, name="maxpool_2"),
    # 扁平化
    keras.layers.Flatten(),
    # 全连接层1：120个神经元，激活函数为tanh
    keras.layers.Dense(120, activation='tanh'),
    # 全连接层2：84个神经元，激活函数为tanh
    keras.layers.Dense(84, activation='tanh'),   
    # Softmax层：10个神经元
    keras.layers.Dense(10, activation='softmax')
])

# # 训练模型

# 定义损失函数为稀疏交叉熵，优化方法为Adam，评价准则为正确率
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 指定GPU进行训练
#import os
## 选择GPU 0还是1
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True     
#sess = tf.Session()
#keras.backend.set_session(sess)
# 训练模型
model.fit(train_images, train_labels, # 指定训练集和训练集标签
          validation_data=(test_images, test_labels), # 指定验证集和验证集标签,便于在训练中随时查看模型效果 
          epochs=5, # 迭代轮数为5轮 
          batch_size=200 # batch的大小为200
         ) 


# In[4]:算不出来

import os
os.system('cls')
reset
clear
import tensorflow as tf

# # 读取数据

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# # 创建LeNet-5网络

# 初始化所有的权值 W，给权重添加截断的正态分布噪声，标准差为0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 初始化所有的偏置项 b
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

# 构建卷积层，步长都为1
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

# 构建池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 定义占位符
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
# 将数据转为合适的维度来进行后续的计算
x_image = tf.reshape(x,[-1,28,28,1])                               
# 构建第一个卷积层
W_conv1 = weight_variable([5,5,1,32])                                
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.tanh(conv2d(x_image,W_conv1) + b_conv1)                
## 采用最大池化 
h_pool1 = max_pool_2x2(h_conv1)                                        
# 构建第二个卷积层
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.tanh(conv2d(h_pool1,W_conv2) + b_conv2)
## 采用最大池化 
h_pool2 = max_pool_2x2(h_conv2)
# 扁平化
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
# 构建第一个全连接层
W_fc1 = weight_variable([7*7*64,120])
b_fc1 = bias_variable([120])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
# 构建第二个全连接层
W_fc2 = weight_variable([120, 84])
b_fc2 = bias_variable([84])
h_fc2 = tf.nn.tanh(tf.matmul(h_fc1,W_fc2) + b_fc2)
# 构建Softmax层
W_fc3 = weight_variable([84,10])
b_fc3 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

# 采用交叉熵作为损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
# 采用Adam作为优化方法
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)                                
correct_predition = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
# 以分类正确率作为评价标准
accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))                                

# # 训练模型

#import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True 
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)  
 # 设定验证集
    validate_feed = {x: mnist.test.images, y_: mnist.test.labels}
    # 训练3000轮
    for i in range(3000):
        if i%200 == 0:
            # 使用测试集作为训练中的验证集，每200轮查看一下网络在验证集上的分类正确率
            validation_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("step %d,validation accuracy %g"%(i,validation_accuracy))
        # 设定batch大小为128
        batch_x, batch_y = mnist.train.next_batch(128)
        # 将数据传入进行训练
        sess.run(train_step, feed_dict={x:batch_x,y_:batch_y}) 


# In[5]:

##############################
import os
os.system('cls')

import tensorflow as tf
import numpy as np

#导入数据
from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder='D:\\work\\sky_drive\\deepleaning\\minist_testdemo\\mnist\\MNIST_DATA'
MNIST_data_folder='D:\\work\\0_lea\\28.燎原计划\\期末考试\\入学摸底测试——深度学习讲评\\二、深度学习实战数据集MNIST'
mnist = input_data.read_data_sets(MNIST_data_folder)

train_lbs = mnist.train.labels
train_imgs = mnist.train.images
test_lbs = mnist.test.labels
test_imgs = mnist.test.images

train_x = train_imgs.astype(np.float32)
test_x = test_imgs.astype(np.float32)
train_y = train_lbs.astype(np.int32)
test_y = test_lbs.astype(np.int32)

# 超参数
n_class = 10
n_epochs = 5
batch_size = 100
n_batchs = len(train_x) // batch_size
hidden_size1 = 250
hidden_size2 = 150
learning_rate = 0.01

# 数据集
data_x = tf.placeholder(tf.float32, shape=train_x.shape)
data_y = tf.placeholder(tf.int32, shape=train_y.shape)
dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
input_x, input_y = iterator.get_next()

# 静态计算图
h1 = tf.layers.flatten(input_x)
h2 = tf.layers.dense(h1, units=hidden_size1, activation='relu')
h3 = tf.layers.dense(h2, units=hidden_size2, activation='relu')
logits = tf.layers.dense(h3, units=n_class)

loss = tf.losses.sparse_softmax_cross_entropy(labels=input_y, logits=logits)
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

acc, acc_op = tf.metrics.accuracy(labels=input_y, predictions=tf.math.argmax(logits, axis=1))

# 运行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    epoch_loss_list = []
    for epoch in range(1, n_epochs + 1):
        sess.run(iterator.initializer, feed_dict={data_x: train_x, data_y: train_y})
        batch_loss_list = []
        for i in range(n_batchs):
            _, loss_value, acc_op_value = sess.run([optim, loss, acc_op])
            batch_loss_list.append(loss_value)
        epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        epoch_loss_list.append(epoch_loss)
        print('[Epoch %d] loss: %.3f | acc: %.3f' % (epoch, epoch_loss, acc_op_value))


#3. Keras
import os
os.system('cls')

import tensorflow as tf
import numpy as np

#导入数据
from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder='D:\\work\\sky_drive\\deepleaning\\minist_testdemo\\mnist\\MNIST_DATA'
MNIST_data_folder='D:\\work\\0_lea\\28.燎原计划\\期末考试\\入学摸底测试——深度学习讲评\\二、深度学习实战数据集MNIST'
mnist = input_data.read_data_sets(MNIST_data_folder)

train_lbs = mnist.train.labels
train_imgs = mnist.train.images
test_lbs = mnist.test.labels
test_imgs = mnist.test.images

train_x = train_imgs.astype(np.float32)
test_x = test_imgs.astype(np.float32)
train_y = train_lbs.astype(np.int32)
test_y = test_lbs.astype(np.int32)
        
#模型建立
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation=tf.nn.relu),
    tf.keras.layers.Dense(150, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=5, batch_size=100)

'''


















