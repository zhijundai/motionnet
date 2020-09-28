import preprocessing
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
import time,os

# print全部信息
np.set_printoptions(threshold=np.inf)

# 读取数据
x_train1,x_train2,x_train3,x_train4,x_train5,x_test1,x_test2,x_test3,x_test4,x_test5,y_train,y_test = preprocessing.read_magnitude()

# 定义权值、偏置值
def weight_variable(shape):
    w = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(w)
def bias_variable(shape):
    b = tf.constant(0.1,shape=shape)
    return tf.Variable(b)

# 占位符
x1 = tf.placeholder(tf.float32,shape=(None,400),name='x1')
x1_r = tf.reshape(x1,[-1,1,400,1])
x2 = tf.placeholder(tf.float32,shape=(None,400),name='x2')
x2_r = tf.reshape(x2,[-1,1,400,1])
x3 = tf.placeholder(tf.float32,shape=(None,400),name='x3')
x3_r = tf.reshape(x3,[-1,1,400,1])
x4 = tf.placeholder(tf.float32,shape=(None,400),name='x4')
x4_r = tf.reshape(x4,[-1,1,400,1])
x5 = tf.placeholder(tf.float32,shape=(None,400),name='x5')
x5_r = tf.reshape(x5,[-1,1,400,1])
y = tf.placeholder(tf.int32,shape=(None),name='y')
learning_rate = tf.placeholder(tf.float32,shape=(None),name='learning_rate')

# 构建神经网络
with tf.name_scope('cnn'):
    # 第一层卷积层
    # 1-x1
    w1_conv1 = weight_variable([1,5,1,4])
    b1_conv1 = bias_variable([4])
    relu1_conv1 = tf.nn.relu(tf.nn.conv2d(x1_r,w1_conv1,strides=[1,1,5,1],padding='SAME')+b1_conv1)
    # 1-x2
    w2_conv1 = weight_variable([1,5,1,4])
    b2_conv1 = bias_variable([4])
    relu2_conv1 = tf.nn.relu(tf.nn.conv2d(x2_r,w2_conv1,strides=[1,1,5,1],padding='SAME')+b2_conv1)
    # 1-x3
    w3_conv1 = weight_variable([1,5,1,4])
    b3_conv1 = bias_variable([4])
    relu3_conv1 = tf.nn.relu(tf.nn.conv2d(x3_r,w3_conv1,strides=[1,1,5,1],padding='SAME')+b3_conv1)
    # 1-x4
    w4_conv1 = weight_variable([1,5,1,4])
    b4_conv1 = bias_variable([4])
    relu4_conv1 = tf.nn.relu(tf.nn.conv2d(x4_r,w4_conv1,strides=[1,1,5,1],padding='SAME')+b4_conv1)
    # 1-x5
    w5_conv1 = weight_variable([1,5,1,4])
    b5_conv1 = bias_variable([4])
    relu5_conv1 = tf.nn.relu(tf.nn.conv2d(x5_r,w5_conv1,strides=[1,1,5,1],padding='SAME')+b5_conv1)
    # 第二层卷积层
    # 2-x1
    w1_conv2 = weight_variable([1,4,4,8])
    b1_conv2 = bias_variable([8])
    relu1_conv2 = tf.nn.relu(tf.nn.conv2d(relu1_conv1,w1_conv2,strides=[1,1,4,1],padding='SAME')+b1_conv2)
    # 2-x2
    w2_conv2 = weight_variable([1,4,4,8])
    b2_conv2 = bias_variable([8])
    relu2_conv2 = tf.nn.relu(tf.nn.conv2d(relu2_conv1,w2_conv2,strides=[1,1,4,1],padding='SAME')+b2_conv2)
    # 2-x3
    w3_conv2 = weight_variable([1,4,4,8])
    b3_conv2 = bias_variable([8])
    relu3_conv2 = tf.nn.relu(tf.nn.conv2d(relu3_conv1,w3_conv2,strides=[1,1,4,1],padding='SAME')+b3_conv2)
    # 2-x4
    w4_conv2 = weight_variable([1,4,4,8])
    b4_conv2 = bias_variable([8])
    relu4_conv2 = tf.nn.relu(tf.nn.conv2d(relu4_conv1,w4_conv2,strides=[1,1,4,1],padding='SAME')+b4_conv2)
    # 2-x5
    w5_conv2 = weight_variable([1,4,4,8])
    b5_conv2 = bias_variable([8])
    relu5_conv2 = tf.nn.relu(tf.nn.conv2d(relu5_conv1,w5_conv2,strides=[1,1,4,1],padding='SAME')+b5_conv2)
    # 第三层卷积层
    # 3-x1
    w1_conv3 = weight_variable([1,2,8,16])
    b1_conv3 = bias_variable([16])
    relu1_conv3 = tf.nn.relu(tf.nn.conv2d(relu1_conv2,w1_conv3,strides=[1,1,2,1],padding='SAME')+b1_conv3)
    # 3-x2
    w2_conv3 = weight_variable([1,2,8,16])
    b2_conv3 = bias_variable([16])
    relu2_conv3 = tf.nn.relu(tf.nn.conv2d(relu2_conv2,w2_conv3,strides=[1,1,2,1],padding='SAME')+b2_conv3)
    # 3-x3
    w3_conv3 = weight_variable([1,2,8,16])
    b3_conv3 = bias_variable([16])
    relu3_conv3 = tf.nn.relu(tf.nn.conv2d(relu3_conv2,w3_conv3,strides=[1,1,2,1],padding='SAME')+b3_conv3)
    # 3-x4
    w4_conv3 = weight_variable([1,2,8,16])
    b4_conv3 = bias_variable([16])
    relu4_conv3 = tf.nn.relu(tf.nn.conv2d(relu4_conv2,w4_conv3,strides=[1,1,2,1],padding='SAME')+b4_conv3)
    # 3-x5
    w5_conv3 = weight_variable([1,2,8,16])
    b5_conv3 = bias_variable([16])
    relu5_conv3 = tf.nn.relu(tf.nn.conv2d(relu5_conv2,w5_conv3,strides=[1,1,2,1],padding='SAME')+b5_conv3)
    # 把输出reshape成一维
    cnn1 = tf.reshape(relu1_conv3,[-1,160])
    cnn2 = tf.reshape(relu2_conv3,[-1,160])
    cnn3 = tf.reshape(relu3_conv3,[-1,160])
    cnn4 = tf.reshape(relu4_conv3,[-1,160])
    cnn5 = tf.reshape(relu5_conv3,[-1,160])

# 全连接层
with tf.name_scope('fcnn'):
    fcnn1_1 = fully_connected(cnn1,40,scope='fcnn1_1')
    fcnn1_2 = fully_connected(cnn2,40,scope='fcnn1_2')
    fcnn1_3 = fully_connected(cnn3,40,scope='fcnn1_3')
    fcnn1_4 = fully_connected(cnn4,40,scope='fcnn1_4')
    fcnn1_5 = fully_connected(cnn5,40,scope='fcnn1_5')
    fcnn2_1 = fully_connected(fcnn1_1,20,scope='fcnn2_1')
    fcnn2_2 = fully_connected(fcnn1_2,20,scope='fcnn2_2')
    fcnn2_3 = fully_connected(fcnn1_3,20,scope='fcnn2_3')
    fcnn2_4 = fully_connected(fcnn1_4,20,scope='fcnn2_4')
    fcnn2_5 = fully_connected(fcnn1_5,20,scope='fcnn2_5')
    # 把五个输出合并
    s = tf.concat((fcnn2_1,fcnn2_2,fcnn2_3,fcnn2_4,fcnn2_5),axis=1)
    fcnn3 = fully_connected(s,40,scope='fcnn3')
    fcnn4 = fully_connected(fcnn3,10,scope='fcnn4')
    fcnn5 = fully_connected(fcnn4,2,activation_fn=None,scope='fcnn5')

# 存取模型
saver = tf.train.Saver()

# 损失函数
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=fcnn5)
loss = tf.reduce_mean(entropy,name='loss')

# Adam优化器
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

# 分类正确率
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(fcnn5,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

# 初始化函数
init = tf.global_variables_initializer()

# 训练模型
n_epochs = 2000
max_acc = 0
batch_size = 100
data_size = len(x_train1)
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        shuffle_index = np.random.permutation(data_size)
        x_train1_rd = np.array(x_train1)[shuffle_index]
        x_train2_rd = np.array(x_train2)[shuffle_index]
        x_train3_rd = np.array(x_train3)[shuffle_index]
        x_train4_rd = np.array(x_train4)[shuffle_index]
        x_train5_rd = np.array(x_train5)[shuffle_index]
        y_train_rd = np.array(y_train)[shuffle_index]
        for batch in range(915):
            start = batch*batch_size
            end = min(start+batch_size,data_size)
            sess.run(train_op,feed_dict={x1:x_train1_rd[start:end],x2:x_train2_rd[start:end],x3:x_train3_rd[start:end],
                                         x4:x_train4_rd[start:end],x5:x_train5_rd[start:end],y:y_train_rd[start:end],
                                         learning_rate:0.001*0.99**(epoch/100)})
        acc_train = sess.run(accuracy,feed_dict={x1:x_train1,x2:x_train2,x3:x_train3,x4:x_train4,x5:x_train5,y:y_train})
        acc_test = sess.run(accuracy,feed_dict={x1:x_test1,x2:x_test2,x3:x_test3,x4:x_test4,x5:x_test5,y:y_test})
        print(epoch,'Train sort accuracy:',acc_train,'Test sort accuracy:',acc_test,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if acc_train+acc_test > 1.7 and acc_test > max_acc:
            max_acc = acc_test
            print(max_acc)
            saver.save(sess, './model/2/acc_'+str(max_acc)+'_model.ckpt')
        saver.save(sess, './model/2/epochs/epoch_'+str(epoch)+'_model.ckpt')



# with tf.Session() as sess:
#     saver.restore(sess, './model/acc_0.91836447_model.ckpt')
#     acc_train = sess.run(accuracy,feed_dict={x1:x_train1,x2:x_train2,x3:x_train3,x4:x_train4,x5:x_train5,y:y_train})
#     print('Accuracy:',acc_train)

