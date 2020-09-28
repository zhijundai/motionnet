from pandas import read_csv
from os import listdir
from numpy import zeros,reshape,abs,array,log,e,linspace,set_printoptions,inf,sqrt,multiply,mean
from numpy.random import normal
from numpy.fft import fft,fftshift,fftfreq,ifft,ifftshift
from scipy import interpolate,fftpack
from matplotlib.pyplot import plot,show
import tensorflow as tf
import generation,processing

set_printoptions(threshold=inf)

freqs = [0.05,1/15,1/14,1/13,1/12,1/11,0.1,1/9.5,1/9,1/8.5,1/8,1/7.5,1/7,1/6.5,1/6,1/5.5,0.2,1/4.8,1/4.6,1/4.4,1/4.2,0.25,
        1/3.8,1/3.6,1/3.5,1/3.4,1/3.2,1/3,1/2.8,1/2.6,0.4,1/2.4,1/2.2,0.5,1/1.9,1/1.8,1/1.7,1/1.6,1/1.5,1/1.4,1/1.3,1/1.2,
        1/1.1,1,1/0.95,1/0.9,1/0.85,1/0.8,1/0.75,1/0.7,1/0.667,1/0.65,1/0.6,1/0.55,2,1/0.48,1/0.46,1/0.45,1/0.44,1/0.42,2.5,
        1/0.38,1/0.36,1/0.35,1/0.34,1/0.32,1/0.3,1/0.29,1/0.28,1/0.26,4,1/0.24,1/0.22,5,1/0.19,1/0.18,1/0.17,1/0.16,1/0.15,
        1/0.14,1/0.133,1/0.13,1/0.12,1/0.11,10,1/0.095,1/0.09,1/0.085,1/0.08,1/0.075,1/0.07,1/0.067,1/0.065,1/0.06,1/0.055,
        1/0.05,1/0.048,1/0.046,1/0.045,1/0.044,1/0.042,1/0.04,1/0.036,1/0.035,1/0.032,1/0.03,1/0.029,40,1/0.022]#50,100
alpha = [0.30302783,52.70260672,0.49589299,0.36646284,0.36389123,0.50829513,51.521895,0.83615846,0.71335165,0.80023908,
         1.19574326,61.49058258,1.85620949,2.35092295,82.13356623,5.1660684,111.71098358,14.75666013,16.57306215,31.04632018,
         69.89470053,56.2822909,166.1643488,81.13852407,89.25130365,121.30551298,250.19013983,192.54842257,174.8316367,205.05816776,
         198.95663296,193.51063079,200.11043911,205.53643052,212.81183979,204.02563972,206.86712805,203.74829902,192.78506884,
         179.90655765,179.7804624,176.12816949,164.18659842,148.88663958,139.33131795,127.43323131,120.96390544,111.10743645,
         101.19259613,91.3855393,85.53304783,81.95602996,71.7926808,61.76615664,52.37869427,49.17560825,47.47023887,45.63355318,
         43.06956582,39.25473124,36.45045296,34.47916435,31.32990128,29.70200838,27.74560804,24.21975572,22.1559851,21.3455658,
         19.66198894,17.1503231,15.93074974,14.64539116,12.56718707,10.44414263,9.2514546,8.61920217,7.68026383,7.09543965,
         6.38621759,5.61175625,4.98802346,4.86913053,4.28002772,3.54905966,3.0438945,2.78117069,2.58535722,2.32383356,2.15121587,
         1.96685555,1.75543746,1.63041599,1.54142527,1.37404959,1.19657335,1.08715459,1.0503461,1.00653963,0.98750986,0.97953111,
         0.92756549,0.90600971,0.85511704,0.84498538,0.80109885,0.77234792,0.76049867,0.74272218,0.70855741]

# print全部信息
set_printoptions(threshold=inf)

# 定义权值、偏置值
def weight_variable(shape):
    w = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(w)
def bias_variable(shape):
    b = tf.constant(0.1,shape=shape)
    return tf.Variable(b)

# 占位符
x1 = tf.placeholder(tf.float32,shape=(400),name='x1',)
x1_r = tf.reshape(x1,[-1,1,400,1])
x2 = tf.placeholder(tf.float32,shape=(400),name='x2')
x2_r = tf.reshape(x2,[-1,1,400,1])
x3 = tf.placeholder(tf.float32,shape=(400),name='x3')
x3_r = tf.reshape(x3,[-1,1,400,1])
x4 = tf.placeholder(tf.float32,shape=(400),name='x4')
x4_r = tf.reshape(x4,[-1,1,400,1])
x5 = tf.placeholder(tf.float32,shape=(400),name='x5')
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
    # avg1_conv1 = tf.nn.avg_pool(relu1_conv1,ksize=[1,1,5,1],strides=[1,1,5,1],padding='SAME')
    # 1-x2
    w2_conv1 = weight_variable([1,5,1,4])
    b2_conv1 = bias_variable([4])
    relu2_conv1 = tf.nn.relu(tf.nn.conv2d(x2_r,w2_conv1,strides=[1,1,5,1],padding='SAME')+b2_conv1)
    # avg2_conv1 = tf.nn.avg_pool(relu2_conv1,ksize=[1,1,5,1],strides=[1,1,5,1],padding='SAME')
    # 1-x3
    w3_conv1 = weight_variable([1,5,1,4])
    b3_conv1 = bias_variable([4])
    relu3_conv1 = tf.nn.relu(tf.nn.conv2d(x3_r,w3_conv1,strides=[1,1,5,1],padding='SAME')+b3_conv1)
    # avg3_conv1 = tf.nn.avg_pool(relu3_conv1,ksize=[1,1,5,1],strides=[1,1,5,1],padding='SAME')
    # 1-x4
    w4_conv1 = weight_variable([1,5,1,4])
    b4_conv1 = bias_variable([4])
    relu4_conv1 = tf.nn.relu(tf.nn.conv2d(x4_r,w4_conv1,strides=[1,1,5,1],padding='SAME')+b4_conv1)
    # avg4_conv1 = tf.nn.avg_pool(relu4_conv1,ksize=[1,1,5,1],strides=[1,1,5,1],padding='SAME')
    # 1-x5
    w5_conv1 = weight_variable([1,5,1,4])
    b5_conv1 = bias_variable([4])
    relu5_conv1 = tf.nn.relu(tf.nn.conv2d(x5_r,w5_conv1,strides=[1,1,5,1],padding='SAME')+b5_conv1)
    # avg5_conv1 = tf.nn.avg_pool(relu5_conv1,ksize=[1,1,5,1],strides=[1,1,5,1],padding='SAME')
    # 第二层卷积层
    # 2-x1
    w1_conv2 = weight_variable([1,4,4,8])
    b1_conv2 = bias_variable([8])
    relu1_conv2 = tf.nn.relu(tf.nn.conv2d(relu1_conv1,w1_conv2,strides=[1,1,4,1],padding='SAME')+b1_conv2)
    # avg1_conv2 = tf.nn.avg_pool(relu1_conv2,ksize=[1,1,4,1],strides=[1,1,4,1],padding='SAME')
    # 2-x2
    w2_conv2 = weight_variable([1,4,4,8])
    b2_conv2 = bias_variable([8])
    relu2_conv2 = tf.nn.relu(tf.nn.conv2d(relu2_conv1,w2_conv2,strides=[1,1,4,1],padding='SAME')+b2_conv2)
    # avg2_conv2 = tf.nn.avg_pool(relu2_conv2,ksize=[1,1,4,1],strides=[1,1,4,1],padding='SAME')
    # 2-x3
    w3_conv2 = weight_variable([1,4,4,8])
    b3_conv2 = bias_variable([8])
    relu3_conv2 = tf.nn.relu(tf.nn.conv2d(relu3_conv1,w3_conv2,strides=[1,1,4,1],padding='SAME')+b3_conv2)
    # avg3_conv2 = tf.nn.avg_pool(relu3_conv2,ksize=[1,1,4,1],strides=[1,1,4,1],padding='SAME')
    # 2-x4
    w4_conv2 = weight_variable([1,4,4,8])
    b4_conv2 = bias_variable([8])
    relu4_conv2 = tf.nn.relu(tf.nn.conv2d(relu4_conv1,w4_conv2,strides=[1,1,4,1],padding='SAME')+b4_conv2)
    # avg4_conv2 = tf.nn.avg_pool(relu4_conv2,ksize=[1,1,4,1],strides=[1,1,4,1],padding='SAME')
    # 2-x5
    w5_conv2 = weight_variable([1,4,4,8])
    b5_conv2 = bias_variable([8])
    relu5_conv2 = tf.nn.relu(tf.nn.conv2d(relu5_conv1,w5_conv2,strides=[1,1,4,1],padding='SAME')+b5_conv2)
    # avg5_conv2 = tf.nn.avg_pool(relu5_conv2,ksize=[1,1,4,1],strides=[1,1,4,1],padding='SAME')
    # 第三层卷积层
    # 3-x1
    w1_conv3 = weight_variable([1,2,8,16])
    b1_conv3 = bias_variable([16])
    relu1_conv3 = tf.nn.relu(tf.nn.conv2d(relu1_conv2,w1_conv3,strides=[1,1,2,1],padding='SAME')+b1_conv3)
    # avg1_conv3 = tf.nn.avg_pool(relu1_conv3,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')
    # 3-x2
    w2_conv3 = weight_variable([1,2,8,16])
    b2_conv3 = bias_variable([16])
    relu2_conv3 = tf.nn.relu(tf.nn.conv2d(relu2_conv2,w2_conv3,strides=[1,1,2,1],padding='SAME')+b2_conv3)
    # avg2_conv3 = tf.nn.avg_pool(relu2_conv3,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')
    # 3-x3
    w3_conv3 = weight_variable([1,2,8,16])
    b3_conv3 = bias_variable([16])
    relu3_conv3 = tf.nn.relu(tf.nn.conv2d(relu3_conv2,w3_conv3,strides=[1,1,2,1],padding='SAME')+b3_conv3)
    # avg3_conv3 = tf.nn.avg_pool(relu3_conv3,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')
    # 3-x4
    w4_conv3 = weight_variable([1,2,8,16])
    b4_conv3 = bias_variable([16])
    relu4_conv3 = tf.nn.relu(tf.nn.conv2d(relu4_conv2,w4_conv3,strides=[1,1,2,1],padding='SAME')+b4_conv3)
    # avg4_conv3 = tf.nn.avg_pool(relu4_conv3,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')
    # 3-x5
    w5_conv3 = weight_variable([1,2,8,16])
    b5_conv3 = bias_variable([16])
    relu5_conv3 = tf.nn.relu(tf.nn.conv2d(relu5_conv2,w5_conv3,strides=[1,1,2,1],padding='SAME')+b5_conv3)
    # avg5_conv3 = tf.nn.avg_pool(relu5_conv3,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')
    # 把输出reshape成一维
    cnn1 = tf.reshape(relu1_conv3,[-1,160])
    cnn2 = tf.reshape(relu2_conv3,[-1,160])
    cnn3 = tf.reshape(relu3_conv3,[-1,160])
    cnn4 = tf.reshape(relu4_conv3,[-1,160])
    cnn5 = tf.reshape(relu5_conv3,[-1,160])

# 全连接层
with tf.name_scope('fcnn'):
    fcnn1_1 = tf.contrib.layers.fully_connected(cnn1,40,scope='fcnn1_1')
    fcnn1_2 = tf.contrib.layers.fully_connected(cnn2,40,scope='fcnn1_2')
    fcnn1_3 = tf.contrib.layers.fully_connected(cnn3,40,scope='fcnn1_3')
    fcnn1_4 = tf.contrib.layers.fully_connected(cnn4,40,scope='fcnn1_4')
    fcnn1_5 = tf.contrib.layers.fully_connected(cnn5,40,scope='fcnn1_5')
    fcnn2_1 = tf.contrib.layers.fully_connected(fcnn1_1,20,scope='fcnn2_1')
    fcnn2_2 = tf.contrib.layers.fully_connected(fcnn1_2,20,scope='fcnn2_2')
    fcnn2_3 = tf.contrib.layers.fully_connected(fcnn1_3,20,scope='fcnn2_3')
    fcnn2_4 = tf.contrib.layers.fully_connected(fcnn1_4,20,scope='fcnn2_4')
    fcnn2_5 = tf.contrib.layers.fully_connected(fcnn1_5,20,scope='fcnn2_5')
    # 把五个输出合并
    s = tf.concat((fcnn2_1,fcnn2_2,fcnn2_3,fcnn2_4,fcnn2_5),axis=1)
    fcnn3 = tf.contrib.layers.fully_connected(s,40,scope='fcnn3')
    fcnn4 = tf.contrib.layers.fully_connected(fcnn3,10,scope='fcnn4')
    fcnn5 = tf.contrib.layers.fully_connected(fcnn4,2,activation_fn=tf.nn.softmax,scope='fcnn5')

# 存取模型
saver = tf.train.Saver()

# 分类正确率
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(fcnn5,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))



path = './datasets/small/'
results = []
n = 0
d = 0
x = 0
records_list = listdir(path)

# 调整频域后输入模型
for record in records_list:
    files_list = listdir(path+record)
    if record+'.EW' in files_list:
        for w in ['.EW','.NS','.UD']:
            print(n)
            paths = path+record+'/'+record+w
            file_m = read_csv(paths, header=None, skiprows=4, nrows=1, sep='\s+')
            magnitude = float(file_m.loc[0, 1])
            if magnitude >= 5.5:
                d += 1
            else:
                x += 1
            g,frequence = generation.get_generation(paths)
            data1,data2,data3,data4,data5 = generation.split_generation(g,frequence)
            with tf.Session() as sess:
                saver.restore(sess, './model/acc_0.91836447_model.ckpt')
                result = sess.run(fcnn5,feed_dict={x1:data1,x2:data2,x3:data3,x4:data4,x5:data5})
                results.append(list(result[0]))
            n += 1
    else:
        for w in ['.EW1','.EW2','.NS1','.NS2','.UD1','.UD2']:
            print(n)
            paths = path+record+'/'+record+w
            file_m = read_csv(paths, header=None, skiprows=4, nrows=1, sep='\s+')
            magnitude = float(file_m.loc[0, 1])
            if magnitude >= 5.5:
                d += 1
            else:
                x += 1
            g,frequence = generation.get_generation(paths)
            data1,data2,data3,data4,data5 = generation.split_generation(g,frequence)
            with tf.Session() as sess:
                saver.restore(sess, './model/acc_0.91836447_model.ckpt')
                result = sess.run(fcnn5,feed_dict={x1:data1,x2:data2,x3:data3,x4:data4,x5:data5})
                results.append(list(result[0]))
            n += 1
results = list(results)
t = [0,0]
for i in results:
    i = list(i)
    if i.index(max(i)) == 0:
        t[0] += 1
    else:
        t[1] += 1
print(t)
print(x,d)

# 直接输入模型
# for record in records_list:
#     files_list = listdir(path+record)
#     if record+'.EW' in files_list:
#         for w in ['.EW','.NS','.UD']:
#             print(n)
#             paths = path+record+'/'+record+w
#             file_m = read_csv(paths, header=None, skiprows=4, nrows=1, sep='\s+')
#             magnitude = float(file_m.loc[0, 1])
#             if magnitude >= 5.5:
#                 d += 1
#             else:
#                 x += 1
#             frequence = processing.get_frequence(paths)
#             duration = processing.get_duration(paths)
#             number = duration * frequence
#             f1, f2 = processing.get_scale_factor(paths)
#             if number % 8 != 0:
#                 file = read_csv(paths, skiprows=17, header=None, nrows=number // 8, sep='\s+')
#                 file = reshape(array(file), number // 8 * 8)
#                 four = read_csv(paths, skiprows=17 + number // 8, header=None, nrows=1, sep='\s+')
#                 file = list(file)
#                 for i in range(4):
#                     file.extend([four.loc[0, i]])
#                 data = array(file) * f1 / f2
#             else:
#                 file = read_csv(paths, sep='\s+', skiprows=17, header=None, keep_default_na=False)
#                 data = reshape(array(file * f1 / f2), number)
#             data = data-mean(data)
#             data = processing.nomalize(data)
#             data1,data2,data3,data4,data5 = generation.split_generation(data,frequence)
#             with tf.Session() as sess:
#                 saver.restore(sess, './model/acc_0.91836447_model.ckpt')
#                 result = sess.run(fcnn5,feed_dict={x1:data1,x2:data2,x3:data3,x4:data4,x5:data5})
#                 results.append(list(result[0]))
#             n += 1
#     else:
#         for w in ['.EW1','.EW2','.NS1','.NS2','.UD1','.UD2']:
#             print(n)
#             paths = path + record + '/' + record + w
#             file_m = read_csv(paths, header=None, skiprows=4, nrows=1, sep='\s+')
#             magnitude = float(file_m.loc[0, 1])
#             if magnitude >= 5.5:
#                 d += 1
#             else:
#                 x += 1
#             frequence = processing.get_frequence(paths)
#             duration = processing.get_duration(paths)
#             number = duration * frequence
#             f1, f2 = processing.get_scale_factor(paths)
#             if number % 8 != 0:
#                 file = read_csv(paths, skiprows=17, header=None, nrows=number // 8, sep='\s+')
#                 file = reshape(array(file), number // 8 * 8)
#                 four = read_csv(paths, skiprows=17 + number // 8, header=None, nrows=1, sep='\s+')
#                 file = list(file)
#                 for i in range(4):
#                     file.extend([four.loc[0, i]])
#                 data = array(file) * f1 / f2
#             else:
#                 file = read_csv(paths, sep='\s+', skiprows=17, header=None)
#                 data = reshape(array(file * f1 / f2), number)
#             data = data - mean(data)
#             data = processing.nomalize(data)
#             data1, data2, data3, data4, data5 = generation.split_generation(data, frequence)
#             with tf.Session() as sess:
#                 saver.restore(sess, './model/acc_0.91836447_model.ckpt')
#                 result = sess.run(fcnn5, feed_dict={x1: data1, x2: data2, x3: data3, x4: data4, x5: data5})
#                 results.append(list(result[0]))
#             n += 1
# results = list(results)
# t = [0,0]
# for i in results:
#     i = list(i)
#     if i.index(max(i)) == 0:
#         t[0] += 1
#     else:
#         t[1] += 1
# print(t)
# print(x,d)