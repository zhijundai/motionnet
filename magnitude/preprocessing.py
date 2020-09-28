import pandas as pd
import os
import numpy as np
import pickle

# 获取记录时间
def get_datetime(path):
    file_datetime = pd.read_csv(path,nrows=1,header=None,sep='\s+')
    date = file_datetime.loc[0,2]
    time = file_datetime.loc[0,3]
    while '/' in date:
        date = date.replace('/','')
    while ':' in time:
        time = time.replace(':','')
    return int(date+time)

# 获取采样频率
def get_frequence(path):
    file_f = pd.read_csv(path,skiprows=10,nrows=1,header=None,sep='\s+')
    frequence = file_f.loc[0,2]
    for w in 'Hz':
        frequence = frequence.replace(w,'')
    return int(frequence)

# 获取持续时间
def get_duration(path):
    file_d = pd.read_csv(path,skiprows=11,nrows=1,header=None,sep='\s+')
    d = file_d.loc[0,2]
    return d

# 缩放系数
def get_scale_factor(paths):
    scale_factor = pd.read_csv(paths, skiprows=13, nrows=1, header=None, sep='\s+')
    scale_factor = scale_factor.loc[0, 2]
    for i in '(gal)':
        scale_factor = scale_factor.replace(i, '')
    scale_factor_1 = int(scale_factor[0:4])
    scale_factor_2 = int(scale_factor[5:])
    return scale_factor_1,scale_factor_2

# 根据震级分类
def get_magnitude_seven_sort(path):
    file_m = pd.read_csv(path,header=None,skiprows=4,nrows=1,sep='\s+')
    m = int(file_m.loc[0,1])
    if m < 3:
        sort_m = 0
    elif m < 4:
        sort_m = 1
    elif m < 5:
        sort_m = 2
    elif m < 6:
        sort_m = 3
    elif m < 7:
        sort_m = 4
    elif m < 8:
        sort_m = 5
    else:
        sort_m = 6
    return sort_m

# 根据震级二分类
def get_magnitude_two_sort(path):
    file_m = pd.read_csv(path,header=None,skiprows=4,nrows=1,sep='\s+')
    m = float(file_m.loc[0,1])
    if m < 5.5:
        sort_m = 0
    else:
        sort_m = 1
    return sort_m

# 获取起点位置
def get_start_point(data1,data2,data3):
    data = np.array(data1)**2 + np.array(data2)**2 + np.array(data3)**2
    s = 0
    for i in range(len(data)):
        for t in range(0,8):
            s += data[i,t]
            if s / data.sum() >= 0.0005:
                return i,t

# 获取终点位置
def get_end_point(data1,data2,data3):
    data = np.array(data1)**2 + np.array(data2)**2 + np.array(data3)**2
    s = 0
    for i in range(len(data)-1,-1,-1):
        for t in range(7,-1,-1):
            s += data[i,t]
            if s / data.sum() >= 0.0005:
                return i,t

# 获取中间三点位置
def get_middle_point(rs,cs,re,ce,f):
    l = re*8+ce-rs*8-cs
    if f == 100:
        d = (l-400)/4
        r = d // 8
        c = d % 8
        r1 = rs + 25 + r
    else:
        d = (l-800)/4
        r = d // 8
        c = d % 8
        r1 = rs + 50 + r
    c1 = cs + c
    if c1 >= 8:
        r1 += 1
        c1 -= 8
    r2 = r1 + r
    c2 = c1 + c
    if c2 >= 8:
        r2 += 1
        c2 -= 8
    r3 = r2 + r
    c3 = c2 + c
    if c3 >= 8:
        r3 += 1
        c3 -= 8
    return r1,c1,r2,c2,r3,c3

# 获取起点数据
def get_start_record(data,row,col,frequence):
    per_data = []
    n = int(4 * frequence)
    data_np = np.array(data).reshape(8 * len(data))
    b,c = 0,0
    for a in data_np[8 * row + col:8 * row + col + n]:
        n_interval = 0.01 * frequence
        if b / n_interval >= c:
            per_data.extend([a])
            c = c + 1
        b = b + 1
    return per_data

# 获取中间点数据
def get_middle_record(data,row,col,frequence):
    per_data = []
    n = int(2 * frequence)
    data_np = np.array(data).reshape(8 * len(data))
    b,c = 0,0
    for a in data_np[int(8*row+col-n):int(8*row+col+n)]:
        n_interval = 0.01 * frequence
        if b / n_interval >= c:
            per_data.extend([a])
            c = c + 1
        b = b + 1
    return per_data

# 获取终点数据
def get_end_record(data,row,col,frequence):
    per_data = []
    n = int(4 * frequence)
    data_np = np.array(data).reshape(8 * len(data))
    b,c = 0,0
    for a in data_np[8 * row + col - n:8 * row + col]:
        n_interval = 0.01 * frequence
        if b / n_interval >= c:
            per_data.extend([a])
            c = c + 1
        b = b + 1
    return per_data

# 归一化
def nomalize(data):
    data = data/np.max(np.abs(data))
    return data

# 处理第一节数据
def write_record1(path):
    x_train1 = []
    x_test1 = []
    list_m = os.listdir(path)
    for m in list_m:
        list_record = os.listdir(path+m)
        for record in list_record:
            list_file = os.listdir(path+m+'/'+record)
            if record + '.EW1' in list_file:
                list_hz = ['.EW1', '.EW2', '.NS1', '.NS2', '.UD1', '.UD2']
            else:
                list_hz = ['.EW', '.NS', '.UD']
            for w in list_hz:
                globals()['path_'+w[1:]] = path+m+'/'+record+'/'+record+w
                datetime = get_datetime(globals()['path_'+w[1:]])
                frequence = get_frequence(globals()['path_'+w[1:]])
                duration = get_duration(globals()['path_'+w[1:]])
                globals()['data_'+w[1:]] = pd.read_csv(globals()['path_'+w[1:]],skiprows=17,nrows=frequence*duration//8,header=None,sep='\s+')
                globals()['data_'+w[1:]] -= np.mean(globals()['data_'+w[1:]])
                globals()['data_'+w[1:]] = nomalize(globals()['data_'+w[1:]])
            if '.EW' in list_hz:
                r1,c1 = get_start_point(data_EW,data_NS,data_UD)
                for w in list_hz:
                    per_data1 = get_start_record(globals()['data_'+w[1:]],r1,c1,frequence)
                    if datetime < 20150530202400:
                        x_train1.append(per_data1)
                    else:
                        x_test1.append(per_data1)
            else:
                r1,c1 = get_start_point(data_EW1,data_NS1,data_UD1)
                per_data1 = get_start_record(data_EW1,r1,c1,frequence)
                per_data2 = get_start_record(data_NS1,r1,c1,frequence)
                per_data3 = get_start_record(data_UD1,r1,c1,frequence)
                if datetime < 20150530202400:
                    x_train1.append(per_data1)
                    x_train1.append(per_data2)
                    x_train1.append(per_data3)
                else:
                    x_test1.append(per_data1)
                    x_test1.append(per_data2)
                    x_test1.append(per_data3)
                r1,c1 = get_start_point(data_EW2,data_NS2,data_UD2)
                per_data1 = get_start_record(data_EW2,r1,c1,frequence)
                per_data2 = get_start_record(data_NS2,r1,c1,frequence)
                per_data3 = get_start_record(data_UD2,r1,c1,frequence)
                if datetime < 20150530202400:
                    x_train1.append(per_data1)
                    x_train1.append(per_data2)
                    x_train1.append(per_data3)
                else:
                    x_test1.append(per_data1)
                    x_test1.append(per_data2)
                    x_test1.append(per_data3)
    with open('./5.5/magnitude_x_train1.pk','wb') as xtr1:
        pickle.dump(x_train1,xtr1)
    with open('./5.5/magnitude_x_test1.pk','wb') as xte1:
        pickle.dump(x_test1,xte1)

# 处理第二节数据
def write_record2(path):
    x_train2 = []
    x_test2 = []
    list_m = os.listdir(path)
    for m in list_m:
        list_record = os.listdir(path+m)
        for record in list_record:
            list_file = os.listdir(path+m+'/'+record)
            if record + '.EW1' in list_file:
                list_hz = ['.EW1', '.EW2', '.NS1', '.NS2', '.UD1', '.UD2']
            else:
                list_hz = ['.EW', '.NS', '.UD']
            for w in list_hz:
                globals()['path_'+w[1:]] = path+m+'/'+record+'/'+record+w
                datetime = get_datetime(globals()['path_'+w[1:]])
                frequence = get_frequence(globals()['path_'+w[1:]])
                duration = get_duration(globals()['path_'+w[1:]])
                globals()['data_'+w[1:]] = pd.read_csv(globals()['path_'+w[1:]],skiprows=17,nrows=frequence*duration//8,header=None,sep='\s+')
                globals()['data_'+w[1:]] -= np.mean(globals()['data_'+w[1:]])
                globals()['data_'+w[1:]] = nomalize(globals()['data_'+w[1:]])
            if '.EW' in list_hz:
                r1,c1 = get_start_point(data_EW,data_NS,data_UD)
                r5,c5 = get_end_point(data_EW,data_NS,data_UD)
                r2,c2,r3,c3,r4,c4 = get_middle_point(r1,c1,r5,c5,frequence)
                for w in list_hz:
                    per_data1 = get_middle_record(globals()['data_'+w[1:]],r2,c2,frequence)
                    if datetime < 20150530202400:
                        x_train2.append(per_data1)
                    else:
                        x_test2.append(per_data1)
            else:
                r1, c1 = get_start_point(data_EW1, data_NS1, data_UD1)
                r5, c5 = get_end_point(data_EW1, data_NS1, data_UD1)
                r2, c2, r3, c3, r4, c4 = get_middle_point(r1, c1, r5, c5, frequence)
                per_data1 = get_middle_record(data_EW1,r2,c2,frequence)
                per_data2 = get_middle_record(data_NS1,r2,c2,frequence)
                per_data3 = get_middle_record(data_UD1,r2,c2,frequence)
                if datetime < 20150530202400:
                    x_train2.append(per_data1)
                    x_train2.append(per_data2)
                    x_train2.append(per_data3)
                else:
                    x_test2.append(per_data1)
                    x_test2.append(per_data2)
                    x_test2.append(per_data3)
                r1, c1 = get_start_point(data_EW2, data_NS2, data_UD2)
                r5, c5 = get_end_point(data_EW2, data_NS2, data_UD2)
                r2, c2, r3, c3, r4, c4 = get_middle_point(r1, c1, r5, c5, frequence)
                per_data1 = get_middle_record(data_EW2,r2,c2,frequence)
                per_data2 = get_middle_record(data_NS2,r2,c2,frequence)
                per_data3 = get_middle_record(data_UD2,r2,c2,frequence)
                if datetime < 20150530202400:
                    x_train2.append(per_data1)
                    x_train2.append(per_data2)
                    x_train2.append(per_data3)
                else:
                    x_test2.append(per_data1)
                    x_test2.append(per_data2)
                    x_test2.append(per_data3)
    with open('./5.5/magnitude_x_train2.pk','wb') as xtr2:
        pickle.dump(x_train2,xtr2)
    with open('./5.5/magnitude_x_test2.pk','wb') as xte2:
        pickle.dump(x_test2,xte2)

# 处理第三节数据
def write_record3(path):
    x_train3 = []
    x_test3 = []
    list_m = os.listdir(path)
    for m in list_m:
        list_record = os.listdir(path+m)
        for record in list_record:
            list_file = os.listdir(path+m+'/'+record)
            if record + '.EW1' in list_file:
                list_hz = ['.EW1', '.EW2', '.NS1', '.NS2', '.UD1', '.UD2']
            else:
                list_hz = ['.EW', '.NS', '.UD']
            for w in list_hz:
                globals()['path_'+w[1:]] = path+m+'/'+record+'/'+record+w
                datetime = get_datetime(globals()['path_'+w[1:]])
                frequence = get_frequence(globals()['path_'+w[1:]])
                duration = get_duration(globals()['path_'+w[1:]])
                globals()['data_'+w[1:]] = pd.read_csv(globals()['path_'+w[1:]],skiprows=17,nrows=frequence*duration//8,header=None,sep='\s+')
                globals()['data_'+w[1:]] -= np.mean(globals()['data_'+w[1:]])
                globals()['data_'+w[1:]] = nomalize(globals()['data_'+w[1:]])
            if '.EW' in list_hz:
                r1,c1 = get_start_point(data_EW,data_NS,data_UD)
                r5,c5 = get_end_point(data_EW,data_NS,data_UD)
                r2,c2,r3,c3,r4,c4 = get_middle_point(r1,c1,r5,c5,frequence)
                for w in list_hz:
                    per_data1 = get_middle_record(globals()['data_'+w[1:]],r3,c3,frequence)
                    if datetime < 20150530202400:
                        x_train3.append(per_data1)
                    else:
                        x_test3.append(per_data1)
            else:
                r1, c1 = get_start_point(data_EW1, data_NS1, data_UD1)
                r5, c5 = get_end_point(data_EW1, data_NS1, data_UD1)
                r2, c2, r3, c3, r4, c4 = get_middle_point(r1, c1, r5, c5, frequence)
                per_data1 = get_middle_record(data_EW1,r3,c3,frequence)
                per_data2 = get_middle_record(data_NS1,r3,c3,frequence)
                per_data3 = get_middle_record(data_UD1,r3,c3,frequence)
                if datetime < 20150530202400:
                    x_train3.append(per_data1)
                    x_train3.append(per_data2)
                    x_train3.append(per_data3)
                else:
                    x_test3.append(per_data1)
                    x_test3.append(per_data2)
                    x_test3.append(per_data3)
                r1, c1 = get_start_point(data_EW2, data_NS2, data_UD2)
                r5, c5 = get_end_point(data_EW2, data_NS2, data_UD2)
                r2, c2, r3, c3, r4, c4 = get_middle_point(r1, c1, r5, c5, frequence)
                per_data1 = get_middle_record(data_EW2,r3,c3,frequence)
                per_data2 = get_middle_record(data_NS2,r3,c3,frequence)
                per_data3 = get_middle_record(data_UD2,r3,c3,frequence)
                if datetime < 20150530202400:
                    x_train3.append(per_data1)
                    x_train3.append(per_data2)
                    x_train3.append(per_data3)
                else:
                    x_test3.append(per_data1)
                    x_test3.append(per_data2)
                    x_test3.append(per_data3)
    with open('./5.5/magnitude_x_train3.pk','wb') as xtr3:
        pickle.dump(x_train3,xtr3)
    with open('./5.5/magnitude_x_test3.pk','wb') as xte3:
        pickle.dump(x_test3,xte3)

# 处理第四节数据
def write_record4(path):
    x_train4 = []
    x_test4 = []
    list_m = os.listdir(path)
    for m in list_m:
        list_record = os.listdir(path+m)
        for record in list_record:
            list_file = os.listdir(path+m+'/'+record)
            if record + '.EW1' in list_file:
                list_hz = ['.EW1', '.EW2', '.NS1', '.NS2', '.UD1', '.UD2']
            else:
                list_hz = ['.EW', '.NS', '.UD']
            for w in list_hz:
                globals()['path_'+w[1:]] = path+m+'/'+record+'/'+record+w
                datetime = get_datetime(globals()['path_'+w[1:]])
                frequence = get_frequence(globals()['path_'+w[1:]])
                duration = get_duration(globals()['path_'+w[1:]])
                globals()['data_'+w[1:]] = pd.read_csv(globals()['path_'+w[1:]],skiprows=17,nrows=frequence*duration//8,header=None,sep='\s+')
                globals()['data_'+w[1:]] -= np.mean(globals()['data_'+w[1:]])
                globals()['data_'+w[1:]] = nomalize(globals()['data_'+w[1:]])
            if '.EW' in list_hz:
                r1,c1 = get_start_point(data_EW,data_NS,data_UD)
                r5,c5 = get_end_point(data_EW,data_NS,data_UD)
                r2,c2,r3,c3,r4,c4 = get_middle_point(r1,c1,r5,c5,frequence)
                for w in list_hz:
                    per_data1 = get_middle_record(globals()['data_'+w[1:]],r4,c4,frequence)
                    if datetime < 20150530202400:
                        x_train4.append(per_data1)
                    else:
                        x_test4.append(per_data1)
            else:
                r1,c1 = get_start_point(data_EW1,data_NS1,data_UD1)
                r5,c5 = get_end_point(data_EW1,data_NS1,data_UD1)
                r2,c2,r3,c3,r4,c4 = get_middle_point(r1,c1,r5,c5,frequence)
                per_data1 = get_middle_record(data_EW1,r4,c4,frequence)
                per_data2 = get_middle_record(data_NS1,r4,c4,frequence)
                per_data3 = get_middle_record(data_UD1,r4,c4,frequence)
                if datetime < 20150530202400:
                    x_train4.append(per_data1)
                    x_train4.append(per_data2)
                    x_train4.append(per_data3)
                else:
                    x_test4.append(per_data1)
                    x_test4.append(per_data2)
                    x_test4.append(per_data3)
                r1, c1 = get_start_point(data_EW2, data_NS2, data_UD2)
                r5, c5 = get_end_point(data_EW2, data_NS2, data_UD2)
                r2, c2, r3, c3, r4, c4 = get_middle_point(r1, c1, r5, c5, frequence)
                per_data1 = get_middle_record(data_EW2,r4,c4,frequence)
                per_data2 = get_middle_record(data_NS2,r4,c4,frequence)
                per_data3 = get_middle_record(data_UD2,r4,c4,frequence)
                if datetime < 20150530202400:
                    x_train4.append(per_data1)
                    x_train4.append(per_data2)
                    x_train4.append(per_data3)
                else:
                    x_test4.append(per_data1)
                    x_test4.append(per_data2)
                    x_test4.append(per_data3)
    with open('./5.5/magnitude_x_train4.pk','wb') as xtr4:
        pickle.dump(x_train4,xtr4)
    with open('./5.5/magnitude_x_test4.pk','wb') as xte4:
        pickle.dump(x_test4,xte4)

# 处理第五节数据
def write_record5(path):
    x_train5 = []
    x_test5 = []
    list_m = os.listdir(path)
    for m in list_m:
        list_record = os.listdir(path+m)
        for record in list_record:
            list_file = os.listdir(path+m+'/'+record)
            if record + '.EW1' in list_file:
                list_hz = ['.EW1', '.EW2', '.NS1', '.NS2', '.UD1', '.UD2']
            else:
                list_hz = ['.EW', '.NS', '.UD']
            for w in list_hz:
                globals()['path_'+w[1:]] = path+m+'/'+record+'/'+record+w
                datetime = get_datetime(globals()['path_'+w[1:]])
                frequence = get_frequence(globals()['path_'+w[1:]])
                duration = get_duration(globals()['path_'+w[1:]])
                globals()['data_'+w[1:]] = pd.read_csv(globals()['path_'+w[1:]],skiprows=17,nrows=frequence*duration//8,header=None,sep='\s+')
                globals()['data_'+w[1:]] -= np.mean(globals()['data_'+w[1:]])
                globals()['data_'+w[1:]] = nomalize(globals()['data_'+w[1:]])
            if '.EW' in list_hz:
                r5,c5 = get_end_point(data_EW,data_NS,data_UD)
                for w in list_hz:
                    per_data1 = get_end_record(globals()['data_'+w[1:]],r5,c5,frequence)
                    if datetime < 20150530202400:
                        x_train5.append(per_data1)
                    else:
                        x_test5.append(per_data1)
            else:
                r5,c5 = get_end_point(data_EW1,data_NS1,data_UD1)
                per_data1 = get_end_record(data_EW1,r5,c5,frequence)
                per_data2 = get_end_record(data_NS1,r5,c5,frequence)
                per_data3 = get_end_record(data_UD1,r5,c5,frequence)
                if datetime < 20150530202400:
                    x_train5.append(per_data1)
                    x_train5.append(per_data2)
                    x_train5.append(per_data3)
                else:
                    x_test5.append(per_data1)
                    x_test5.append(per_data2)
                    x_test5.append(per_data3)
                r5,c5 = get_end_point(data_EW2,data_NS2,data_UD2)
                per_data1 = get_end_record(data_EW2,r5,c5,frequence)
                per_data2 = get_end_record(data_NS2,r5,c5,frequence)
                per_data3 = get_end_record(data_UD2,r5,c5,frequence)
                if datetime < 20150530202400:
                    x_train5.append(per_data1)
                    x_train5.append(per_data2)
                    x_train5.append(per_data3)
                else:
                    x_test5.append(per_data1)
                    x_test5.append(per_data2)
                    x_test5.append(per_data3)
    with open('./5.5/magnitude_x_train5.pk','wb') as xtr5:
        pickle.dump(x_train5,xtr5)
    with open('./5.5/magnitude_x_test5.pk','wb') as xte5:
        pickle.dump(x_test5,xte5)

# 处理标签
def write_labels(path):
    y_train = []
    y_test = []
    list_m = os.listdir(path)
    for m in list_m:
        list_record = os.listdir(path+m)
        for record in list_record:
            list_file = os.listdir(path+m+'/'+record)
            if record + '.EW1' in list_file:
                list_hz = ['.EW1', '.EW2', '.NS1', '.NS2', '.UD1', '.UD2']
            else:
                list_hz = ['.EW', '.NS', '.UD']
            for w in list_hz:
                globals()['path_'+w[1:]] = path+m+'/'+record+'/'+record+w
                datetime = get_datetime(globals()['path_'+w[1:]])
                per_sort_m = get_magnitude_two_sort(globals()['path_'+w[1:]])
                if datetime < 20150530202400:
                    y_train.append(per_sort_m)
                else:
                    y_test.append(per_sort_m)
    with open('./5.5/magnitude_y_train.pk','wb') as ytr:
        pickle.dump(y_train,ytr)
    with open('./5.5/magnitude_y_test.pk','wb') as yte:
        pickle.dump(y_test,yte)

# 读取数据
def read_magnitude():
    with open('./5.5/magnitude_x_train1.pk','rb') as xtr1:
        x_train1 = pickle.load(xtr1)
    with open('./5.5/magnitude_x_train2.pk','rb') as xtr2:
        x_train2 = pickle.load(xtr2)
    with open('./5.5/magnitude_x_train3.pk','rb') as xtr3:
        x_train3 = pickle.load(xtr3)
    with open('./5.5/magnitude_x_train4.pk','rb') as xtr4:
        x_train4 = pickle.load(xtr4)
    with open ('./5.5/magnitude_x_train5.pk','rb') as xtr5:
        x_train5 = pickle.load(xtr5)
    with open('./5.5/magnitude_x_test1.pk','rb') as xte1:
        x_test1 = pickle.load(xte1)
    with open('./5.5/magnitude_x_test2.pk','rb') as xte2:
        x_test2 = pickle.load(xte2)
    with open('./5.5/magnitude_x_test3.pk','rb') as xte3:
        x_test3 = pickle.load(xte3)
    with open('./5.5/magnitude_x_test4.pk','rb') as xte4:
        x_test4 = pickle.load(xte4)
    with open('./5.5/magnitude_x_test5.pk','rb') as xte5:
        x_test5 = pickle.load(xte5)
    with open('./5.5/magnitude_y_train.pk','rb') as ytr:
        y_train = pickle.load(ytr)
    with open('./5.5/magnitude_y_test.pk','rb') as yte:
        y_test = pickle.load(yte)
    return x_train1,x_train2,x_train3,x_train4,x_train5,x_test1,x_test2,x_test3,x_test4,x_test5,y_train,y_test

if __name__ == '__main__':
    write_record1('./datasets/')
    write_record2('./datasets/')
    write_record3('./datasets/')
    write_record4('./datasets/')
    write_record5('./datasets/')
    write_labels('./datasets/')