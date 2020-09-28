from pandas import read_csv
from os import listdir
from numpy import zeros,reshape,abs,array,log,e,linspace,set_printoptions,inf,sqrt,multiply,mean,real
from numpy.random import normal
from numpy.fft import fft,fftshift,fftfreq,ifft,ifftshift
from scipy import interpolate,fftpack
from matplotlib.pyplot import plot,show,xlabel,ylabel,legend
import processing
import pickle as pk
from math import atan,pi
set_printoptions(threshold=inf)

freqs = [0.05,1/15,1/14,1/13,1/12,1/11,0.1,1/9.5,1/9,1/8.5,1/8,1/7.5,1/7,1/6.5,1/6,1/5.5,0.2,1/4.8,1/4.6,1/4.4,1/4.2,0.25,
        1/3.8,1/3.6,1/3.5,1/3.4,1/3.2,1/3,1/2.8,1/2.6,0.4,1/2.4,1/2.2,0.5,1/1.9,1/1.8,1/1.7,1/1.6,1/1.5,1/1.4,1/1.3,1/1.2,
        1/1.1,1,1/0.95,1/0.9,1/0.85,1/0.8,1/0.75,1/0.7,1/0.667,1/0.65,1/0.6,1/0.55,2,1/0.48,1/0.46,1/0.45,1/0.44,1/0.42,2.5,
        1/0.38,1/0.36,1/0.35,1/0.34,1/0.32,1/0.3,1/0.29,1/0.28,1/0.26,4,1/0.24,1/0.22,5,1/0.19,1/0.18,1/0.17,1/0.16,1/0.15,
        1/0.14,1/0.133,1/0.13,1/0.12,1/0.11,10,1/0.095,1/0.09,1/0.085,1/0.08,1/0.075,1/0.07,1/0.067,1/0.065,1/0.06,1/0.055,
        1/0.05,1/0.048,1/0.046,1/0.045,1/0.044,1/0.042,1/0.04,1/0.036,1/0.035,1/0.032,1/0.03,1/0.029,40,1/0.022]#50,100
periods = [20,15, 14,13,12,11,10,9.5,9,8.5,8,7.5,7,6.5,6,5.5,5,4.8,4.6,4.4,4.2,4,3.8,3.6,3.5,3.4,3.2,3,2.8,2.6,2.5,2.4,2.2,
           2,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1,0.95,0.9,0.85,0.8,0.75,0.7,0.667,0.65,0.6,0.55,0.5,0.48,0.46,0.45,0.44,
           0.42,0.4,0.38,0.36,0.35,0.34,0.32,0.3,0.29,0.28,0.26,0.25,0.24,0.22,0.2,0.19,0.18,0.17,0.16,0.15,0.14,0.133,0.13,
           0.12,0.11,0.1,0.095,0.09,0.085,0.08,0.075,0.07,0.067,0.065,0.06,0.055,0.05,0.048,0.046,0.045,0.044,0.042,0.04,
           0.036,0.035,0.032,0.03,0.029,0.025,0.022]

# 相位
def get_phase(path):
    m_list = listdir(path)
    phase_x = zeros([109,109])
    phase_d = zeros([109,109])
    phase_difference = zeros([109,109])
    xn = 0
    dn = 0
    for m in m_list:
        records_list = listdir(path + m)
        for record in records_list:
            file_list = listdir(path + m + '/' + record)
            if record + '.EW' in file_list:
                for w in ['.EW', '.NS', '.UD']:
                    print(xn, dn)
                    paths = path + m + '/' + record + '/' + record + w
                    frequence = processing.get_frequence(paths)
                    duration = processing.get_duration(paths)
                    number = frequence * duration
                    file_m = read_csv(paths, header=None, skiprows=4, nrows=1, sep='\s+')
                    magnitude = int(file_m.loc[0, 1])
                    if number % 8 != 0:
                        file = read_csv(paths, skiprows=17, header=None, nrows=number // 8, sep='\s+')
                        file = reshape(array(file), number // 8 * 8)
                        four = read_csv(paths, skiprows=17 + number // 8, header=None, nrows=1, sep='\s+')
                        file = list(file)
                        for i in range(4):
                            file.extend([four.loc[0, i]])
                        data = array(file)
                    else:
                        file = read_csv(paths, sep='\s+', skiprows=17, header=None, keep_default_na=False)
                        data = reshape(array(file), number)
                    data_flat = reshape(array(data), number)
                    f1, f2 = processing.get_scale_factor(paths)
                    data_flat = data_flat * f1 / f2
                    data_flat = data_flat - mean(data_flat)
                    f_freq = fftfreq(number, 1 / frequence)
                    f_freq_half = f_freq[range(int(number / 2))]
                    data_fft = fft(data_flat)
                    data_fft_half = data_fft[range(int(number/2))]
                    data_fft_half = abs(data_fft_half-real(data_fft_half))/real(data_fft_half)
                    f = interpolate.interp1d(f_freq_half, data_fft_half, kind='linear')
                    pt = f(freqs)
                    for n in range(109):
                        pt[n] = atan(pt[n])/pi
                    for i in range(109):
                        for j in range(109):
                            phase_difference[i,j] = abs(pt[j]-pt[i])
                    if magnitude < 5.5:
                        xn += 1
                        for i in range(109):
                            for j in range(109):
                                phase_x[i,j] = phase_x[i,j] + phase_difference[i,j]
                    else:
                        dn += 1
                        for i in range(109):
                            for j in range(109):
                                phase_d[i,j] = phase_d[i,j] + phase_difference[i,j]
            else:
                for w in ['.EW1','.EW2','.NS1','.NS2','.UD1','.UD2']:
                    print(xn, dn)
                    paths = path + m + '/' + record + '/' + record + w
                    frequence = processing.get_frequence(paths)
                    duration = processing.get_duration(paths)
                    number = frequence * duration
                    file_m = read_csv(paths, header=None, skiprows=4, nrows=1, sep='\s+')
                    magnitude = int(file_m.loc[0, 1])
                    if number % 8 != 0:
                        file = read_csv(paths, skiprows=17, header=None, nrows=number // 8, sep='\s+')
                        file = reshape(array(file), number // 8 * 8)
                        four = read_csv(paths, skiprows=17 + number // 8, header=None, nrows=1, sep='\s+')
                        file = list(file)
                        for i in range(4):
                            file.extend([four.loc[0, i]])
                        data = array(file)
                    else:
                        file = read_csv(paths, sep='\s+', skiprows=17, header=None, keep_default_na=False)
                        data = reshape(array(file), number)
                    data_flat = reshape(array(data), number)
                    f1, f2 = processing.get_scale_factor(paths)
                    data_flat = data_flat * f1 / f2
                    data_flat = data_flat - mean(data_flat)
                    f_freq = fftfreq(number, 1 / frequence)
                    f_freq_half = f_freq[range(int(number / 2))]
                    data_fft = fft(data_flat)
                    data_fft_half = data_fft[range(int(number / 2))]
                    data_fft_half = abs(data_fft_half-real(data_fft_half))/real(data_fft_half)
                    f = interpolate.interp1d(f_freq_half, data_fft_half, kind='linear')
                    pt = f(freqs)
                    for n in range(109):
                        pt[n] = atan(pt[n])/pi
                    for i in range(109):
                        for j in range(109):
                            phase_difference[i,j] = abs(pt[j]-pt[i])
                    if magnitude < 5.5:
                        xn += 1
                        for i in range(109):
                            for j in range(109):
                                phase_x[i, j] = phase_x[i, j] + phase_difference[i, j]
                    else:
                        dn += 1
                        for i in range(109):
                            for j in range(109):
                                phase_d[i, j] = phase_d[i, j] + phase_difference[i, j]
    dphase = phase_d/dn
    xphase = phase_x/xn
    with open('./phase/dphase.pk','wb') as dp:
        pk.dump(dphase,dp)
    with open('./phase/xphase.pk','wb') as xp:
        pk.dump(xphase,xp)


def get_alpha(path):
    m_list = listdir(path)
    ats_x = zeros([109])
    ats_d = zeros([109])
    xn = 0
    dn = 0
    for m in m_list:
        records_list = listdir(path+m)
        for record in records_list:
            file_list = listdir(path+m+'/'+record)
            if record+'.EW' in file_list:
                for w in ['.EW','.NS','.UD']:
                    print(xn,dn)
                    paths = path + m + '/' + record + '/' + record + w
                    frequence = processing.get_frequence(paths)
                    duration = processing.get_duration(paths)
                    number = frequence*duration
                    file_m = read_csv(paths, header=None, skiprows=4, nrows=1, sep='\s+')
                    magnitude = int(file_m.loc[0, 1])
                    if number % 8 != 0:
                        file = read_csv(paths, skiprows=17, header=None, nrows=number // 8, sep='\s+')
                        file = reshape(array(file), number // 8 * 8)
                        four = read_csv(paths, skiprows=17 + number // 8, header=None, nrows=1, sep='\s+')
                        file = list(file)
                        for i in range(4):
                            file.extend([four.loc[0, i]])
                        data = array(file)
                    else:
                        file = read_csv(paths, sep='\s+', skiprows=17, header=None, keep_default_na=False)
                        data = reshape(array(file), number)
                    data_flat = reshape(array(data),number)
                    f1,f2 = processing.get_scale_factor(paths)
                    data_flat = data_flat*f1/f2
                    data_flat = data_flat-mean(data_flat)
                    f_freq =  fftfreq(number,1/frequence)
                    f_freq_half = f_freq[range(int(number/2))]
                    data_fft = fft(data_flat)
                    data_fft_half = abs(data_fft[range(int(number/2))])
                    f = interpolate.interp1d(f_freq_half,data_fft_half,kind='linear')
                    at = f(freqs)
                    at_log = log(abs(at))
                    if magnitude < 5.5:
                        xn += 1
                        for i in range(109):
                            ats_x[i] = ats_x[i] + at_log[i]
                    else:
                        dn += 1
                        for i in range(109):
                            ats_d[i] = ats_d[i] + at_log[i]
            else:
                for w in ['.EW1','.EW2','.NS1','.NS2','.UD1','.UD2']:
                    print(xn,dn)
                    paths = path + m + '/' + record + '/' + record + w
                    frequence = processing.get_frequence(paths)
                    duration = processing.get_duration(paths)
                    number = frequence*duration
                    file_m = read_csv(paths, header=None, skiprows=4, nrows=1, sep='\s+')
                    magnitude = int(file_m.loc[0, 1])
                    if number % 8 != 0:
                        file = read_csv(paths, skiprows=17, header=None, nrows=number // 8, sep='\s+')
                        file = reshape(array(file), number // 8 * 8)
                        four = read_csv(paths, skiprows=17 + number // 8, header=None, nrows=1, sep='\s+')
                        file = list(file)
                        for i in range(4):
                            file.extend([four.loc[0, i]])
                        data = array(file)
                    else:
                        file = read_csv(paths, sep='\s+', skiprows=17, header=None, keep_default_na=False)
                        data = reshape(array(file), number)
                    data_flat = reshape(array(data),number)
                    f1,f2 = processing.get_scale_factor(paths)
                    data_flat = data_flat*f1/f2
                    data_flat = data_flat-mean(data_flat)
                    f_freq =  fftfreq(number,1/frequence)
                    f_freq_half = f_freq[range(int(number/2))]
                    data_fft = fft(data_flat)
                    data_fft_half = abs(data_fft[range(int(number/2))])
                    f = interpolate.interp1d(f_freq_half,data_fft_half,kind='linear')
                    at = f(freqs)
                    at_log = log(abs(at))
                    if magnitude < 5.5:
                        xn += 1
                        for i in range(109):
                            ats_x[i] = ats_x[i] + at_log[i]
                    else:
                        dn += 1
                        for i in range(109):
                            ats_d[i] = ats_d[i] + at_log[i]
    alpha = ats_d/dn-ats_x/xn
    dalpha = ats_d/dn
    xalpha = ats_x/xn
    for i in range(109):
        dalpha[i] = e**dalpha[i]
    for i in range(109):
        xalpha[i] = e**xalpha[i]
    for i in range(109):
        alpha[i] = e**alpha[i]
    with open('./alpha/lin_alpha1.pk','wb') as a:
        pk.dump(alpha,a)
    with open('./alpha/lin_dalpha1.pk','wb') as da:
        pk.dump(dalpha,da)
    with open('./alpha/lin_xalpha1.pk','wb') as xa:
        pk.dump(xalpha,xa)


alpha_lin = [69.20605812,88.58265895,88.54681441,93.77024269,99.862341,97.4761523,104.93543325,106.92108488,112.83098327,
             118.18717618,122.48853863,127.87299931,129.91324563,126.59544682,129.75105717,126.93578281,130.19548216,130.7844142,
             133.29426975,127.12924473,128.37133967,130.85899028,127.43112262,127.41705145,129.84089351,129.65673616,127.40979684,
             125.00789609,114.21271965,115.9729795,116.57705423,112.84402088,103.4280003,96.63276591,89.13403812,80.63950893,
             77.36895084,72.64405551,65.3844417,58.57786145,51.9712505,47.01409738,40.95400847,33.73672424,31.93459553,28.81461262,
             25.84374577,23.23853077,21.63110778,18.90593215,17.32093203,15.99233563,14.32300599,12.38341672,10.1665041,
             9.59685565,8.74702624,8.63745717,8.1184336,7.51994212,6.78149203,6.14185022,5.59184569,5.32464036,5.05562355,
             4.66444775,4.15440392,3.92765856,3.68983265,3.19424304,2.95597972,2.82124035,2.3362663,1.9867108,1.84464637,
             1.7217963,1.56385616,1.44098591,1.282259,1.16863397,1.07429702,1.04147625,0.92139708,0.82027278,0.79271225,0.70190059,
             0.67354343,0.64688462,0.61719314,0.59401011,0.58037769,0.56291919,0.55766144,0.53467035,0.53007647,0.58088635,
             0.51843879,0.51506999,0.52042343,0.524131,0.52714134,0.5503456,0.5484209,0.56504734,0.59990749,0.64485655,0.68680285,
             1.20385532,2.31444754]


def get_generation(path):
    duration = processing.get_duration(path)
    frequence = processing.get_frequence(path)
    number = duration*frequence
    noise = normal(0, scale=1, size=(number))
    noise_fft = fft(noise)
    noise_freq = fftfreq(number, 1/frequence)
    new_freq = list(noise_freq.copy())
    for i in freqs:
        if i in noise_freq:
            continue
        else:
            for n in range(len(new_freq)):
                if i > new_freq[n - 1] and i < new_freq[n]:
                    new_freq[n:n] = (i,)
                    break
    f = interpolate.interp1d(noise_freq, noise_fft)
    new_fft = f(new_freq)
    for i in freqs:
        new_fft[list(new_freq).index(i)] *= alpha_lin[list(freqs).index(i)]
    for n in range(len(new_freq)):
        if new_freq[n] < 0:
            half_new_fft = new_fft[0:n]
            break
    half_new_fft = list(half_new_fft)
    half_new_fft.extend(half_new_fft)
    scale_noise = ifft(half_new_fft, n=number)
    f1, f2 = processing.get_scale_factor(path)
    if number % 8 != 0:
        file = read_csv(path, skiprows=17, header=None, nrows=number // 8, sep='\s+')
        file = reshape(array(file), number // 8 * 8)
        four = read_csv(path, skiprows=17 + number // 8, header=None, nrows=1, sep='\s+')
        file = list(file)
        for i in range(4):
            file.extend([four.loc[0, i]])
        data = array(file) * f1 / f2
    else:
        file = read_csv(path, sep='\s+', skiprows=17, header=None, keep_default_na=False)
        data = reshape(array(file * f1 / f2), number)
    bl = []
    bl_x = []
    for i in range(number):
        if data[i] == max(data[max(i - 100, 0):min(number, i + 100)]):
            bl.extend([data[i]])
            bl_x.extend([i])
    bl_x = array(bl_x) / 100
    bl = list(bl)
    bl_x = list(bl_x)
    bl[0:0] = (0,)
    bl[len(bl) - 1:len(bl) - 1] = (0,)
    bl_x[0:0] = (0,)
    bl_x[len(bl_x) - 1:len(bl_x) - 1] = (duration,)
    x = linspace(0, duration, number)
    f = interpolate.interp1d(bl_x, bl)
    y = f(x)
    y_nom = abs(y) / max(abs(y))
    y_scale = multiply(y_nom, abs(scale_noise))
    data_nom = data / max(abs(data))
    generation = 0.1 * y_scale / max(abs(y_scale)) + data_nom
    return generation,frequence


# 获取起点位置
def get_start_point(data):
    s = 0
    data2 = data**2
    for i in range(len(data2)):
        s += data2[i]
        if s / data2.sum() >= 0.0005:
            return i

# 获取终点位置
def get_end_point(data):
    s = 0
    data2 = data**2
    for i in range(len(data2)-1,-1,-1):
        s += data2[i]
        if s / data2.sum() >= 0.0005:
            return i

# 获取中间三点位置
def get_middle_point(n1,n5,f):
    l = n5 - n1
    if f == 100:
        d = (l-400)/4
        n2 = n1 + d
        n3 = n2 + d
        n4 = n3 + d
    else:
        d = (l-800)/4
        n2 = n1 + d
        n3 = n2 + d
        n4 = n3 + d
    return n2,n3,n4

# 获取起点数据
def get_start_record(data,n1,frequence):
    per_data = []
    n = int(4 * frequence)
    b,c = 0,0
    for a in data[n1:n1+n]:
        n_interval = 0.01 * frequence
        if b / n_interval >= c:
            per_data.extend([a])
            c = c + 1
        b = b + 1
    return per_data

# 获取中间点数据
def get_middle_record(data,n_,frequence):
    per_data = []
    n = int(2 * frequence)
    b,c = 0,0
    for a in data[int(n_-n):int(n_+n)]:
        n_interval = 0.01 * frequence
        if b / n_interval >= c:
            per_data.extend([a])
            c = c + 1
        b = b + 1
    return per_data

# 获取终点数据
def get_end_record(data,n5,frequence):
    per_data = []
    n = int(4 * frequence)
    b,c = 0,0
    for a in data[n5-n:n5]:
        n_interval = 0.01 * frequence
        if b / n_interval >= c:
            per_data.extend([a])
            c = c + 1
        b = b + 1
    return per_data

def split_generation(data,frequence):
    n1 = get_start_point(data)
    n5 = get_end_point(data)
    n2,n3,n4 = get_middle_point(n1,n5,frequence)
    data1 = get_start_record(data,n1,frequence)
    data2 = get_middle_record(data,n2,frequence)
    data3 = get_middle_record(data,n3,frequence)
    data4 = get_middle_record(data,n4,frequence)
    data5 = get_end_record(data,n5,frequence)
    return data1,data2,data3,data4,data5



