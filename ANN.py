import random
import numpy as np
#import matplotlib.pyplot as plt
def Sigmoid(z):
    return 1./(1.+np.exp(z))
def Transfrom(t):
    return t.flatten()
def Paint2(t):
    temp = t.tolist()
    temp1 = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.]
    for i in range(36):
        temp1[i] = temp[0][i]
    p = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.]
    for i in range(36):
        p[i] = i+1
    print(temp1)
    plt.plot(p,temp1)
    plt.show()
def Paint2(t):
    temp = t.tolist()
    temp1 = [1.,2.]
    for i in range(2):
        temp1[i] = temp[0][i]
    p = [0,0,]
    for i in range(2):
        p[i] = i+1
    print(temp1)
    plt.plot(p,temp1)
    plt.show()
    #绘出编码图
def Prime_Sigmoid(z):
    #sigmoid函数的导数，反向求的时候会用到，basis的减量就是导数，weigh的减量是导数乘对应的输入量
    return Sigmoid(z)*(1-Sigmoid(z))
class NeuralNetwork(object):
    def __init__(self,sizes):
         #sizes 为各层的特征
        self.layers = len(sizes)
        self.sizes = sizes
        #总共的层数
        self.basis = [np.random.randn(1, y) for y in sizes[1:]]
        #basis随机生成
        self.weigh = [np.random.randn(x, y) for x, y in zip(sizes[:-1],sizes[1:])]
    def Feed_Forward(self,in_put):
        #前馈，用于结果比较部分，通过一个输入计算到最后，可以比较点，看点是否一样。
        for b, w in zip(self.basis, self.weigh):
            in_put = Sigmoid(np.dot(in_put, w)+b)
        return in_put
    def Stochastic_Gradient_Descent(self,training_data, epochs, mini_training_size, learing_rate, test_data):
        #epochs为训练次数，mini_traing_size为stochastic里面的最小分组
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            #打乱训练样本
            mini_training_samples = [training_data[k:k+mini_training_size] for k in range(0,n,mini_training_size)]
            #创造临时样本集
            for mini_training_sample in mini_training_samples:
                self.Train(mini_training_sample, learing_rate)
            #暂时不考虑有测试集的情况
            if test_data:                                                           #之后这里与评价有关暂时不知道是干啥的
                print ("测试集 {0}:正确率为 {1} / {2}".format(                           #反应的应该是测试的成功率，其实可视化是可以多做的一个部分
                    i, self.Evaluate(test_data), n_test))
            else:
                print ("测试集 {0} 训练完毕,无训练集检测正确率".format(i+1))
    def Train(self, mini_training_sample, learing_rate):
        new_b_list = [np.zeros(b.shape) for b in self.basis]
        new_w_list = [np.zeros(w.shape) for w in self.weigh]
        for x,y in mini_training_sample:
            #其中x是样本即图像，y为点向量（1*2）
            #对于最小训练集当中的每一个训练对象,先去求他的BP减少量
            #Paint(x)
            temp1 = self.Feed_Forward(x)
            print(temp1)
#            if len(temp1[0])>1: Paint2(temp1)
            delta_b , delta_w = self.Back_Propagation(x,y)
            new_b_list = [nb+dnb for nb, dnb in zip(new_b_list, delta_b)]
            new_w_list = [nw+dnw for nw, dnw in zip(new_w_list, delta_w)]
            #计算每一层的更新量
            self.basis = [b-(learing_rate/len(mini_training_sample))*nb for b, nb in zip(self.basis, new_b_list)]
            self.weigh = [w-(learing_rate/len(mini_training_sample))*nw for w, nw in zip(self.weigh, new_w_list)]
            #更新weigh,basis
    def Back_Propagation(self,x,y):
        new_b_list = [np.zeros(b.shape) for b in self.basis]
        new_w_list = [np.zeros(w.shape) for w in self.weigh]
        in_put = x
        in_puts = [x]
        calculations = []
        for b, w in zip(self.basis, self.weigh):
            calculation = np.dot(in_put, w) + b
            calculations.append(calculation)
            in_put = Sigmoid(calculation)
            in_puts.append(in_put)
        #in_put的最后一个得到也是输出序列，即两个点的坐标
        delta = self.cost(in_puts[-1], y) * Prime_Sigmoid(calculations[-1])
#        print(in_puts)
        #这里得到所求的导数，之后对b不变，对w乘一个系数即可
        new_b_list[-1] = delta
        new_w_list[-1] = np.dot(in_puts[-2].transpose(),delta)
        #Back_Propagation这里为先赋值，对输出层
        for j in range(2, self.layers):
            calculation = calculations[-1*j]
            #先找到最后的计算结果
            temp = Sigmoid(calculation)
            delta = np.dot(delta, self.weigh[-j+1].transpose()) * temp
            new_b_list[-j] = delta
            new_w_list[-j] = np.dot(in_puts[-j-1].transpose(), delta)
        #每一层更新，返回每一层需要的delta
        return (new_b_list, new_w_list)
    def Print_temp_result(self,in_put):
        print(self.Feed_Forward(in_put))
    def Input_Test_Data(self):
        test_matrix = input()
        test_label = input()
        return test_matrix, test_label
    def cost(self, out_put, y):                                                        #
        return (out_put - y)
    def Evaluate(self,test_data):
        cals= [self.Feed_Forward(x)-y for (x,y) in test_data]
        cnt = 0
        for (x,y) in test_data:
            #Paint(x)
            temp1 = self.Feed_Forward(x)
            print(temp1)
        if len(temp1[0])>1: Paint2(temp1)
        for cal in cals:
            sum = cal[0][0]*cal[0][0]+cal[0][1]*cal[0][1]                    #这里对误差的评判也是基于对测试得到的点和label的点之间的距离计算得到的。
            sum = np.sqrt(sum)
            if sum==0:
                cnt+=1
        return cnt
#Test NeuralNetwork Building AND FeedForward Calculate
#p = NeuralNetwork([3,2,1])
#t = np.ones(3).reshape(1,3)
#k = p.Feed_Forward(t)
#print(k)
#Sample Training Data
t1_data = [249,240,244,247.5,243,238,245.5,244,244,244,232.5,232.5,235.5,239,236,236.5,236.5,231.5,240,238,233,234,239,236,228.5,222.5,219,218,221,219,217,206.5,206.5,206.5,206.5,221]
t2_data = [36,59.5,101,140.5,182,209,234,269,301,301,305.5,287,260,237,214,185,185,165,162.5,179.5,208,235,264,291,323.5,361,396,422.5,456,495.5,534.5,567.5,567.5,567.5,567.5,577.5]
t1_res = [1,0]
t2_res = [1,0]
TR1 = np.zeros(36).reshape(1,36)
TR2 = np.zeros(36).reshape(1,36)
TR1ZB = np.zeros(2).reshape(1,2)
TR2ZB = np.zeros(2).reshape(1,2)
for i in range(36):
    TR1[0][i] = t1_data[i]/100
    TR2[0][i] = t2_data[i]/100
for i in range(2):
    TR1ZB[0][i] = t1_res[i]/100
    TR2ZB[0][i] = t2_res[i]/100
Test_data = []
Test_data.append((TR1,TR1ZB))
Test_data.append((TR2,TR2ZB))
p = NeuralNetwork([36,18,2])
p.Stochastic_Gradient_Descent(Test_data,10,2,0.4,None)
