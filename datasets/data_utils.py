import numpy as np
import os
import pickle

#def load_pickle(f):



def load_CIFAR10_batch(filname):
    #定义一个batch读取数据
    #从数据文件中读取数据，并转换为python的数据结构,只读+二进制
    with open(filname, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')#python编码问题
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")#转置形状B,H,W,C
        Y = np.array(Y)
        return X, Y




def load_CIFAR10(ROOT):
    #定义加载数据函数

    xs = []
    ys = []
    for n in range(1, 6):
        filname = os.path.join(ROOT, 'data_batch_%d' %(n, ))
        X, Y = load_CIFAR10_batch(filname)
        xs.append(X)
        ys.append(Y)
    #向量化，把每个数据变成一个行向量(50000, 3072),(10000,)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR10_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte