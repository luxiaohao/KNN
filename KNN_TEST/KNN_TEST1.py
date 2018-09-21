import numpy as np
import matplotlib.pyplot as plt
from KNN_1.datasets.data_utils import load_CIFAR10
from KNN_1.classifiers import K_Nearest_Neighbor
from KNN_1.datasets.data_utils import *

from pandas import read_csv
import pandas as pd


class KNearestNeighbor(object):
    """定义KNN分类器，用L1，L2距离计算"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练数据，k近邻算法训练过程只是保存训练数据。

        :param X: 输入是一个包含训练样本nun_train，和每个样本信息的维度D的二维数组
        :param y: 对应标签的一维向量，y[i] 是 X[i]的标签
        :return: 无
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """

        :param X:  训练数据输入值，一个二维数组
        :param k:  确定K值进行预测投票
        :param num_loops: 选择哪一种循环方式，计算训练数据和测试数据之间的距离
        :return: 返回一个测试数据预测的向量(num_test,)，y[i] 是训练数据 X[i]的预测标签。
        """

        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        """
        计算距离没有运用循环。

        只使用基本的数组操作来实现这个函数，特别是不使用SCIPY的函数。
        提示：尝试使用矩阵乘法和两个广播求和来制定L2距离。

        :param X: 输入是一个(num_test, D)的训练数据。
        :return: 返回值是一个(num_test, num_train)的二维数组，dists[i, j]对应相应位置测试数据和训练数据的距离。
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        """
        M = np.dot(X, self.X_train.T)
        nrow = M.shape[0]
        ncol = M.shape[1]
        te = np.diag(np.dot(X, X.T))
        tr = np.diag(np.dot(self.X_train, self.X_train.T))
        te = np.reshape(np.repeat(te, ncol), M.shape)
        tr = np.reshape(np.repeat(tr, nrow), M.T.shape)
        sq = -2 * M + te + tr.T
        dists = np.sqrt(sq)

        #这里利用numpy的broadcasting性质，例如A = [1, 2], B = [[3], [4]], A + B = [[3 + 1, 3 + 2], [4 + 1, 4 + 2]]。
        #以及(a - b) ^ 2 = a ^ 2 + b ^ 2 - 2ab。

        """
        test_sum = np.sum(np.square(X), axis=1, keepdims=True)
        train_sum = np.sum(np.square(self.X_train), axis=1)
        test_mul_train = np.matmul(X, self.X_train.T)
        dists = test_sum + train_sum - 2 * test_mul_train

        return dists

    def compute_distances_one_loop(self, X):
        """

        应用1层循环的计算方式，计算测试数据和每个训练数据之间的距离。
        按训练数据的行索引计算，少了一个循环。

        :param X: 输入是一个(num_test, D)的训练数据。
        :return: 返回值是一个(num_test, num_train)的二维数组，dists[i, j]对应相应位置测试数据和训练数据的距离。

        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # axis = 1按每个行索引计算
            distances = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1))
            # L1距离
            # distances = np.sum(np.abs(self.X_train - X[i, :]), axis=1)

            dists[i, :] = distances

        return dists

    def compute_distances_two_loops(self, X):
        '''

        应用2层循环（嵌套循环）的计算方式，计算测试数据和每个训练数据之间的距离。

        :param X: 输入是一个(num_test, D)的训练数据。
        :return: 返回值是一个(num_test, num_train)的二维数组，dists[i, j]对应相应位置测试数据和训练数据的距离。

        '''

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                # 应用L2距离求解每个对应测试集数据和训练集数据的距离，并放在dists数组中，两层循环时间复杂度高
                distances = np.sqrt(np.sum(np.square(self.X_train[j] - X[i])))
                dists[i, j] = distances
        return dists

    def predict_labels(self, dists, k=1):
        '''

        输入一个测试数据和训练数据的距离矩阵，预测训练数据的标签。

        :param dists: 距离矩阵(num_test, num_train) 对应dists[i, j]
                       给出第i个测试数据和第j个训练数据的距离。
        :param k: KNN算法超参数K
        :return:返回(num_test,)向量，y[i]是X[i]对应的预测标签。

        '''

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        """
        这一步，要做的是：用距离矩阵找到与测试集最近的K个训练数据的距离，
        并且找到他们对应的标签，存放在closest_y中。

        函数介绍：
        numpy.argsort(a, axis=-1, kind=’quicksort’, order=None) 
        功能: 将矩阵a按照axis排序，并返回排序后的下标 
        参数: a:输入矩阵， axis:需要排序的维度 
        返回值: 输出排序后的下标，从小到大排
        """

        for i in range(num_test):
            # 创建一个长度为K的列表，用来存放与测试数据最近的K个训练数据的距离。
            closest_y = []

            distances = dists[i, :]
            indexes = np.argsort(distances)

            # 返回对应索引的标签值
            closest_y = self.y_train[indexes[: k]]
            """
            # 增加程序,解决维度过深问题，上面closest_y是2维
            if np.shape(np.shape(closest_y))[0] != 1:
                closest_y = np.squeeze(closest_y)
            """

            print(closest_y.astype(np.int))
            """
            通过上一步得到了最近的K个训练样本的标签，下一步就是找到其中最多的那个标签，
            并且把这个标签给预测值y_pred[i]。

            计算所有数字的出现次数
            numpy.bincount(x, weights=None, minlength=None)统计对应标签位置出现次数

            np.bincount(y_train.astype(np.int32))
            np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
            array([1, 3, 1, 1, 0, 0, 0, 1], dtype=int32)
            分别统计0-7分别出现的次数
            """
            count = np.bincount(closest_y.astype(np.int))
            # 返回最大位置的索引
            y_pred[i] = np.argmax(count)

        return y_pred


def Write(KNN):

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'K近邻': KNN})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(''r'C:\Users\Administrator\Desktop\ML_test\分类结果_1112.csv', index=False, sep=',')


if __name__ == '__main__':
    # 分别导入训练集和测试集数据,以csv的文件形式
    filename = ''r'C:\Users\Administrator\Desktop\ML_test\train1.csv'
    dataset = read_csv(filename, header=None)
    #测试集
    filename1 = ''r'C:\Users\Administrator\Desktop\ML_test\test.csv'
    dataset1 = read_csv(filename1, header=None)

    #若数据集里有空值，则删除该行数据
    dataset1.dropna(inplace=True)
    #获得数据集值，分别定义训练、测试数据的变量
    array = dataset.values
    X = array[:, 0:3].astype(np.float64)
    Y = array[:, 3]

    array1 = dataset1.values
    X1 = array1[:, 0:3].astype(np.float64)
    Y1 = array1[:, 3]

    x_train = X
    x_test = X1
    y_train = Y
    y_test = Y1

    classifier = K_Nearest_Neighbor.KNearestNeighbor()
    classifier.train(x_train, y_train)
    dists = classifier.compute_distances_two_loops(x_test)
    print(dists.shape)
    plt.imshow(dists, interpolation='none')
    plt.show()

    num_test = 411
    classifier = KNearestNeighbor()
    classifier.train(x_train, y_train)
    dists = classifier.compute_distances_two_loops(x_test)
    y_test_pred = classifier.predict_labels(dists, k=15)

    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

    """
    print(y_test_pred)
    #Write(KNearestNeighbor())"""