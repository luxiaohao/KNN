import numpy as np
import matplotlib.pyplot as plt
from KNN_1.datasets.data_utils import load_CIFAR10
from KNN_1.classifiers import K_Nearest_Neighbor
from KNN_1.datasets.data_utils import *


def Look_data():
    '''

    :return: 返回训练数据，测试数据的数据形状，及标检形状
              并把数据向量化，便于后面计算
    '''
    # plt画图属性参数，一是画图大小；二是插值方式最近邻；三是灰度空间
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
    # 指定dpi=200，图片尺寸为 1200*800
    # 指定dpi=300，图片尺寸为 1800*1200
    # 设置figsize可以在不改变分辨率情况下改变比例
    #plt.rcParams['savefig.dpi'] = 300 #图片像素
    #plt.rcParams['figure.dpi'] = 300 #分辨率


    # 载入cifar10数据
    cifar10_dir = '../KNN_1/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 查看数据形状
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)



    #显示数据集中一些数据可视化参考
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    #enumerate枚举
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.figure(1)
    plt.show()
    return X_train, y_train, X_test, y_test


def Verification_dists():
    # 定义验证3种方法所计算的距离矩阵是否一样
    dists_one = classifier.compute_distances_one_loop(X_test)

    difference = np.linalg.norm(dists - dists_one, ord='fro')
    print('Difference was: %f' % (difference, ))
    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')
    return difference


def time_function(f, *args):
    """

    :param f:  所选择方法
    :param args:  输入测试数据
    :return:  返回运行时间
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


def time_cost():
    """
    :return: 三种方法所花时间，2层循环居然比一层循环所花时间少
        Two loop version took 38.655800 seconds
        One loop version took 77.333827 seconds
        No loop version took 1.334549 seconds
    """

    two_loop_time = time_function(K_Nearest_Neighbor.KNearestNeighbor.compute_distances_two_loops, X_test)
    print('Two loop version took %f seconds' % two_loop_time)

    one_loop_time = time_function(K_Nearest_Neighbor.KNearestNeighbor.compute_distances_one_loop, X_test)
    print('One loop version took %f seconds' % one_loop_time)

    no_loop_time = time_function(K_Nearest_Neighbor.KNearestNeighbor.compute_distances_no_loops, X_test)
    print('No loop version took %f seconds' % no_loop_time)

    return two_loop_time, one_loop_time, no_loop_time




def Cross_validation():
    """
    定义交叉验证函数
    在实践中，人们倾向于避免交叉验证，而喜欢一次验证分割，因为交叉验证计算量很大。
    人们一般使用训练集的50%~90%作为真正的训练集，剩下的作为验证集。
    然而，这也取决于多个因素：比如如果超参数的数量很多，你可能倾向于使用更大的验证集。
    如果验证集的数目太少（只有几百个），那么使用交叉验证比较安全。
    通常的交叉验证为3重，5重，10重交叉验证。

。
    :return: 最优的K值

    """
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []

    """
    分割数据集，分割后y_train_folds[i]对应X_train_folds[i]
    """

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    # 字典形式保存，不同的K对应不同的精确度
    k_to_accuracies = {}
    for i in k_choices:
        k_to_accuracies[i] = []

    # 在每个验证集上运行不同的K，寻找使结果最优的K值
    for ki in k_choices:
        for fi in range(num_folds):
            """
            np.stack()矩阵堆叠函数，分水平和竖直方向。注意：除第0维外，形状要相同，列表数组都可以
            
            """
            valindex = fi
            X_traini = np.vstack((X_train_folds[0:fi] + X_train_folds[fi + 1:num_folds]))
            y_traini = np.hstack((y_train_folds[0:fi] + y_train_folds[fi + 1:num_folds]))

            X_vali = np.array(X_train_folds[valindex])
            y_vali = np.array(y_train_folds[valindex])
            num_val = len(y_vali)

            # 训练分类器
            classifier = K_Nearest_Neighbor.KNearestNeighbor()
            classifier.train(X_traini, y_traini)

            # 验证精确度
            dists = classifier.compute_distances_one_loop(X_vali)
            y_val_pred = classifier.predict_labels(dists, k=ki)
            num_correct = np.sum(y_val_pred == y_vali)
            accuracy = float(num_correct) / num_val
            print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
            k_to_accuracies[ki].append(accuracy)


    # 循环打印出各个K值在不同验证集上的精确度

    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))


    # 画出不同K值，可视化精确度图形，便于观察。
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # 用对应于标准偏差的误差条绘制趋势线
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    return k_to_accuracies



if __name__ == '__main__':
    cifar10_dir = '../KNN_1/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 数据处理，先用5000训练数据，和500测试数据看大体精度，若用全部的则精度大概38%
    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print(X_train.shape, X_test.shape)

    classifier = K_Nearest_Neighbor.KNearestNeighbor()
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances_two_loops(X_test)
    print(dists.shape)
    plt.imshow(dists, interpolation='none')
    plt.show()

    y_test_pred = classifier.predict_labels(dists, k=15)


    # 预测精度
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


    # 可以使用交叉验证寻找最优的K值
    # Cross_validation()
    """ 
    # 知道了最优的K值之后，用最优的K值预测数据
    best_k = 1

    classifier = K_Nearest_Neighbor.KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=best_k)

    # Compute and display the accuracy
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    """