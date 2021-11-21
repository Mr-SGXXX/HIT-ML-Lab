import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_NUM = 10  # 生成数据量
COV = 0  # 0时表示符合朴素贝叶斯，非零时不符合
SIGMA1 = 0.3  # x1轴方差
SIGMA2 = 0.2  # x2轴方差
MEAN1 = (5.8, 6.2)  # 高斯分布一均值
MEAN2 = (4.8, 4.2)  # 高斯分布二均值
LMD = 1e-3  # 正则项参数大小
MAX_TIME = 100  # 牛顿迭代最大迭代次数
DELTA = 1e-5  # 精度大小
PATH = "./blood.csv"


def generate_data(data_num, mean1, mean2, cov_mat):
    """
    生成数据
    :param data_num: 生成数据总量
    :param mean1: 生成正例的均值
    :param mean2: 生成反例的均值
    :param cov_mat: 协方差矩阵
    :return: 生成的样本点及对应的标签
    """
    data_x = np.zeros([data_num, 2])
    data_y = np.zeros(data_num)
    pos_num = data_num // 2 + data_num % 2
    data_x[0:pos_num, :] = np.random. \
        multivariate_normal(mean1, cov_mat, size=pos_num)
    data_x[pos_num:, :] = np.random. \
        multivariate_normal(mean2, cov_mat, size=data_num - pos_num)
    data_x = np.c_[data_x, np.ones(data_num)]
    data_y[0:pos_num] = 1
    return data_x, data_y


def load_data_from_UCI(path):
    """
    从UCI数据集文件中提出数据
    :param path: 数据集路径
    :return: UCI数据点及对应的标签
    """
    data = pd.read_csv(path)
    data = np.array(data)
    return np.c_[data[:, 0:data.shape[1] - 1], np.ones(data.shape[0])], data[:, data.shape[1] - 1]


def newton_iterate(data_x, data_y, func_grad, func_hessian, func_loss, max_time, delta):
    """
    牛顿迭代法求最优化参数
    :param data_x: 训练数据属性
    :param data_y: 训练数据标签
    :param func_grad: 损失值函数梯度
    :param func_hessian: 损失值函数黑森矩阵
    :param func_loss: 损失值函数
    :param max_time: 最大迭代次数
    :param delta: 精度
    :return: 模型最优化参数，每次迭代损失值
    """
    w0 = np.zeros(data_x.shape[1])
    loop_time = 0
    l = func_loss(data_x, data_y, w0)
    while True:
        w = w0 - np.linalg.inv(func_hessian(data_x, data_y, w0)). \
            dot(func_grad(data_x, data_y, w0))
        nl = func_loss(data_x, data_y, w)
        l = np.c_[l, nl]
        if np.linalg.norm(func_grad(data_x, data_y, w)) < delta or loop_time >= max_time:
            break
        loop_time += 1
        w0 = w
        if loop_time % 10 == 0:
            print("轮数：" + str(loop_time) + "\t Loss:" + str(nl))
    print("迭代结束")
    return w, l.reshape((l.shape[1],))


def test(x_test, y_test, weight):
    """
    测试模型准确率
    :param x_test: 测试数据的属性
    :param y_test: 测试数据的标签
    :param weight: 模型权重
    :return: 模型测试的准确率
    """
    right = 0
    for i in range(x_test.shape[0]):
        if sigmoid(weight @ x_test[i, :]) >= 0.5:
            predict = 1
        else:
            predict = 0
        if predict == y_test[i]:
            right += 1
    return right / x_test.shape[0]


def sigmoid(z):
    """
    Sigmoid函数
    :param z: 自变量
    :return: 因变量
    """
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))


def grad(data_x, data_y, w):
    """
    带有正则项的对率回归的梯度函数
    :param data_x: 测试数据属性
    :param data_y: 测试数据标签
    :param w: 模型参数
    :return: 对率回归的梯度函数值
    """
    rst = np.zeros(data_x.shape[1])
    for i in range(data_x.shape[0]):
        rst -= data_x[i, :] * (data_y[i] - sigmoid(w.T @ data_x[i, :]))
    return rst + LMD * w


def hessian(data_x, data_y, w):
    """
    带有正则项的对率回归的黑森矩阵函数
    :param data_x: 测试数据属性
    :param data_y: 测试数据标签
    :param w: 模型参数
    :return: 对率回归的黑森矩阵函数值
    """
    rst = np.zeros((data_x.shape[1], data_x.shape[1]))
    for i in range(data_x.shape[0]):
        p = sigmoid(w.T.dot(data_x[i, :]))
        xi = data_x[i, :].reshape((data_x.shape[1], 1))
        rst += xi.dot(xi.T) * p * (1 - p)
    return rst + LMD * np.eye(rst.shape[0])


def loss(data_x, data_y, w):
    """
    带正则项对率回归损失函数
    :param data_x: 测试数据属性
    :param data_y: 测试数据标签
    :param w: 模型参数
    :return: 对率回归的损失函数值
    """
    rst = 0
    for i in range(data_x.shape[0]):
        z = w @ data_x[i, :]
        rst += -data_y[i] * z + (np.log(1 + np.exp(z)) if z < 0 else z + np.log(1 + np.exp(-z)))
    return rst + LMD * np.linalg.norm(w) / 2


def grad_noreg(data_x, data_y, w):
    """
    不带有正则项的对率回归的梯度函数
    :param data_x: 测试数据属性
    :param data_y: 测试数据标签
    :param w: 模型参数
    :return: 对率回归的梯度函数值
    """
    rst = np.zeros(data_x.shape[1])
    for i in range(data_x.shape[0]):
        rst += data_x[i, :] * (data_y[i] - sigmoid(w.T @ data_x[i, :]))
    return -rst


def hessian_noreg(data_x, data_y, w):
    """
    不带有正则项的对率回归的黑森矩阵函数
    :param data_x: 测试数据属性
    :param data_y: 测试数据标签
    :param w: 模型参数
    :return: 对率回归的黑森矩阵函数值
    """
    rst = np.zeros((data_x.shape[1], data_x.shape[1]))
    for i in range(data_x.shape[0]):
        p = sigmoid(w.T @ data_x[i, :])
        xi = data_x[i, :].reshape((data_x.shape[1], 1))
        rst += xi @ xi.T * p * (1 - p)
    return rst


def loss_noreg(data_x, data_y, w):
    """
    无正则项对率回归损失函数
    :param data_x: 测试数据属性
    :param data_y: 测试数据标签
    :param w: 模型参数
    :return: 对率回归的损失函数值
    """
    rst = 0
    for i in range(data_x.shape[0]):
        z = w @ data_x[i, :]
        rst += -data_y[i] * z + (np.log(1 + np.exp(z)) if z < 0 else z + np.log(1 + np.exp(-z)))
    return rst


if __name__ == "__main__":
    # 二维高斯分布分类
    x, y = generate_data(DATA_NUM, MEAN1, MEAN2, np.asarray([[SIGMA1, COV], [COV, SIGMA2]]))
    x_test, y_test = generate_data(100000, MEAN1, MEAN2, [[SIGMA1, COV], [COV, SIGMA2]])
    weight1, l1 = newton_iterate(x, y, grad, hessian, loss, MAX_TIME, DELTA)  # 有正则
    weight2, l2 = newton_iterate(x, y, grad_noreg, hessian_noreg, loss_noreg, MAX_TIME, DELTA)  # 无正则
    point_1a = (0, -weight1[2] / weight1[1])
    point_2a = (-weight1[2] / weight1[0], 0)
    point_1b = (0, -weight2[2] / weight2[1])
    point_2b = (-weight2[2] / weight2[0], 0)

    # 绘图
    plt.figure(figsize=(10, 7))
    plt.plot(x[0:DATA_NUM // 2, 0], x[0:DATA_NUM // 2, 1], 'b.', label='pos-data')
    plt.plot(x[DATA_NUM // 2:, 0], x[DATA_NUM // 2:, 1], 'r.', label='neg-data')
    plt.legend(loc='upper right')
    plt.title("Data Points Number = " + str(DATA_NUM))
    plt.show()
    plt.figure(figsize=(10, 7))
    plt.axline(point_1b, point_2b, color='y', label='predict-line-no-reg')
    plt.axline(point_1a, point_2a, color='k', label='predict-line-with-reg')
    plt.plot(x[0:DATA_NUM // 2, 0], x[0:DATA_NUM // 2, 1], 'b.', label='pos-data')
    plt.plot(x[DATA_NUM // 2:, 0], x[DATA_NUM // 2:, 1], 'r.', label='neg-data')
    plt.legend(loc='upper right')
    plt.title("Data Points Number = " + str(DATA_NUM) + "\nAccuracy of no reg:" + str(
        test(x_test, y_test, weight2)) + "\nAccuracy of with reg:" + str(
        test(x_test, y_test, weight1)))
    plt.show()

    # 损失函数随迭代次数变化曲线

    plt.title("predict-line-no-reg")
    plt.plot(l2, 'r-')
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.show()
    plt.title("predict-line-with-reg")
    plt.plot(l1, 'b-')
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.show()

    # UCI数据集
    x, y = load_data_from_UCI(PATH)
    weight, l = newton_iterate(x[0::2], y[0::2], grad, hessian, loss, MAX_TIME, DELTA)
    plt.title("UCI-Data\nAccuracy:" + str(test(x[1::2], y[1::2], weight)))
    plt.plot(l, 'k-')
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.show()
