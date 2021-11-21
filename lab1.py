import math

import numpy as np
import matplotlib.pyplot as plt

DATA_NUM = 10  # 数据量
SIGMA = 0.1  # 生成数据方差
LMD = 1e-3  # 惩罚项参数
LEARN_RATE = 100  # 梯度下降法初始学习率
MIN_LEARN_RATE = 0.01  # 梯度下降法最小学习率
LEARN_TIME = -1  # 最大学习次数，值为-1时不限制最大次数
DELTA = 1e-6  # 梯度下降法精度
DEF_ORDER = 9  # 默认阶数
MAX_ORDER = 15  # 多项式的最大阶数


def get_data(data_num=DATA_NUM, start=0, end=1, sigma=SIGMA, func=lambda x: np.sin(2 * np.pi * x)):
    """
    根据函数及生成数据量，求得对应横纵坐标数据
    :param data_num:数据量
    :param start:生成数据区间起点
    :param end:生成数据区间终点
    :param sigma:生成数据方差
    :param func:生成数据公式
    :return:横坐标X与纵坐标Y的向量
    """
    X = np.linspace(start, end, data_num)
    Y = func(X) + np.random.normal(0, sigma, X.shape)
    return X, Y


def cal_mat_X(sin_X, order):
    """
    计算X矩阵
    :param sin_X: 横坐标X向量
    :param order: 多项式阶数
    :return: X矩阵
    """
    mat_X = np.ones((sin_X.shape[0], 1))
    for n in range(1, order + 1):
        mat_X = np.c_[mat_X, np.power(sin_X, n)]
    return mat_X


def cal_weight_withoutreg(mat_X, sin_Y):
    """
    求解参数不带正则项的解析解
    :param mat_X: 范德蒙德矩阵X
    :param sin_Y: 训练样本纵坐标Y向量
    :return:参数向量不带正则项的解析解
    """
    mat_X_trans = np.transpose(mat_X)
    return np.linalg.inv(np.dot(mat_X_trans, mat_X)).dot(mat_X_trans).dot(sin_Y)


def cal_weight_withreg(mat_X, sin_Y, lmd=LMD):
    """
    求解参数带有惩罚项的解析解
    :param mat_X: 范德蒙德矩阵X
    :param sin_Y: 训练样本纵坐标Y向量
    :param lmd: 正则项参数
    :return: 参数向量带有正则项的解析解
    """
    mat_X_trans = np.transpose(mat_X)
    return np.linalg.inv(np.dot(mat_X_trans, mat_X) + lmd * np.eye(mat_X.shape[1])).dot(mat_X_trans).dot(sin_Y)


def cal_weight_graddown(mat_X, sin_Y, order, lmd=LMD, learn_rate=LEARN_RATE, min_learn_rate=MIN_LEARN_RATE,
                        learn_time=LEARN_TIME, delta=DELTA):
    """
    梯度下降法求解带有惩罚项的解
    :param mat_X: 范德蒙德矩阵X
    :param sin_Y: 训练样本纵坐标Y向量
    :param order: 多项式阶数
    :param lmd: 正则项参数大小
    :param learn_rate: 初始学习率
    :param min_learn_rate: 最小学习率
    :param learn_time: 最大学习次数，当次数为-1时不做限制
    :param delta: 精度
    :return: 迭代次数与梯度下降算出的参数向量带有正则项的解
    """
    times = 0
    A = mat_X.T.dot(mat_X) + lmd * np.eye(mat_X.shape[1])
    b = mat_X.T.dot(sin_Y)
    # 初始参数随机选择
    weight = np.random.rand(order + 1)
    loss = (weight.T.dot(mat_X.T).dot(mat_X).dot(weight) + lmd * weight.T.dot(weight)) / 2
    while learn_time > 0 or learn_time == -1:
        times += 1
        old_loss = loss
        weight -= learn_rate * (A.dot(weight) - b)
        loss = (weight.T.dot(mat_X.T).dot(mat_X).dot(weight) + lmd * weight.T.dot(weight)) / 2
        if old_loss < loss and learn_rate > min_learn_rate:  # 出现损失值上升时，学习率减半
            learn_rate /= 2
        # if abs(old_loss - loss) < delta:  # 前后损失值差小于精度则退出循环
        #     break
        if np.sum(np.power((A.dot(weight) - b), 2)) < delta:  # 损失值梯度小于精度则退出循环
            break
        if learn_time != -1:
            learn_time -= 1
    #     if times % 1000 == 0:
    #         print("轮数：" + str(times))
    # print("迭代结束")
    return times, weight


def cal_weight_congrad(mat_X, sin_Y, order, lmd=LMD, delta=DELTA):
    """
    共轭梯度法求解带有惩罚项的解
    :param mat_X: 范德蒙德矩阵X
    :param sin_Y: 训练样本纵坐标Y向量
    :param order: 多项式阶数
    :param lmd: 正则项参数大小
    :param delta: 精度
    :return: 迭代次数和共轭梯度法算出的参数向量带有正则项的解
    """
    times = 0
    # 初始化权值与残差
    weight = np.random.rand(order + 1)
    A = mat_X.T.dot(mat_X) + lmd * np.eye(mat_X.shape[1])
    b = mat_X.T.dot(sin_Y)
    g = A.dot(weight) - b
    p = -g
    while True:
        times += 1
        alpha = g.T.dot(g) / p.T.dot(A).dot(p)
        weight = weight + alpha * p
        new_g = g + alpha * A.dot(p)
        beta = new_g.T.dot(new_g) / g.T.dot(g)
        p = -new_g + beta * p
        if new_g.T.dot(new_g) < delta:
            break
        g = new_g
    return times, weight.reshape(order + 1)


def cal_ERMS(sin_Y: np.ndarray, rst_Y: np.ndarray):
    """
    计算拟合结果的ERMS损失值
    :param sin_Y: 原数据的Y向量
    :param rst_Y: 根据拟合后多项式计算出的Y向量
    :return:ERMS损失值
    """
    data_num = sin_Y.shape[0]
    return math.sqrt(np.sum(np.power(rst_Y - sin_Y, 2)) / data_num)


def find_best_lmd(sin_train_X, sin_train_Y, sin_test_Y, order=DEF_ORDER):
    """
    画出带有正则项的解析解关于系数的对数的图像，并求得其最佳值
    :param sin_train_X: 训练集数据横轴向量
    :param sin_train_Y: 训练集数据纵轴向量
    :param sin_test_Y: 测试集纵轴向量
    :param order: 所求多项式的阶数
    :return:使得ERMS最小的正则项系数
    """
    mat_X = cal_mat_X(sin_train_X, order)
    best_lnlmd = 0
    best_lmd_ERMS = 1e100
    list_lnlmd = []
    list_ERMS_test = []
    list_ERMS_train = []

    for lnlmd in range(-50, 1):
        weight_reg = cal_weight_withreg(mat_X, sin_train_Y, math.exp(lnlmd))
        rst_train_X_reg, rst_train_Y_reg = get_data(DATA_NUM, sigma=0, func=np.poly1d(weight_reg[::-1]))
        rst_test_X_reg, rst_test_Y_reg = get_data(100, sigma=0, func=np.poly1d(weight_reg[::-1]))
        list_ERMS_train.append(cal_ERMS(sin_train_Y, rst_train_Y_reg))
        ERMS = cal_ERMS(sin_test_Y, rst_test_Y_reg)
        list_ERMS_test.append(ERMS)
        list_lnlmd.append(lnlmd)
        if ERMS < best_lmd_ERMS:
            best_lnlmd = lnlmd
            best_lmd_ERMS = ERMS
    print("正则项参数最合适值为：e^" + str(best_lnlmd) + "即：" + str(math.exp(best_lnlmd)))
    plt.title("ERMS:Different Lambda")
    plt.xlabel("lnlmd")
    plt.ylabel("ERMS")
    plt.plot(list_lnlmd, list_ERMS_train, 'b.-', label="Train Set")
    plt.plot(list_lnlmd, list_ERMS_test, 'r.-', label="Test Set")
    plt.legend(loc='upper right')
    plt.show()
    return math.exp(best_lnlmd)


def show_rst(data_num=DATA_NUM, max_order=MAX_ORDER, lmd=LMD, index=-1):
    """
    显示ERMS关于lnlmd的图像，并计算出最佳lmd值，显示计算结果，包括真实曲线、阶数从0到9的拟合后的曲线及训练和测试的ERMS图像
    :param data_num: 数据量大小
    :param max_order: 最大的阶数
    :param lmd: 所有带有正则项方法的正则项系数，若为-1表示自动选择最优值
    :param index: 特别查看某些维度的图像，等于空元组时，显示最大阶数为止的所有阶数图像
    """

    sin_train_X, sin_train_Y = get_data(DATA_NUM)  # 得到训练数据
    sin_test_X, sin_test_Y = get_data(100)  # 得到测试数据
    sin_origin_X, sin_origin_Y = get_data(100, sigma=0)  # 得到真实曲线

    # 绘制关于lnlmd的ERMS图像，如果参数lmd为-1，则用所求的最佳lmd值赋值
    lmd_best = find_best_lmd(sin_train_X, sin_train_Y, sin_test_Y)
    if lmd == -1:
        lmd = lmd_best

    # ERMS图像所需向量初始化
    list_order = []
    list_ERMS_test_noreg = []
    list_ERMS_test_reg = []
    list_ERMS_test_graddown = []
    list_ERMS_test_congrad = []
    list_ERMS_train_noreg = []
    list_ERMS_train_reg = []
    list_ERMS_train_graddown = []
    list_ERMS_train_congrad = []
    if index == ():
        plt.figure(figsize=(12, 8))

    # for order in range(0, min((data_num, max_order + 1))):
    for order in (1, 3, 9):
        mat_X = cal_mat_X(sin_train_X, order)
        # 各方法根据拟合后的多项式计算训练数据横坐标对应的纵坐标
        # 无正则项解析解
        weight_noreg = cal_weight_withoutreg(mat_X, sin_train_Y)
        rst_train_X_noreg, rst_train_Y_noreg = get_data(DATA_NUM, sigma=0, func=np.poly1d(weight_noreg[::-1]))
        # 有正则项解析解
        weight_reg = cal_weight_withreg(mat_X, sin_train_Y, lmd=lmd)
        rst_train_X_reg, rst_train_Y_reg = get_data(DATA_NUM, sigma=0, func=np.poly1d(weight_reg[::-1]))
        # 梯度下降法
        times_graddown, weight_graddown = cal_weight_graddown(mat_X, sin_train_Y, order, lmd=lmd)
        rst_train_X_graddown, rst_train_Y_graddown = get_data(DATA_NUM, sigma=0, func=np.poly1d(weight_graddown[::-1]))
        # 共轭梯度法
        times_congrad, weight_congrad = cal_weight_congrad(mat_X, sin_train_Y, order, lmd=lmd)
        rst_train_X_congrad, rst_train_Y_congrad = get_data(DATA_NUM, sigma=0, func=np.poly1d(weight_congrad[::-1]))

        # 使用100个点计算各阶数下的测试数据ERMS
        rst_test_X_noreg, rst_test_Y_noreg = get_data(100, sigma=0, func=np.poly1d(weight_noreg[::-1]))
        rst_test_X_reg, rst_test_Y_reg = get_data(100, sigma=0, func=np.poly1d(weight_reg[::-1]))
        rst_test_X_graddown, rst_test_Y_graddown = get_data(100, sigma=0, func=np.poly1d(weight_graddown[::-1]))
        rst_test_X_congrad, rst_test_Y_congrad = get_data(100, sigma=0, func=np.poly1d(weight_congrad[::-1]))

        # 用于绘制ERMS图的数据
        list_order.append(order)
        list_ERMS_train_noreg.append(cal_ERMS(sin_train_Y, rst_train_Y_noreg))
        list_ERMS_train_reg.append(cal_ERMS(sin_train_Y, rst_train_Y_reg))
        list_ERMS_train_graddown.append(cal_ERMS(sin_train_Y, rst_train_Y_graddown))
        list_ERMS_train_congrad.append(cal_ERMS(sin_train_Y, rst_train_Y_congrad))
        list_ERMS_test_noreg.append(cal_ERMS(sin_test_Y, rst_test_Y_noreg))
        list_ERMS_test_reg.append(cal_ERMS(sin_test_Y, rst_test_Y_reg))
        list_ERMS_test_graddown.append(cal_ERMS(sin_test_Y, rst_test_Y_graddown))
        list_ERMS_test_congrad.append(cal_ERMS(sin_test_Y, rst_test_Y_congrad))

        # 打印信息
        print("-" * 25)
        print("阶数：" + str(order) + "\t训练集规模：" + str(data_num))
        print("梯度下降法迭代次数：" + str(times_graddown))
        print("共轭梯度法迭代次数：" + str(times_congrad))
        print("-" * 25 + "\n")

        # 图像显示部分
        if index == ():
            plt.legend(loc='upper right')
            plt.subplot((min((data_num, max_order + 1)) + 2) // 4, 4, order + 1)
            plt.title("order=" + str(order))
            plt.plot(sin_train_X, sin_train_Y, 'k.', label="Sample Points")
            plt.plot(sin_origin_X, sin_origin_Y, 'b', label="True Function")
            plt.plot(rst_test_X_noreg, rst_test_Y_noreg, 'r', label="No Regular Terms")
            plt.plot(rst_test_X_reg, rst_test_Y_reg, 'g', label="With Regular Terms")
            plt.plot(rst_test_X_graddown, rst_test_Y_graddown, 'c', label="Gradient Descent")
            plt.plot(rst_test_X_congrad, rst_test_Y_congrad, 'y', label="Conjugate Gradient")
        elif order in index:
            plt.figure(figsize=(15, 8))
            plt.title("order=" + str(order) + "    train data number =" + str(data_num))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(sin_train_X, sin_train_Y, 'k.', label="Sample Points")
            plt.plot(sin_origin_X, sin_origin_Y, 'b', label="True Function")
            plt.plot(rst_test_X_noreg, rst_test_Y_noreg, 'r', label="No Regular Terms")
            plt.plot(rst_test_X_reg, rst_test_Y_reg, 'g', alpha=0.5, linewidth=1, label="With Regular Terms")
            plt.plot(rst_test_X_graddown, rst_test_Y_graddown, 'c-.', alpha=0.6, linewidth=3, label="Gradient Descent")
            plt.plot(rst_test_X_congrad, rst_test_Y_congrad, 'y--', alpha=0.6, linewidth=5, label="Conjugate Gradient")
            plt.legend(loc='upper right')
            plt.show()
    if index == ():
        plt.show()
    plt.title("ERMS:No Regular Terms")
    plt.xlabel("M")
    plt.ylabel("ERMS")
    plt.plot(list_order, list_ERMS_train_noreg, 'b.-', label="Train Set")
    plt.plot(list_order, list_ERMS_test_noreg, 'r.-', label="Test Set")
    plt.legend(loc='upper right')
    plt.show()
    plt.title("ERMS:With Regular Terms")
    plt.xlabel("M")
    plt.ylabel("ERMS")
    plt.plot(list_order, list_ERMS_train_reg, 'b.-', label="Train Set")
    plt.plot(list_order, list_ERMS_test_reg, 'r.-', label="Test Set")
    plt.legend(loc='upper right')
    plt.show()
    plt.title("ERMS:Gradient Descent")
    plt.xlabel("M")
    plt.ylabel("ERMS")
    plt.plot(list_order, list_ERMS_train_graddown, 'b.-', label="Train Set")
    plt.plot(list_order, list_ERMS_test_graddown, 'r.-', label="Test Set")
    plt.legend(loc='upper right')
    plt.show()
    plt.title("ERMS:Conjugate Gradient")
    plt.xlabel("M")
    plt.ylabel("ERMS")
    plt.plot(list_order, list_ERMS_train_congrad, 'b.-', label="Train Set")
    plt.plot(list_order, list_ERMS_test_congrad, 'r.-', label="Test Set")
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    # show_rst(lmd=-1, index=(1, 3, 9))
    show_rst(lmd=-1, index=(1, 3, 9))
