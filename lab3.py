import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

NUMBER = 4
MEAN1 = [NUMBER, NUMBER]
MEAN2 = [NUMBER, 3 * NUMBER]
MEAN3 = [3 * NUMBER, NUMBER]
MEAN4 = [3 * NUMBER, 3 * NUMBER]
RATE = (0.3, 0.5, 0.1, 0.1)
COV_MAT = (
    np.mat([[2, 0], [0, 2]]), np.mat([[2, 0], [0, 2]]), np.mat([[2, 0], [0, 2]]),
    np.mat([[2, 0], [0, 2]]))
DATA_NUM = 1000
TYPE_NUM = 4
MAX_SIZE = sys.maxsize
MAX_TIMES = 100
DELTA = 1e-8
UCI_flag = False
UCI_DATA = "./lab3/Iris.csv"


def generate_data(data_num=DATA_NUM, cov_mat=COV_MAT, mean=(MEAN1, MEAN2, MEAN3, MEAN4), rate=RATE):
    """
    生成数据点
    :param data_num: 数据量
    :param cov_mat: 协方差矩阵
    :param mean: 均值
    :param rate: 各高斯分布占比
    :return: 数据点及其标签
    """
    data = np.zeros([data_num, 2])
    data_label = np.ones(data_num)
    start = 0
    for i in range(len(mean)):
        num = int(data_num * rate[i])
        end = num + start
        data[start:end, :] = np.random.multivariate_normal(mean[i], cov_mat[i], size=num)
        data_label[start:end] = i * data_label[start:end]
        start = end
    return data, data_label


def load_UCI_data(file_path=UCI_DATA):
    """
    从UCI数据文件中加载数据（数据的标签必须在第一列）
    :param file_path: 文件路径
    :return: 数据及其标签
    """
    data = pd.read_csv(file_path, header=None)
    data = np.array(data)
    labels = []
    data_label = data[:, 0]
    new_data_label = np.zeros_like(data_label)
    for i in range(data.shape[0]):
        if not data_label[i] in labels:
            labels.append(data_label[i])
        new_data_label[i] = labels.index(data_label[i])
    return data[:, 1:], new_data_label


def cal_eu_dist(point1, point2):
    """
    计算两数据点间欧式距离
    :param point1: 数据点一
    :param point2: 数据点二
    :return: 两点间欧式距离
    """
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


def cal_cluster_min_dist(type1, type2, cal_point_dist_func=cal_eu_dist):
    """
    计算两簇间最短距离
    :param type1: 簇一
    :param type2: 簇二
    :param cal_point_dist_func: 两点间距离计算公式
    :return: 两簇间最短距离
    """
    min = float('inf')
    for i in range(type1.shape[0]):
        for j in range(type2.shape[0]):
            temp = cal_point_dist_func(type1[i], type2[j])
            if temp < min:
                min = temp
    return min


def cal_cluster_max_dist(type1, type2, cal_point_dist_func=cal_eu_dist):
    """
    计算两簇间最大距离
    :param type1: 簇一
    :param type2: 簇二
    :param cal_point_dist_func: 两点间距离计算公式
    :return: 两簇间最大距离
    """
    max = 0
    for i in range(type1.shape[0]):
        for j in range(type2.shape[0]):
            temp = cal_point_dist_func(type1[i], type2[j])
            if temp > max:
                max = temp
    return max


def cal_cluster_mean_dist(type1, type2, cal_point_dist_func=cal_eu_dist):
    """
    计算两簇间平均距离
    :param type1: 簇一
    :param type2: 簇二
    :param cal_point_dist_func: 两点间距离计算公式
    :return: 两簇间平均距离
    """
    sum = 0
    for i in range(type1.shape[0]):
        for j in range(type2.shape[0]):
            sum += cal_point_dist_func(type1[i], type2[j])
    return sum / (type1.shape[0] * type2.shape[0])


def level_clustering(data, type_num=TYPE_NUM, cal_cluster_dist_func=cal_cluster_mean_dist):
    """
    进行层次聚类
    :param data: 原始数据
    :param type_num: 目标聚类类别数
    :param cal_cluster_dist_func: 计算两簇间距函数
    :return: 类别与数据点的映射，各类中心点位置
    """
    # 初始化
    type_map = []
    cluster_dist = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        type_map.append(data[i])
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            cluster_dist[i, j] = cal_cluster_dist_func(type_map[i], type_map[j])
            cluster_dist[j, i] = cluster_dist[i, j]
    cur_type_num = data.shape[0]

    while cur_type_num > type_num:
        # 找到最近两簇
        min_dist = float('inf')
        min_i = 0
        min_j = 0
        for i in range(cur_type_num):
            for j in range(i + 1, cur_type_num):
                if cluster_dist[i, j] < min_dist:
                    min_i = i
                    min_j = j
                    min_dist = cluster_dist[i, j]
        # 合并簇
        type_map[min_i] = np.c_[type_map[min_i], type_map[min_j]]
        del type_map[min_j]

        # 重新计算簇间长度、
        cluster_dist = np.delete(cluster_dist, min_j, axis=0)
        cluster_dist = np.delete(cluster_dist, min_j, axis=1)
        cur_type_num -= 1
        for j in range(cur_type_num):
            cluster_dist[min_i, j] = cal_cluster_dist_func(type_map[min_i].T, type_map[j].T)
            cluster_dist[j, min_i] = cluster_dist[min_i, j]
    centers = np.zeros((type_num, data.shape[1]))
    for i in range(type_num):
        type_map[i] = type_map[i].T
    for i in range(type_num):
        centers[i] = np.mean(type_map[i], axis=0)
    return type_map, centers


def k_means(data: np.ndarray, centers, cal_dist_func=cal_eu_dist):
    """
    K-means方法聚类
    :param data: 数据点
    :param centers: 初始各簇中心位置
    :param cal_dist_func: 距离计算函数
    :return: 类别与数据点的映射，各类中心点位置
    """
    data_num = data.shape[0]
    type_map = []
    first_flag = []
    type_num = centers.shape[0]
    # check_label = []
    # centers = np.zeros([type_num, data.shape[1]])
    new_centers = np.zeros([type_num, data.shape[1]])
    for i in range(type_num):
        type_map.append(0)
        first_flag.append(True)
    #     # 初始化中心位置
    #     for j in range(data_num):
    #         if not data_label[j] in check_label:
    #             centers[i, :] = data[j, :]
    #             check_label.append(data_label[j])
    #             break

    # 使用层次聚类结果初始化中心位置

    while True:
        # 计算各点到各中心点距离，从而计算各点的类别
        for i in range(data_num):
            min_length = MAX_SIZE
            for j in range(type_num):
                dist = cal_dist_func(data[i], centers[j])
                if min_length > dist:
                    min_length = dist
                    type = j
            if first_flag[type]:
                type_map[type] = data[i].reshape([1, data.shape[1]])
                first_flag[type] = False
            else:
                type_map[type] = np.r_[type_map[type], data[i].reshape([1, data.shape[1]])]
        # 更新各类中心点位置
        for i in range(type_num):
            temp = np.zeros(data.shape[1])
            for j in range(type_map[i].shape[0]):
                temp = temp + type_map[i][j]
            temp /= type_map[i].shape[0]
            new_centers[i, :] = temp
        if (new_centers == centers).all():
            break
        else:
            centers = new_centers
        for i in range(type_num):
            type_map[i] = 0
            first_flag[i] = True
    return type_map, centers


def GMM_EM(data, map, means, max_times=MAX_TIMES, delta=DELTA):
    """
    GMM-EM聚类
    :param data: 数据点
    :param map: 初始聚类情况
    :param means: 初始各簇中心位置
    :param max_times: 最大迭代次数
    :param delta: 精度
    :return: 类别与数据点的映射，各类中心点位置，协方差矩阵，各类占比
    """
    # 初始化
    time = 0
    type_num = means.shape[0]
    reply_mat = np.zeros([data.shape[0], type_num])
    alpha = np.zeros(type_num)
    for i in range(type_num):
        alpha[i] = map[i].shape[0] / data.shape[0]
    cov_mat = np.zeros([type_num, data.shape[1], data.shape[1]])
    for i in range(type_num):
        for j in range(map[i].shape[0]):
            map[i][j] -= means[i]
        cov_mat[i] = (map[i].T @ map[i]) / (map[i].shape[0] - 1)
    loss = cal_loss(data, means, cov_mat, alpha)
    while True:
        # E-step
        for i in range(data.shape[0]):
            d = 0
            for j in range(type_num):
                d += alpha[j] * cal_p(data[i], means[j], cov_mat[j])
            for j in range(type_num):
                m = alpha[j] * cal_p(data[i], means[j], cov_mat[j])
                reply_mat[i][j] = m / d
        # M-step
        for i in range(type_num):
            d = 0
            n_means = np.zeros([1, data.shape[1]])
            n_cov = np.zeros([data.shape[1], data.shape[1]])
            for j in range(data.shape[0]):
                d += reply_mat[j][i]
                n_means = n_means + reply_mat[j][i] * data[j, :]
            means[i, :] = n_means / d
            for j in range(data.shape[0]):
                a = ((data[j] - means[i]).reshape([data.shape[1], 1]) @ (data[j] - means[i]).reshape(
                    [1, data.shape[1]]))
                n_cov = n_cov + reply_mat[j][i] * a
            cov_mat[i, :] = n_cov / d
            alpha[i] = d / data.shape[0]
        time += 1
        old_loss = loss
        loss = cal_loss(data, means, cov_mat, alpha)
        if time > max_times or np.abs(loss - old_loss) < delta:
            break
    type_map = []
    first_flag = []
    # 计算类别映射
    for i in range(type_num):
        type_map.append(0)
        first_flag.append(True)
    for i in range(data.shape[0]):
        max_arg = 0
        max_index = -1
        for j in range(type_num):
            if reply_mat[i][j] > max_arg:
                max_arg = reply_mat[i][j]
                max_index = j
        if first_flag[max_index]:
            type_map[max_index] = data[i].reshape([1, data.shape[1]])
            first_flag[max_index] = False
        else:
            type_map[max_index] = np.r_[type_map[max_index], data[i].reshape([1, data.shape[1]])]
    return type_map, means, cov_mat, alpha


def cal_type_num(data_label):
    """
    根据数据标签计算类别数
    :param data_label: 数据标签
    :return: 类别数
    """
    type_num = 0
    check_label = []
    for i in range(data_label.shape[0]):
        if not data_label[i] in check_label:
            type_num += 1
            check_label.append(data_label[i])
    return type_num


def cal_p(point, mean, cov_mat):
    """
    计算高斯概率
    :param point: 样本点
    :param mean: 样本均值
    :param cov_mat: 样本协方差矩阵
    :return: 高斯概率
    """
    # 分母
    d = (2 * np.pi) ** (mean.shape[0] / 2) * np.linalg.det(cov_mat) ** 0.5
    # 分子
    m = np.exp(-0.5 * ((point - mean).T @ np.linalg.pinv(cov_mat) @ (point - mean)))
    return m / d


# def cal_right_rate(data, data_label, type_map: list):
#     """
#     计算分类准确率
#     :param data: 数据点
#     :param data_label: 数据类别
#     :param type_map: 类别与数据点映射
#     :return: 预测准确率
#     """
#     right_num = 0
#     data_flag = np.zeros_like(data_label)
#     for type in range(len(type_map)):
#         for i in range(data.shape[0]):
#             for j in range(type_map[type].shape[0]):
#                 if data_flag[i] == 0 and type == data_label[i] and (data[i, :] == type_map[type][j, :]).all():
#                     right_num = right_num + 1
#                     data_flag[i] = 1
#     return right_num / data.shape[0]


def cal_loss(data, means, cov_mat, rate):
    """
    计算GMM-EM方法损失值
    :param data: 数据点
    :param means: 各类均值
    :param cov_mat: 各类协方差矩阵
    :param rate: 各类占比
    :return: 损失值
    """
    loss = 0
    type_num = means.shape[0]
    for i in range(data.shape[0]):
        temp = 0
        for j in range(type_num):
            temp += rate[j] * cal_p(data[i], means[j], cov_mat[j])
        loss -= np.log(temp)
    return loss


if __name__ == "__main__":
    # 生成数据
    if not UCI_flag:
        Data, Data_label = generate_data()
        plt.subplot(2, 2, 1)
        plt.title("Original Data")
        Start = 0
        End = Start + int(DATA_NUM * RATE[0])
        plt.plot(Data[Start:End, 0], Data[Start:End, 1], 'b.', label='data type 1')
        Start = End
        End = Start + int(DATA_NUM * RATE[1])
        plt.plot(Data[Start:End, 0], Data[Start:End, 1], 'r.', label='data type 2')
        Start = End
        End = Start + int(DATA_NUM * RATE[2])
        plt.plot(Data[Start:End, 0], Data[Start:End, 1], 'y.', label='data type 3')
        Start = End
        End = Start + int(DATA_NUM * RATE[3])
        plt.plot(Data[Start:End, 0], Data[Start:End, 1], 'g.', label='data type 4')
        plt.legend(loc='upper right')
        # plt.show()
    # UCI 数据
    if UCI_flag:
        Data, Data_label = load_UCI_data()
        TYPE_NUM = cal_type_num(Data_label)
        Map = []
        for i in range(TYPE_NUM):
            temp = Data[np.where(Data_label == i), :]
            Map.append(temp.reshape((temp.shape[1], temp.shape[2])))
    if Data.shape[1] == 2:
        if UCI_flag:
            plt.title("Original Data")
            plt.subplot(2, 2, 1)
            plt.plot(Map[0][:, 0], Map[0][:, 1], 'b.', label='data type 1')
            plt.plot(Map[1][:, 0], Map[1][:, 1], 'r.', label='data type 2')
            plt.plot(Map[2][:, 0], Map[2][:, 1], 'y.', label='data type 3')
            plt.legend(loc='upper right')
        plt.subplot(2, 2, 2)
        Map, Centers = level_clustering(Data)
        plt.title("level-clustering")
        plt.plot(Map[0][:, 0], Map[0][:, 1], 'b.', label='data type 1')
        plt.plot(Map[1][:, 0], Map[1][:, 1], 'r.', label='data type 2')
        plt.plot(Map[2][:, 0], Map[2][:, 1], 'y.', label='data type 3')
        if TYPE_NUM > 3:
            print("type number = " + str(TYPE_NUM))
            plt.plot(Map[3][:, 0], Map[3][:, 1], 'g.', label='data type 4')
        plt.plot(Centers[:, 0], Centers[:, 1], 'k+', markersize='15', label='center points')
        plt.legend(loc='upper right')
        # plt.show()
        plt.subplot(2, 2, 3)
        Map, Centers = k_means(Data, Centers)
        plt.title("K-means")
        plt.plot(Map[0][:, 0], Map[0][:, 1], 'b.', label='data type 1')
        plt.plot(Map[1][:, 0], Map[1][:, 1], 'r.', label='data type 2')
        plt.plot(Map[2][:, 0], Map[2][:, 1], 'y.', label='data type 3')
        if TYPE_NUM > 3:
            print("type number = " + str(TYPE_NUM))
            plt.plot(Map[3][:, 0], Map[3][:, 1], 'g.', label='data type 4')
        plt.plot(Centers[:, 0], Centers[:, 1], 'k+', markersize='15', label='center points')
        plt.legend(loc='upper right')
        # plt.show()
        Map, Centers, Cov_mat, Rate = GMM_EM(Data, Map, Centers)
        plt.subplot(2, 2, 4)
        plt.title("GMM_EM")
        plt.plot(Map[0][:, 0], Map[0][:, 1], 'b.', label='data type 1')
        plt.plot(Map[1][:, 0], Map[1][:, 1], 'r.', label='data type 2')
        plt.plot(Map[2][:, 0], Map[2][:, 1], 'y.', label='data type 3')
        if TYPE_NUM > 3:
            plt.plot(Map[3][:, 0], Map[3][:, 1], 'g.', label='data type 4')
        plt.plot(Centers[:, 0], Centers[:, 1], 'k+', markersize='15', label='center points')
        plt.legend(loc='upper right')
        plt.show()
    else:
        ax = plt.axes(projection='3d')
        plt.title("Original Data")
        ax.plot3D(Map[0][:, 0], Map[0][:, 1], Map[0][:, 2], 'b.', label='data type 1')
        ax.plot3D(Map[1][:, 0], Map[1][:, 1], Map[1][:, 2], 'r.', label='data type 2')
        ax.plot3D(Map[2][:, 0], Map[2][:, 1], Map[2][:, 2], 'y.', label='data type 3')
        if TYPE_NUM > 3:
            ax.plot3D(Map[3][:, 0], Map[3][:, 1], Map[3][:, 2], 'g.', label='data type 4')
        plt.legend(loc='upper right')
        plt.show()
        ax = plt.axes(projection='3d')
        Map, Centers = level_clustering(Data, TYPE_NUM)
        plt.title("level-clustering")
        ax.plot3D(Map[0][:, 0].reshape(Map[0][:, 0].shape[0]), Map[0][:, 1].reshape(Map[0][:, 1].shape[0]),
                  Map[0][:, 2].reshape(Map[0][:, 2].shape[0]), 'b.', label='data type 1')
        ax.plot3D(Map[1][:, 0].reshape(Map[1][:, 0].shape[0]), Map[1][:, 1].reshape(Map[1][:, 1].shape[0]),
                  Map[1][:, 2].reshape(Map[1][:, 2].shape[0]), 'r.', label='data type 2')
        ax.plot3D(Map[2][:, 0].reshape(Map[2][:, 0].shape[0]), Map[2][:, 1].reshape(Map[2][:, 1].shape[0]),
                  Map[2][:, 2].reshape(Map[2][:, 2].shape[0]), 'y.', label='data type 3')
        if TYPE_NUM > 3:
            print("type number = " + str(TYPE_NUM))
            ax.plot3D(Map[3][:, 0], Map[3][:, 1], Map[3][:, 2], 'g.', label='data type 4')
        ax.plot3D(Centers[:, 0], Centers[:, 1], Centers[:, 2], 'k+', markersize='15', label='center points')
        ax.legend(loc='upper right')
        plt.show()
        ax = plt.axes(projection='3d')
        Map, Centers = k_means(Data, Centers)
        plt.title("K-means")
        ax.plot3D(Map[0][:, 0], Map[0][:, 1], Map[0][:, 2], 'b.', label='data type 1')
        ax.plot3D(Map[1][:, 0], Map[1][:, 1], Map[1][:, 2], 'r.', label='data type 2')
        ax.plot3D(Map[2][:, 0], Map[2][:, 1], Map[2][:, 2], 'y.', label='data type 3')
        if TYPE_NUM > 3:
            print("type number = " + str(TYPE_NUM))
            ax.plot3D(Map[3][:, 0], Map[3][:, 1], Map[3][:, 2], 'g.', label='data type 4')
        ax.plot3D(Centers[:, 0], Centers[:, 1], Centers[:, 2], 'k+', markersize='15', label='center points')
        ax.legend(loc='upper right')
        plt.show()
        ax = plt.axes(projection='3d')
        Map, Centers, Cov_mat, Rate = GMM_EM(Data, Map, Centers)
        plt.title("GMM_EM")
        ax.plot3D(Map[0][:, 0], Map[0][:, 1], Map[0][:, 2], 'b.', label='data type 1')
        ax.plot3D(Map[1][:, 0], Map[1][:, 1], Map[1][:, 2], 'r.', label='data type 2')
        ax.plot3D(Map[2][:, 0], Map[2][:, 1], Map[2][:, 2], 'y.', label='data type 3')
        if TYPE_NUM > 3:
            ax.plot3D(Map[3][:, 0], Map[3][:, 1], Map[3][:, 2], 'g.', label='data type 4')
        ax.plot3D(Centers[:, 0], Centers[:, 1], Centers[:, 2], 'k+', markersize='15', label='center points')
        ax.legend(loc='upper right')
        plt.show()
    if not UCI_flag:
        Mean = np.asarray([MEAN1, MEAN2, MEAN3, MEAN4])
        print("原均值：" + str(Mean))
        print("求得均值：" + str(Centers))
        print("原协方差矩阵：" + str(np.asarray(COV_MAT)))
        print("求解协方差矩阵：" + str(Cov_mat))
        print("原占比：" + str(np.asarray(RATE)))
        print("求解占比：" + str(Rate))
