import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

DATA_NUM = 1000
DIM = 3
MEAN_3D = (1, 2, 3)
MEAN_2D = (1, 2)
COV_2D = ((1, 0), (0, 0.01))
COV_3D = ((0.01, 0, 0), (0, 1, 0), (0, 0, 1))
THETA = np.pi / 4  # 旋转角度，弧度值
AXIS = 'z'  # 三维图像时绕哪一轴旋转,以逆时针为正
FILE_PATH_LIST = ('./lab4/img1.png', './lab4/img2.png', './lab4/img3.png', './lab4/img4.png')
SIZE = (80, 80)
TARGET_DIM = 4
LOAD_DATA = True


def generate_data(data_num=DATA_NUM, mean=MEAN_2D if DIM == 2 else MEAN_3D,
                  cov=COV_2D if DIM == 2 else COV_3D):
    """
    生成数据
    :param data_num: 数据量
    :param mean: 数据均值
    :param cov: 数据协方差矩阵
    :return: 数据矩阵（每行是一个样本）
    """
    data = np.random.multivariate_normal(mean, cov, size=data_num)
    data = rotate(data)
    return data


def img_pre_process(img, size=SIZE):
    """
    图像预处理
    :param img: 原图像
    :param size: 标准尺寸
    :return: 标准化的图像
    """
    img = cv.resize(img, size)
    # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    return img


def load_img(file_path_list):
    """
    载入图像
    :param file_path_list: 图像的路径列表
    :return: 图像的数据矩阵（每行是一个样本）
    """
    # data_r = []
    # data_g = []
    # data_b = []
    data_grey = []
    img_list = []
    h = w = 0
    i = 1
    for file in file_path_list:
        plt.subplot(2, 2, i)
        img = cv.imread(file)
        img = img_pre_process(img)
        plt.imshow(img, cmap='gray')
        img_list.append(img)
        # h, w, c = img.shape
        h, w = img.shape
        # img_data = img.reshape((h * w, c))
        img_data = img.reshape(h * w)
        img_data = img_data.astype('int32')
        # data_b.append(img_data[:, 0])
        # data_g.append(img_data[:, 1])
        # data_r.append(img_data[:, 2])
        data_grey.append(img_data)
        i += 1
    print("Original Images")
    plt.show()
    # return np.asarray(data_b), np.asarray(data_g), np.asarray(data_r), h, w
    return img_list, np.asarray(data_grey), h, w


def cal_PSNR(src_img, dst_img):
    """
    计算PSNR信噪比
    :param src_img: 原图像
    :param dst_img: 重构后的图像
    :return: PSNR信噪比
    """
    mse = np.mean((src_img / 255. - dst_img / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # 使用的信噪比公式为20 log_10^(MAX/sqrt(MSE))
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def rotate(data, dim=DIM, theta=THETA, axis=AXIS):
    """
    将数据点旋转
    :param data: 数据
    :param dim: 数据的维数
    :param theta: 旋转角度
    :param axis: 旋转轴
    :return: 旋转后的数据
    """
    if dim == 2:
        data = (np.asarray(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))) @ data.T).T
    else:
        if axis == 'z':
            data = (np.asarray(
                ((np.cos(theta), -np.sin(theta), 0), (np.sin(theta), np.cos(theta), 0), (0, 0, 1))) @ data.T).T
        elif axis == 'x':
            data = (np.asarray(
                ((1, 0, 0), (0, np.cos(theta), -np.sin(theta)), (0, np.sin(theta), np.cos(theta)))) @ data.T).T
        elif axis == 'y':
            data = (np.asarray(
                ((np.cos(theta), 0, np.sin(theta)), (0, 1, 0), (-np.sin(theta), 0, np.cos(theta)))) @ data.T).T
    return data


def PCA(data, k):
    """
    PCA降维并重构
    :param data: 数据
    :param k: 降维维数
    :return: 重构后的数据
    """
    dim = data.shape[1]
    mean = np.mean(data, axis=0)
    temp = np.zeros_like(data)
    # print(mean)
    for i in range(dim):
        temp[:, i] = data[:, i] - mean[i]
    cov = temp.T @ temp
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    eigen_values = np.real(eigen_values)
    eigen_value_Index = np.argsort(eigen_values)
    wanted_eigen_vector = eigen_vectors[:, eigen_value_Index[:-(k + 1):-1]]
    wanted_eigen_vector = np.real(wanted_eigen_vector)
    # 计算各点降维后的与均值的偏移量
    new_temp = temp @ wanted_eigen_vector  # n * k维
    # new_data = np.zeros_like(data)
    # img = np.reshape(wanted_eigen_vector[:, 0], SIZE)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # print(new_data.shape, new_temp.shape, wanted_eigen_vector.shape, mean.shape)
    # for i in range(dim):
    #     if i == 0:
    #         print(new_data[:, 0], mean[0])
    #     new_data[:, i] = new_temp @ wanted_eigen_vector[i] +mean[i]
    #     if i == 0:
    #         print(new_data[:,0], new_temp @ wanted_eigen_vector[0], new_temp @ wanted_eigen_vector[0]+mean[0])
    # print(new_data, new_temp @ wanted_eigen_vector.T + mean)
    new_data = new_temp @ wanted_eigen_vector.T + mean
    # for i in range(dim):
    #     new_data[:, i] = new_data[:, i] + mean[i]
    # for i in range(data.shape[0]):
    #     new_data[i] = new_data[i] + mean
    return new_data


if __name__ == '__main__':
    if not LOAD_DATA:
        Data = generate_data()
        if DIM == 3:
            New_data = PCA(Data, 2)
            ax = plt.axes(projection='3d')
            ax.plot3D(Data[:, 0], Data[:, 1], Data[:, 2], 'k.', label='Generated Data')
            ax.plot3D(New_data[:, 0], New_data[:, 1], New_data[:, 2], 'r', label='PCA Data')
            plt.legend(loc='upper right')
            plt.show()
        else:
            New_data = PCA(Data, 1)
            plt.plot(Data[:, 0], Data[:, 1], 'k.', label='Generated Data')
            plt.plot(New_data[:, 0], New_data[:, 1], 'r', label='PCA Data')
            plt.legend(loc='upper right')
            plt.show()
    else:
        # Data_b, Data_g, Data_r, H, W = load_img(file_path_list=FILE_PATH_LIST)
        # New_data_b = PCA(Data_b, TARGET_DIM)
        # New_data_g = PCA(Data_g, TARGET_DIM)
        # New_data_r = PCA(Data_r, TARGET_DIM)
        # for I in range(Data_b.shape[0]):
        #     Img = np.c_[New_data_b[I], New_data_g[I], New_data_r[I]]
        #     # Img = np.c_[Data_b[I], Data_g[I], Data_r[I]]
        #     Img = Img.reshape((H, W, 3))
        #     plt.imshow(Img)
        #     plt.show()
        Img_list, Data_grey, H, W = load_img(file_path_list=FILE_PATH_LIST)
        New_data_grey = PCA(Data_grey, TARGET_DIM)
        PSNR_list = []
        for I in range(Data_grey.shape[0]):
            plt.subplot(2, 4, 2 * (I + 1) - 1)
            plt.title("Original Image")
            plt.imshow(Img_list[I], cmap='gray')
            plt.subplot(2, 4, 2 * (I + 1))
            plt.title("PCA Image")
            Img = New_data_grey[I]
            Img = Img.reshape(H, W)
            PSNR_list.append(cal_PSNR(Img_list[I], Img))
            plt.imshow(Img, cmap='gray')
        print("PCA IMAGES\nPSNR1:" + str(PSNR_list[0])
              + "\nPSNR2:" + str(PSNR_list[1])
              + "\nPSNR3:" + str(PSNR_list[2])
              + "\nPSNR4:" + str(PSNR_list[3]))
        plt.show()
