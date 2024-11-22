import os  # 导入操作系统相关的库
from os.path import join  # 导入用于路径拼接的函数
import numpy as np  # 导入处理数组的库
from tqdm import tqdm  # 导入进度条显示库
'''
用于将训练和测试数据加载到内存中，以供模型训练和评估使用。get_data函数用于读取训练数据，而get_data_for_test函数用于读取测试数据。

这些函数接受包含地标文件的文件夹路径、标签（真实或伪造）以及块的长度作为输入。函数返回用于模型的特征向量和标签，以及用于视频级别评估的附加信息。'''
def get_data(path, fake, block):
    """
    读取数据到内存中（用于训练）。
    :param path: 包含地标文件的文件夹路径。
    :param fake: 分配数据的标签。原始（真实）= 0，篡改（伪造）= 1。
    :param block: “块”的长度，即视频样本的帧数。
    :return: x: 特征向量A，包含数据集中的所有数据。形状：[N, 136]。
             x_diff：特征向量B。形状：[N-1, 136]
             y: 标签。形状：[N]
    """
    files = os.listdir(path)
    x = []
    x_diff = []
    y = []

    print("Loading data from: ", path)
    for file in tqdm(files):
        vectors = np.loadtxt(join(path, file))
        for i in range(0, vectors.shape[0] - block, block):
            vec = vectors[i:i + block, :]
            x.append(vec)
            vec_next = vectors[i + 1:i + block, :]
            vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
            vec_diff = (vec_next - vec)[:block - 1, :]
            x_diff.append(vec_diff)
            y.append(fake)
    return np.array(x), np.array(x_diff), np.array(y)

def get_data_for_test(path, fake, block):
    """
    读取数据到内存中（用于评估）。
    :param path: 包含地标文件的文件夹路径。
    :param fake: 分配数据的标签。原始（真实）= 0，篡改（伪造）= 1。
    :param block: “块”的长度，即视频样本的帧数。
    :return: x: 特征向量A，包含数据集中的所有数据。形状：[N, 136]。
             x_diff：特征向量B。形状：[N-1, 136]
             y: 标签。形状：[N]
             video_y: 视频级别的标签（用于视频级别的评估）。
             sample_to_video: 记录样本（固定长度段）与其对应视频的映射的列表。形状：[N]
             count_y: 用于计算每个视频中包含的段数的字典。键：视频的名称。值：段数。
    """

    files = os.listdir(path)
    x = []
    x_diff = []
    y = []

    video_y = []
    count_y = {}
    sample_to_video = []

    print("Loading data from: ", path)
    for file in tqdm(files):
        vectors = np.loadtxt(join(path, file))

        # 添加一个判断，当向量的长度小于块时，直接丢弃
        if vectors.shape[0] < block:
            continue
        for i in range(0, vectors.shape[0] - block, block):
            vec = vectors[i:i + block, :]
            x.append(vec)
            vec_next = vectors[i + 1:i + block, :]
            vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
            vec_diff = (vec_next - vec)[:block - 1, :]
            x_diff.append(vec_diff)

            y.append(fake)

            # 计数每个视频中样本的数量
            file_dir = join(path, file)
            if file_dir not in count_y:
                count_y[file_dir] = 1
            else:
                count_y[file_dir] += 1

            # 记录每个样本属于哪个视频
            sample_to_video.append(file_dir)

        video_y.append(fake)
    return np.array(x), np.array(x_diff), np.array(y), np.array(video_y), np.array(sample_to_video), count_y
