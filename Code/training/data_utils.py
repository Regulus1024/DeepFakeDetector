import os
import numpy as np
from tqdm import tqdm
from os.path import join


def get_data(path, fake, block):
    """
    Read the data into memory (for training).
    :param path: The path of the folder containing landmarks files.
    :param fake: Assign the label of the data. Original(real) = 0, and manipulated(fake) = 1.
    :param block: block: The length of a 'block', i.e., the frames number of a video sample.
    :return:
    """
    """
    读取数据到内存（用于训练）。
    :param path: 包含地标文件的文件夹路径。
    :param fake: 分配数据的标签。原始（真实）= 0，篡改（假）= 1。
    :param block: 'block'的长度，即视频样本的帧数。
    :return:
    """
    files = os.listdir(path)
    x = []
    x_diff = []
    y = []

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
    Read the data into memory (for evaluating).
    :param path: The path of the folder containing landmarks files.
    :param fake: Assign the label of the data. Original(real) = 0, and manipulated(fake) = 1.
    :param block: The length of a 'block', i.e., the frames number of a video sample.
    :return:x: The feature vector A. It contains all the data in the datasets. Shape: [N, 136].
            x_diff; The feature vector B.  Shape: [N-1, 136]
            y: The labels. Shape: [N]
            video_y: The video-level labels (used for video-level evaluation).
            sample_to_video: A list recording the mappings of the samples(fix-length segments) to
                                their corresponding video. Shape: [N]
            count_y: A dictionary for counting the number of segments included in each video.
                                Keys: videos' name. Values: number of the segments.
    """
    """
    读取数据到内存（用于评估）。
    :param path: 包含地标文件的文件夹路径。
    :param fake: 分配数据的标签。原始（真实）= 0，篡改（假）= 1。
    :param block: 'block'的长度，即视频样本的帧数。
    :return:x: 特征向量A。包含数据集中的所有数据。形状：[N, 136]。
            x_diff：特征向量B。形状：[N-1, 136]
            y: 标签。形状：[N]
            video_y: 视频级别的标签（用于视频级别评估）。
            sample_to_video: 记录样本（固定长度段）到其相应视频的映射的列表。形状：[N]
            count_y: 一个字典，用于计算每个视频中包含的段数。键：视频名称。值：段数。
    """
    files = os.listdir(path)
    x = []
    x_diff = []
    y = []

    video_y = []
    count_y = {}
    sample_to_video = []

    print("Loading data and embedding...")
    for file in tqdm(files):
        vectors = np.loadtxt(join(path, file))
        video_y.append(fake)

        for i in range(0, vectors.shape[0] - block, block):
            vec = vectors[i:i + block, :]
            x.append(vec)
            vec_next = vectors[i + 1:i + block, :]
            vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
            vec_diff = (vec_next - vec)[:block - 1, :]
            x_diff.append(vec_diff)

            y.append(fake)

            # Dict for counting number of samples in video
            if file not in count_y:
                count_y[file] = 1
            else:
                count_y[file] += 1

            # Recording each samples belonging
            sample_to_video.append(file)
    return np.array(x), np.array(x_diff), np.array(y), np.array(video_y), np.array(sample_to_video), count_y


def merge_video_prediction(mix_prediction, s2v, vc):
    """
    :param mix_prediction: The mixed prediction of 2 branches. (of each sample)
    :param s2v: Sample-to-video. Refer to the 'sample_to_video' in function get_data_for_test()
    :param vc: Video-Count. Refer to the 'count_y' in function get_data_for_test()
    :return: prediction_video: The prediction of each video.
    """
    """
    :param mix_prediction: 2个分支的混合预测（每个样本）
    :param s2v: 样本到视频的映射。参考get_data_for_test()中的'sample_to_video'
    :param vc: 视频-计数。参考get_data_for_test()中的'count_y'
    :return: prediction_video: 每个视频的预测。
    """
    prediction_video = []
    pre_count = {}
    for p, v_label in zip(mix_prediction, s2v):
        p_bi = 0
        if p >= 0.5:
            p_bi = 1
        if v_label in pre_count:
            pre_count[v_label] += p_bi
        else:
            pre_count[v_label] = p_bi
    for key in pre_count.keys():
        prediction_video.append(pre_count[key] / vc[key])
    return prediction_video

