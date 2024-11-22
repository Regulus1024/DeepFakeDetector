from os.path import join, exists
import numpy as np
import torch
import torch.utils.data as Data
from .data import get_data, get_data_for_test

'''初始化（__init__ 方法）：

接受参数如 name、level 和 add_root（默认为 './datasets'）。
设置实例变量以存储关于数据集的信息，例如真实和伪造地址（add_real 和 add_fake）、数据集名称（name）、数据集级别（level）以及初始化标志（if_inited）。
调用 init_add 方法来初始化数据集。
数据集初始化（init_add 方法）：

根据数据集名称（name）设置真实和伪造数据目录的地址。
检查数据集目录是否存在，如果不存在则打印错误消息。
如果一切都设置成功，将 if_inited 标志设置为 True。
数据加载方法：

有几种方法（load_data_training_all_interface、load_data_training_interface、load_data_train_all 等）以不同的配置加载训练和测试数据。这些方法似乎处理数据加载的不同方面，包括拆分数据集、加载真实和伪造样本，并创建 PyTorch 数据加载器。
测试数据加载（load_data_test_all、load_data_test_g1、load_data_test_g2 方法）：

这些方法加载测试数据，似乎设计用于在测试期间处理模型的不同分支或组。它们返回真实和伪造样本的数据加载器，以及额外的信息，如标签、视频级标签和其他元数据。
使用外部 get_data 和 get_data_for_test 方法：

代码依赖于外部方法（get_data 和 get_data_for_test）来实际从指定的目录加载数据。
使用 PyTorch DataLoader：

使用 PyTorch 的 DataLoader 类创建训练和测试数据的数据加载器。'''
class Dataset:
    def __init__(self, name, level, add_root='./datasets'):
        self.add_root = add_root
        self.name = name
        self.level = level
        self.add_real = []
        self.add_fake = []
        self.if_inited = False
        self.init_add()
        assert self.if_inited

    def init_add(self):
        """
        初始化数据集。

        参数：
        - name: 数据集名称
        - level: 数据集级别
        - add_root: 数据集根目录，默认为'./datasets'
        """
        if self.name in ['DF', 'NT', 'F2F', 'FS']:
            self.add_real.append(join(self.add_root, 'Origin', self.level))
            self.add_fake.append(join(self.add_root, self.name, self.level))
        elif self.name == 'FF_all':
            for name_sub_dataset in ['DF', 'NT', 'F2F', 'FS']:
                self.add_real.append(join(self.add_root, 'Origin', self.level))
                self.add_fake.append(join(self.add_root, name_sub_dataset, self.level))
        else:
            print("Unsupported dataset name:", self.name, ". Please check and restart.")
            return

        # Ensure the dataset directory exists.
        for add in self.add_real:
            if not exists(add):
                print("The dataset directory:", add, "does not exist. Please check and restart.")
                return
        for add in self.add_fake:
            if not exists(add):
                print("The dataset directory:", add, "does not exist. Please check and restart.")
                return

        self.if_inited = True

    def load_data_training_all_interface(self, block_size, batch_size, split_name):
        """
        The general interface for load the data for training ('train' and 'val').
        The differences between this function and the testing codes is:
            this function ONLY handle the [sample-level] message, but not [video-level].

        :param block_size: The number of the frames in a video sample. [Type: Int]
        :param batch_size: The batch size [Type: Int]
        :param split_name: 'train' or 'val' [Type: Str]
        :return: Return 2 dataloader (iter_A and iter_B)
        """
        """
        加载用于训练的数据集的通用接口。

        参数：
        - block_size: 视频样本中的帧数
        - batch_size: 批处理大小
        - split_name: 'train' 或 'val'

        返回值：
        两个数据加载器（iter_A 和 iter_B）
        """
        assert split_name in ['train', 'val']
        samples = None
        samples_diff = None
        labels = None

        for add_r in self.add_real:
            real_samples, real_samples_diff, real_labels = get_data(join(add_r, split_name), 0, block_size)
            if samples is None:
                samples = real_samples
                samples_diff = real_samples_diff
                labels = real_labels

                if split_name == 'val':
                    # For the 'FF_all' setting, the add_r = [Origin] x 4
                    # The same as test dataset. DO NOT augment the real sample in val dataset.
                    break
            else:
                samples = np.concatenate((samples, real_samples), axis=0)
                samples_diff = np.concatenate((samples_diff, real_samples_diff), axis=0)
                labels = np.concatenate((labels, real_labels), axis=0)

        # Flush the memory
        real_samples = None
        real_samples_diff = None
        real_labels = None

        for add_f in self.add_fake:
            fake_samples, fake_samples_diff, fake_labels = get_data(join(add_f, split_name), 1, block_size)
            samples = np.concatenate((samples, fake_samples), axis=0)
            samples_diff = np.concatenate((samples_diff, fake_samples_diff), axis=0)
            labels = np.concatenate((labels, fake_labels), axis=0)

        # Flush the memory
        fake_samples = None
        fake_samples_diff = None
        fake_labels = None

        # Convert to PyTorch dataset
        samples = torch.tensor(samples, dtype=torch.float32)
        samples_diff = torch.tensor(samples_diff, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset_A = Data.TensorDataset(samples, labels)
        dataset_B = Data.TensorDataset(samples_diff, labels)
        dataset_iter_A = Data.DataLoader(dataset_A, batch_size, shuffle=True)
        dataset_iter_B = Data.DataLoader(dataset_B, batch_size, shuffle=True)

        return dataset_iter_A, dataset_iter_B

    def load_data_training_interface(self, block_size, batch_size, split_name, branch_selection):
        """
        The general interface for load the data for training ('train' and 'val').
        The differences between this function and the testing codes is:
            this function ONLY handle the [sample-level] message, but not [video-level].

        :param block_size: The number of the frames in a video sample. [Type: Int]
        :param batch_size: The batch size [Type: Int]
        :param split_name: 'train' or 'val' [Type: Str]
        :param branch_selection: 'g1', 'g2' [Type: Str]
        :return: Only return 1 dataloader (iter_A or iter_B)
        """
        """
        加载用于训练的数据集的通用接口。

        参数：
        - block_size: 视频样本中的帧数
        - batch_size: 批处理大小
        - split_name: 'train' 或 'val'
        - branch_selection: 'g1' 或 'g2'

        返回值：
        一个数据加载器（iter_A 或 iter_B）
        """
        assert split_name in ['train', 'val']
        assert branch_selection in ['g1', 'g2']
        samples = None
        labels = None

        for add_r in self.add_real:
            if branch_selection == 'g1':
                real_samples, _, real_labels = get_data(join(add_r, split_name), 0, block_size)
            else:
                # Only preserve the _diff part. But for the concision we only mark it as 'samples' here.
                _, real_samples, real_labels = get_data(join(add_r, split_name), 0, block_size)

            if samples is None:
                samples = real_samples
                labels = real_labels

                if split_name == 'val':
                    # For the 'FF_all' setting, the add_r = [Origin] x 4
                    # The same as test dataset. DO NOT augment the real sample in val dataset.
                    break
            else:
                samples = np.concatenate((samples, real_samples), axis=0)
                labels = np.concatenate((labels, real_labels), axis=0)

        # Flush the memory
        real_samples = None
        real_labels = None

        for add_f in self.add_fake:
            if branch_selection == 'g1':
                fake_samples, _, fake_labels = get_data(join(add_f, split_name), 1, block_size)
            else:
                _, fake_samples, fake_labels = get_data(join(add_f, split_name), 1, block_size)
            samples = np.concatenate((samples, fake_samples), axis=0)
            labels = np.concatenate((labels, fake_labels), axis=0)

        # Flush the memory
        fake_samples = None
        fake_labels = None

        # Convert to PyTorch dataset
        samples = torch.tensor(samples, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset = Data.TensorDataset(samples, labels)
        dataset_iter = Data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)

        return dataset_iter

    def load_data_train_all(self, block_size, batch_size):
        return self.load_data_training_all_interface(block_size, batch_size, 'train')

    def load_data_val_all(self, block_size, batch_size):
        return self.load_data_training_all_interface(block_size, batch_size, 'val')

    def load_data_train_g1(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'train', 'g1')

    def load_data_val_g1(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'val', 'g1')

    def load_data_train_g2(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'train', 'g2')

    def load_data_val_g2(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'val', 'g2')

    def load_data_test_all(self, block_size, batch_size):
        """
        加载所有测试数据。

        Args:
            block_size (int): 视频样本中的帧数.
            batch_size (int): 批处理大小.

        Returns:
            tuple: 包含两个数据加载器 (test_iter_A 和 test_iter_B), 以及其他测试信息.
        """
        test_samples = None
        test_samples_diff = None
        test_labels = None
        test_labels_video = None
        test_sv = None
        test_vc = None

        for add_r in self.add_real:
            real_samples, real_samples_diff, real_labels, real_labels_video, real_sv, real_vc = \
                get_data_for_test(join(add_r, "test/"), 0, block_size)

            # Only load the real samples once.
            test_samples = real_samples
            test_samples_diff = real_samples_diff
            test_labels = real_labels
            test_labels_video = real_labels_video
            test_sv = real_sv
            test_vc = real_vc
            break

        # Flush the memory
        real_samples = None
        real_samples_diff = None
        real_labels = None

        for add_f in self.add_fake:
            fake_samples, fake_samples_diff, fake_labels, fake_labels_video, fake_sv, fake_vc = \
                get_data_for_test(join(add_f, "test/"), 1, block_size)
            test_samples = np.concatenate((test_samples, fake_samples), axis=0)
            test_samples_diff = np.concatenate((test_samples_diff, fake_samples_diff), axis=0)
            test_labels = np.concatenate((test_labels, fake_labels), axis=0)
            test_labels_video = np.concatenate((test_labels_video, fake_labels_video), axis=0)
            test_sv = np.concatenate((test_sv, fake_sv), axis=0)
            test_vc.update(fake_vc)

        # Flush the memory
        fake_samples = None
        fake_samples_diff = None
        fake_labels = None

        # Convert to PyTorch dataset
        test_samples = torch.tensor(test_samples, dtype=torch.float32)
        test_samples_diff = torch.tensor(test_samples_diff, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        test_dataset_A = Data.TensorDataset(test_samples, test_labels)
        test_dataset_B = Data.TensorDataset(test_samples_diff, test_labels)

        test_iter_A = Data.DataLoader(test_dataset_A, batch_size, shuffle=False)
        test_iter_B = Data.DataLoader(test_dataset_B, batch_size, shuffle=False)

        return test_iter_A, test_iter_B, test_labels, test_labels_video, test_sv, test_vc

    def load_data_test_g1(self, block_size, batch_size):
        test_samples = None
        test_labels = None
        test_labels_video = None
        test_sv = None
        test_vc = None

        for add_r in self.add_real:
            real_samples, _, real_labels, real_labels_video, real_sv, real_vc = \
                get_data_for_test(join(add_r, "test/"), 0, block_size)

            # Only load the real samples once.
            test_samples = real_samples
            test_labels = real_labels
            test_labels_video = real_labels_video
            test_sv = real_sv
            test_vc = real_vc
            break

        # Flush the memory
        real_samples = None
        real_labels = None

        for add_f in self.add_fake:
            fake_samples, _, fake_labels, fake_labels_video, fake_sv, fake_vc = \
                get_data_for_test(join(add_f, "test/"), 1, block_size)
            test_samples = np.concatenate((test_samples, fake_samples), axis=0)
            test_labels = np.concatenate((test_labels, fake_labels), axis=0)
            test_labels_video = np.concatenate((test_labels_video, fake_labels_video), axis=0)
            test_sv = np.concatenate((test_sv, fake_sv), axis=0)
            test_vc.update(fake_vc)

        # Flush the memory
        fake_samples = None
        fake_labels = None

        # Convert to PyTorch dataset
        test_samples = torch.tensor(test_samples, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        test_dataset_A = Data.TensorDataset(test_samples, test_labels)
        test_iter_A = Data.DataLoader(test_dataset_A, batch_size, shuffle=False)

        return test_iter_A, test_labels, test_labels_video, test_sv, test_vc

    def load_data_test_g2(self, block_size, batch_size):
        test_samples_diff = None
        test_labels = None
        test_labels_video = None
        test_sv = None
        test_vc = None

        for add_r in self.add_real:
            _, real_samples_diff, real_labels, real_labels_video, real_sv, real_vc = \
                get_data_for_test(join(add_r, "test/"), 0, block_size)

            # Only load the real samples once.
            test_samples_diff = real_samples_diff
            test_labels = real_labels
            test_labels_video = real_labels_video
            test_sv = real_sv
            test_vc = real_vc
            break

        # Flush the memory
        real_samples_diff = None
        real_labels = None

        for add_f in self.add_fake:
            _, fake_samples_diff, fake_labels, fake_labels_video, fake_sv, fake_vc = \
                get_data_for_test(join(add_f, "test/"), 1, block_size)
            test_samples_diff = np.concatenate((test_samples_diff, fake_samples_diff), axis=0)
            test_labels = np.concatenate((test_labels, fake_labels), axis=0)
            test_labels_video = np.concatenate((test_labels_video, fake_labels_video), axis=0)
            test_sv = np.concatenate((test_sv, fake_sv), axis=0)
            test_vc.update(fake_vc)

        # Flush the memory
        fake_samples_diff = None
        fake_labels = None

        # Convert to PyTorch dataset
        test_samples_diff = torch.tensor(test_samples_diff, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        test_dataset_B = Data.TensorDataset(test_samples_diff, test_labels)
        test_iter_B = Data.DataLoader(test_dataset_B, batch_size, shuffle=False)

        return test_iter_B, test_labels, test_labels_video, test_sv, test_vc
