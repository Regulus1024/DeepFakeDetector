import argparse  # 导入命令行参数解析库
import os  # 导入操作系统相关库
from tqdm import tqdm, trange  # 导入用于显示进度条的库
from torch import optim  # 导入PyTorch的优化模块
from os.path import join  # 导入用于拼接文件路径的函数
from utils.model import *  # 从自定义模块中导入模型
from utils.logger import Logger  # 导入用于日志记录的自定义Logger类
from utils.metric import *  # 从自定义模块中导入度量函数
from utils.dataset import Dataset  # 导入用于处理数据集的自定义Dataset类
from configs.loader import load_yaml  # 导入加载YAML配置文件的函数


def train_loop(model, train_iter, val_iter, optimizer, loss, epochs, device, add_weights_file):
    # 训练循环函数
    log_training_loss = []  # 用于存储训练损失的列表
    log_training_accuracy = []  # 用于存储训练精度的列表
    log_val_accuracy = []  # 用于存储验证精度的列表
    best_val_acc = 0.0  # 用于存储最佳验证精度的变量

    model.to(device)  # 将模型移动到指定的设备（GPU或CPU）
    model.train()  # 将模型设置为训练模式
    for epoch in trange(1, epochs + 1):  # 对每个训练周期进行迭代
        loss_sum, acc_sum, samples_sum = 0.0, 0.0, 0  # 初始化损失、精度和样本数的累加变量
        for X, y in train_iter:  # 对每个训练批次进行迭代
            # 将数据加载到GPU/CPU
            X = X.to(device)
            y = y.to(device)
            samples_num = X.shape[0]  # 计算批次中样本的数量

            # 正向传播
            output = model(X)
            log_softmax_output = torch.log(output)
            l = loss(log_softmax_output, y)  # 计算损失
            optimizer.zero_grad()  # 梯度归零
            l.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            # 生成日志
            loss_sum += l.item() * samples_num
            acc_sum += calculate_accuracy(output, y) * samples_num
            samples_sum += samples_num
        train_acc = acc_sum / samples_sum  # 计算训练精度

        val_acc = evaluate(model, val_iter, device)  # 在验证集上评估模型精度

        if val_acc >= best_val_acc:
            save_hint = "save the model to {}".format(add_weights_file)
            torch.save(model.state_dict(), add_weights_file)  # 保存模型参数到文件
            best_val_acc = val_acc
        else:
            save_hint = ""

        tqdm.write("epoch:{}, loss:{:.4}, train_acc:{:.4}, test_acc:{:.4}, best_record:{:.4}  "
                   .format(epoch, loss_sum / samples_sum, train_acc, val_acc, best_val_acc)
                   + save_hint)  # 打印训练过程的日志信息

        log_training_loss.append(loss_sum / samples_sum)
        log_training_accuracy.append(train_acc)
        log_val_accuracy.append(val_acc)

    log = {"loss": log_training_loss, "acc_train": log_training_accuracy, "acc_val": log_val_accuracy}
    return log


def main(args):
    if_gpu = args.gpu  # 获取是否使用GPU的标志
    dataset_name = args.dataset  # 获取数据集名称
    dataset_level = args.level  # 获取数据集压缩级别
    branch_selection = args.branch  # 获取LRNet的分支选择

    # 初始化配置参数
    args_model = load_yaml("configs/args_model.yaml")
    args_train = load_yaml("configs/args_train.yaml")

    BLOCK_SIZE = args_train["BLOCK_SIZE"]
    BATCH_SIZE = args_train["BATCH_SIZE"]

    add_weights = args_train["add_weights"]
    if not os.path.exists(add_weights):
        os.makedirs(add_weights)

    EPOCHS_g1 = args_train["EPOCHS_g1"]
    LEARNING_RATE_g1 = args_train["LEARNING_RATE_g1"]
    weights_name_g1 = args_train["weights_name_g1"]

    EPOCHS_g2 = args_train["EPOCHS_g2"]
    LEARNING_RATE_g2 = args_train["LEARNING_RATE_g2"]
    weights_name_g2 = args_train["weights_name_g2"]

    # 检查是否使用GPU
    if if_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    # 初始化数据集
    dataset = Dataset(add_root=args_train["add_dataset_root"],
                      name=dataset_name, level=dataset_level)

    # 初始化日志记录器
    logger = Logger()
    logger.register_status(dataset=dataset,
                           device=device,
                           branch_selection=branch_selection)
    logger.register_args(**args_train, **args_model)
    logger.print_logs_training()

    # 加载训练和验证数据
    train_iter_A = None
    train_iter_B = None
    val_iter_A = None
    val_iter_B = None
    if branch_selection == 'g1':
        train_iter_A = dataset.load_data_train_g1(BLOCK_SIZE, BATCH_SIZE)
        val_iter_A = dataset.load_data_val_g1(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'g2':
        train_iter_B = dataset.load_data_train_g2(BLOCK_SIZE, BATCH_SIZE)
        val_iter_B = dataset.load_data_val_g2(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'all':
        train_iter_A, train_iter_B = dataset.load_data_train_all(BLOCK_SIZE, BATCH_SIZE)
        val_iter_A, val_iter_B = dataset.load_data_val_all(BLOCK_SIZE, BATCH_SIZE)
    else:
        print("Unknown branch selection:", branch_selection, '. Please check and restart')
        return

    # 训练
    if branch_selection == 'g1' or branch_selection == 'all':
        assert train_iter_A, val_iter_A is not None
        g1 = LRNet(**args_model)
        optimizer = optim.Adam(g1.parameters(), lr=LEARNING_RATE_g1)
        loss = nn.NLLLoss()
        add_weights_file = join(add_weights, weights_name_g1)
        log_g1 = train_loop(g1, train_iter_A, val_iter_A, optimizer, loss, EPOCHS_g1, device, add_weights_file)

    if branch_selection == 'g2' or branch_selection == 'all':
        assert train_iter_B, val_iter_B is not None
        g2 = LRNet(**args_model)
        optimizer = optim.Adam(g2.parameters(), lr=LEARNING_RATE_g2)
        loss = nn.NLLLoss()
        add_weights_file = join(add_weights, weights_name_g2)
        log_g2 = train_loop(g2, train_iter_B, val_iter_B, optimizer, loss, EPOCHS_g2, device, add_weights_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training codes of LRNet (PyTorch version).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gpu', action='store_true',
                        help="If use the GPU(CUDA) for training."
                        )
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['DF', 'F2F', 'FS', 'NT', 'FF_all'],
                        default='DF',
                        help="Select the dataset used for training. "
                             "Valid selections: ['DF', 'F2F', 'FS', 'NT', 'FF_all'] "
                        )
    parser.add_argument('-l', '--level', type=str,
                        choices=['raw', 'c23', 'c40'],
                        default='raw',
                        help="Select the dataset compression level. "
                             "Valid selections: ['raw', 'c23', 'c40'] ")
    parser.add_argument('-b', '--branch', type=str,
                        choices=['g1', 'g2', 'all'],
                        default='all',
                        help="Select which branch of the LRNet to be trained. "
                             "Valid selections: ['g1', 'g2', 'all']")
    args = parser.parse_args()
    main(args)

