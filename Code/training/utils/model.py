import torch
from torch import nn

'''
这个文件的主要作用是定义了一个名为LRNet的神经网络模型，该模型结合了Landmark RNN网络和全连接网络。LRNet用于处理具有特定特征大小的输入数据
，其中Landmark RNN网络包含一个双向GRU层，用于处理时间序列数据，并通过LandmarkDropout模块在训练时对部分位置进行丢弃。
全连接网络包含一个残差连接的前馈神经网络和一个输出层，用于最终的分类任务。
整体上，这个模型似乎设计用于处理带有时间信息和空间信息的特定数据，并进行分类预测。
'''
class LandmarkDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(LandmarkDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def generate_mask(self, landmark, frame):
        position_p = torch.bernoulli(torch.Tensor([1 - self.p]*(landmark//2)))
        return position_p.repeat(1, frame, 2)

    def forward(self, x: torch.Tensor):
        if self.training:
            _, frame, landmark = x.size()
            landmark_mask = self.generate_mask(landmark, frame)
            scale = 1/(1-self.p)
            return x*landmark_mask.to(x.device)*scale
        else:
            return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class LRNet(nn.Module):
    def __init__(self, feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                 num_layers=1, rnn_dropout_rate=0,
                 fc_dropout_rate=0.5, res_hidden=64):
        super(LRNet, self).__init__()
        self.hidden_size = rnn_unit
        self.hidden_state = nn.Parameter(torch.randn(2 * num_layers, 1, rnn_unit))
        self.dropout_landmark = LandmarkDropout(lm_dropout_rate)
        self.gru = nn.GRU(input_size=feature_size, hidden_size=rnn_unit,
                          num_layers=num_layers, dropout=rnn_dropout_rate,
                          batch_first=True, bidirectional=True)

        self.dense = nn.Sequential(
            nn.Dropout(fc_dropout_rate),
            Residual(FeedForward(rnn_unit * 2 * num_layers, res_hidden, fc_dropout_rate)),
            nn.Dropout(fc_dropout_rate),

            # MLP-Head
            nn.Linear(rnn_unit * 2 * num_layers, 2)
        )
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout_landmark(x)
        _, hidden = self.gru(x, self.hidden_state.repeat(1, x.shape[0], 1))
        x = torch.cat(list(hidden), dim=1)
        x = self.dense(x)
        x = self.output(x)
        return x
