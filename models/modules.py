from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

# CARE: This has a trailing ReLU!!
def get_mlp(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct and MLP for use in other models.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                for i in range(1, len(channels))
            ]
        )
### mlp = get_mlp([64, 128, 256], add_batchnorm=True) 
# 将生成mlp模型
# nn.Linear(64, 128)
# nn.BatchNorm1d(128)
# nn.ReLU()
# nn.Linear(128, 256)
# nn.BatchNorm1d(256)
# nn.ReLU()

### mlp = get_mlp([64, 128, 256], add_batchnorm=False)
# 将生成mlp模型
# nn.Linear(64, 128)
# nn.ReLU()
# nn.Linear(128, 256)
# nn.ReLU()

class LanguageEncoder(torch.nn.Module):
    def __init__(self, known_words, embedding_dim, bi_dir, num_layers=1):
        """Language encoder to encode a set of hints"""
        super(LanguageEncoder, self).__init__()
        
        # 通过遍历 known_words 列表，将每个单词映射到一个整数，并将这些映射存储在 self.known_words 字典中。其中，键是单词，值是对应的整数编码。
        self.known_words = {c: (i + 1) for i, c in enumerate(known_words)}
        
        # 为了处理未知单词，将 "<unk>" 单词映射到整数 0。
        self.known_words["<unk>"] = 0

        # 使用 nn.Embedding 类创建了一个嵌入层 self.word_embedding，用于将单词映射为密集向量表示。
        # len(self.known_words) 给出了嵌入层的输入大小（即已知单词的数量），embedding_dim 给出了每个单词嵌入的维度。padding_idx=0 表示使用索引为 0 的单词作为填充符。
        self.word_embedding = nn.Embedding(len(self.known_words), embedding_dim, padding_idx=0)

        # 使用 nn.LSTM 类创建了一个 LSTM 层 self.lstm。
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            bidirectional=bi_dir,
            num_layers=num_layers,
        )

    """
    Encodes descriptions as batch [d1, d2, d3, ..., d_B] with d_i a string. Strings can be of different sizes.
    """

    def forward(self, descriptions):
        # 将字符串转换为整数编码的列表
        word_indices = [
            [
                self.known_words.get(word, 0)
                for word in description.replace(".", "").replace(",", "").lower().split()
            ]
            for description in descriptions
        ]
        
        # 计算每段描述的长度
        description_lengths = [len(w) for w in word_indices]
        
        # batch_size设置为描述的数量，max_length为最长一段描述的长度
        batch_size, max_length = len(word_indices), max(description_lengths)

        # 初始化一个用来储存描述单词索引的数组
        padded_indices = np.zeros((batch_size, max_length), np.int)

        # 将每个描述的单词索引列表填充到 padded_indices 中，保证每个描述都有相同的长度，不足的部分用 0 填充
        for i, caption_length in enumerate(description_lengths):
            padded_indices[i, :caption_length] = word_indices[i]

        # 将填充后的索引转换为 PyTorch 张量
        padded_indices = torch.from_numpy(padded_indices)
        padded_indices = padded_indices.to(self.device)  # Possibly move to cuda

        # 使用词嵌入层 self.word_embedding 将填充后的索引转换为单词嵌入表示
        embedded_words = self.word_embedding(padded_indices)

        # 对填充后的嵌入表示进行打包，以便输入到 LSTM 层中
        description_inputs = nn.utils.rnn.pack_padded_sequence(
            embedded_words,
            torch.tensor(description_lengths),
            batch_first=True,
            enforce_sorted=False,
        )
        
        # 根据 LSTM 是否双向，确定隐藏状态的深度。如果 LSTM 是双向的，则隐藏状态深度为 LSTM 层数的两倍，否则为 LSTM 层数
        d = 2 * self.lstm.num_layers if self.lstm.bidirectional else 1 * self.lstm.num_layers
        
        # 初始化hidden_state和cell_state，并移动到GPU上
        h = torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)
        c = torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)

        # 将序列打包后的输入 description_inputs 和初始化的隐藏状态 (h, c) 输入到 LSTM 层中进行前向传播
        # _ 是 LSTM 输出的结果，而 (h, c) 则是 LSTM 最终的隐藏状态
        _, (h, c) = self.lstm(description_inputs, (h, c))

        # 取 LSTM 输出的隐藏状态 h 的平均值，以得到描述的编码表示。dim=0 指定了在哪个维度上进行平均，
        # 这里我们取了所有描述的平均值，得到形状为 [batch_size, embedding_dim] 的描述编码表示。
        description_encodings = torch.mean(h, dim=0)  # [B, DIM] TODO: cat even better?

        return description_encodings

    @property
    def device(self):
        return next(self.lstm.parameters()).device
