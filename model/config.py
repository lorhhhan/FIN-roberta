import os
import torch

class Config:
    def __init__(self):
        # 数据路径
        self.raw_data_path = "./data/ccf_data.csv"  # 原始CCF数据路径
        self.processed_data_path = "./data/processed_data.txt"  # 预处理后数据路径

        # 模型参数
        self.model_type = "roberta"  
        self.pretrain_model_path = "./state_dict/roberta_wwm_pytorch_model.bin"  # 预训练模型路径
        self.vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # 词表路径
        self.num_classes = 2  # 二分类任务

        # 训练超参数
        self.batch_size = 4
        self.learning_rate = 1e-5
        self.weight_decay = 1e-4  # AdamW优化器的L2正则化
        self.epochs = 5
        self.max_seq_len = 128  # 文本最大长度

        # 保存
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_save_path = "./model/sentiment_model.bin"
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)


# 实例化配置对象
config = Config()