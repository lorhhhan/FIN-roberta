# 基于RoBERTa模型的金融领域中文文本的情感分类
基于预训练的RoBERTa模型，数据集选择CCF，实现金融领域中文文本的情感二分类，适用于新闻、公告等金融文本的情绪判断 。

本项目基于bert_seq2seq (https://github.com/920232796/bert_seq2seq) 。
#  项目结构

```bash
.
├── train.py                # 模型训练脚本
├── test.py                 # 模型推理脚本（读取CSV输出预测结果）
├── sentiment_model.bin     # 已训练模型（本地使用，未上传）
├── bert_seq2seq/           # 模型与Tokenizer定义模块
├── state_dict/             # 预训练模型词表（roberta_wwm_pytorch_model.bin未上传）
├── model/		    # config参数
├── .ipynb_checkpoints/             
└── data/		    #数据集
    ├── train.csv          
    └── test.csv            

本仓库未包含 .bin 模型文件
