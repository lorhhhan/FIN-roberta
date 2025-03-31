本项目基于预训练的RoBERTa模型进行金融领域中文文本的情感分类。

. ├── train.py # 模型训练脚本
  ├── test.py # 模型推理脚本，读取 CSV 写入预测结果
  ├── sentiment_model.bin # 已训练好的模型（本地使用，不上传）
  ├── bert_seq2seq/ # 模型与Tokenizer定义模块
  ├── state_dict/ # 包含预训练模型的 vocab.txt
  ├── data/ 
	├── train.csv # 自行准备的训练数据（text + label） 
	└── test.csv # 待预测的测试数据（text）

本仓库未包含 .bin 模型文件