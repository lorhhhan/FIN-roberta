import torch
from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from torch.utils.data import Dataset, DataLoader
# from model.roberta_model import load_bert
from bert_seq2seq import load_bert
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# 配置
vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # 词表路径
model_path = "./state_dict/roberta_wwm_pytorch_model.bin"  # 预训练模型
data_path = "./data/cleaned_data.txt"  # 预处理后的数据
model_save_path = "./sentiment_model.bin"
model_name = "roberta" # 选择模型名字
batch_size = 16
lr = 1e-5
num_classes = 2  # 二分类


# torch.cuda.empty_cache()
# 加载词表
word2idx = load_chinese_base_vocab(vocab_path)


class SentimentDataset(Dataset):
    def __init__(self, sents_src, sents_tgt):
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = Tokenizer(word2idx)
        # print("Loaded tokenizer type:", type(self.tokenizer))

    def __getitem__(self, idx):
        text = self.sents_src[idx]
        label = self.sents_tgt[idx]
        token_ids, _ = self.tokenizer.encode(text)

        token_ids = [tid if tid < len(word2idx) else word2idx["[UNK]"] for tid in token_ids]  
        
        return {"token_ids": token_ids, "label": label}  # 返回字典格式

    def __len__(self):
        return len(self.sents_src)


def read_data():
    sents_src, sents_tgt = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            if len(line) == 2:
                sents_tgt.append(int(line[0]))
                sents_src.append(line[1])
    return sents_src, sents_tgt


def collate_fn(batch):
    token_ids = [torch.tensor(item["token_ids"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # 检查是否有非法 token id
    for i, tokens in enumerate(token_ids):
        max_id = tokens.max().item()
        if max_id >= len(word2idx):
            print(f"[ERROR] Token id overflow in sample {i}, max_id={max_id}, vocab_size={len(word2idx)}")
            print("Tokens:", tokens)
            raise ValueError("Token id exceeds vocab size.")

    token_ids_padded = torch.nn.utils.rnn.pad_sequence(
        token_ids,
        batch_first=True,
        padding_value=0
    )
    return token_ids_padded, labels


# def evaluate(model, dataloader, device):
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for token_ids, labels in dataloader:
#             # print("labels:", labels)
#             assert (labels >= 0).all() and (labels < num_classes).all(), "label 越界了"
#             token_ids = token_ids.to(device)
#             labels = labels.to(device)
#             outputs = model(token_ids)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
#     return {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "f1": f1_score(y_true, y_pred, average="binary")
#     }


class Trainer:
    def __init__(self):
        self.sents_src, self.sents_tgt = read_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
         # 划分数据集
        train_src, test_src, train_tgt, test_tgt = train_test_split(
            self.sents_src, self.sents_tgt, test_size=0.2, random_state=42
        )

        # 加载训练集
        train_dataset = SentimentDataset(train_src, train_tgt)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        # 加载测试集
        test_dataset = SentimentDataset(test_src, test_tgt)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        # 加载RoBERTa分类模型
        self.bert_model = load_bert(
            word2idx,
            model_name="roberta",
            model_class="cls",
            target_size=num_classes
        )
        
        self.bert_model.load_pretrain_params(model_path)
        self.bert_model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.bert_model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        # 数据加载
        dataset = SentimentDataset(self.sents_src, self.sents_tgt)
        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    # def train(self, epochs=5):
    #     for epoch in range(epochs):
    #         self.bert_model.train()
    #         total_loss = 0
    #         for token_ids, labels in self.train_loader:
    #             token_ids = token_ids.to(self.device)
    #             labels = labels.to(self.device)

    #             self.optimizer.zero_grad()
    #             outputs = self.bert_model(token_ids)
    #             loss = self.criterion(outputs, labels)
    #             loss.backward()
    #             self.optimizer.step()
    #             total_loss += loss.item()

    #         # 每个epoch评估
    #         # metrics = evaluate(self.bert_model, self.train_loader, self.device)
    #         metrics = evaluate(self.bert_model, self.test_loader, self.device)
    #         print(
    #             f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f} | "
    #             f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}"
    #         )
    def train(self, epochs=5):
        for epoch in range(epochs):
            self.bert_model.train()
            total_loss = 0
            correct = 0
            total = 0
    
            # 使用 tqdm 包装 train_loader
            for token_ids, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device)

    
                self.optimizer.zero_grad()
                outputs = self.bert_model(token_ids)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item()
                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    
            train_acc = correct / total
            train_loss = total_loss / len(self.train_loader)
    
            #测试集评估
            self.bert_model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0
    
            with torch.no_grad():
                for token_ids, labels in tqdm(self.test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]"):
                    token_ids = token_ids.to(self.device)
                    labels = labels.to(self.device)

    
                    outputs = self.bert_model(token_ids)
                    loss = self.criterion(outputs, labels)
    
                    test_loss += loss.item()
                    preds = outputs.argmax(dim=-1)
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)
    
            test_acc = test_correct / test_total
            test_loss = test_loss / len(self.test_loader)
    
            
            print(f"[train] loss: {train_loss:.4f}, acc: {train_acc * 100:.2f}")
            print(f"[test]  loss: {test_loss:.4f}, acc: {test_acc * 100:.2f}")
            
        # 保存模型
        self.bert_model.save_all_params(model_save_path)
        print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()