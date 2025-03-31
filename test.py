import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from bert_seq2seq.tokenizer import load_chinese_base_vocab, Tokenizer
from bert_seq2seq.bert_cls_classifier import BertClsClassifier

# ========= 配置 =========
vocab_path = "./state_dict/roberta_wwm_vocab.txt"
model_path = "./sentiment_model.bin"
test_csv_path = "data/test.csv"
output_csv_path = "data/test_with_predictions.csv"
batch_size = 32
target_size = 2  # 二分类

# ========= 加载词表和 tokenizer =========
word2idx = load_chinese_base_vocab(vocab_path)
tokenizer = Tokenizer(word2idx)

# ========= 定义 Dataset =========
class TestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids, _ = tokenizer.encode(text)
        token_ids = [tid if tid < len(word2idx) else word2idx["[UNK]"] for tid in token_ids]
        return torch.tensor(token_ids)

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

df = pd.read_csv(test_csv_path)
texts = df["text"].astype(str).tolist()
dataset = TestDataset(texts)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClsClassifier(word2idx, target_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


#推理
all_preds = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting"):
        batch = batch.to(device)
        outputs = model(batch)
        preds = outputs.argmax(dim=-1).tolist()
        all_preds.extend(preds)

#保存结果
df["negative"] = all_preds
df.to_csv(output_csv_path, index=False)
print(f"预测结果已保存至：{output_csv_path}")
