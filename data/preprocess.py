import pandas as pd

# 读取CCF
df = pd.read_csv("ccf_data.csv")
print(df.columns.tolist())
df = df.dropna(subset=["text", "negative"])  # 去除空值

# 确保negative只有 0 1
df["label"] = df["negative"].astype(int)
print(df["label"].value_counts())  # 检查类别分布

# 保存为所需格式 label ttext
with open("processed_data.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f"{row['label']}\t{row['text']}\n")