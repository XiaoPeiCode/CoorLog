import os
import pickle

import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
# 设备选择 (支持 GPU 加速)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设你有四张卡，编号为0, 1, 2, 3
device_ids = [3, 0, 1, 2]  # 将第四张卡设置为主卡
# 加载预训练的 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/home/xiaopei/XPLog/cache_dir/')
bert_model = BertModel.from_pretrained('bert-base-uncased', cache_dir='/home/xiaopei/XPLog/cache_dir/')
bert_model = torch.nn.DataParallel(bert_model,device_ids=device_ids).cuda(device_ids[0])  # 启用多 GPU 加速
# bert_model = DataParallel(bert_model, device_ids=device_ids)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # 显式指定主卡为 cuda:3
print(device)

bert_model = bert_model.to(device)

# 定义 Dataset 类用于批量处理日志消息
class LogDataset(Dataset):
    def __init__(self, log_messages, tokenizer, max_length=512):
        self.log_messages = log_messages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.log_messages)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.log_messages[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return tokens


# 批量处理日志并获取 BERT 嵌入
def get_bert_embeddings_batch(log_messages, tokenizer, model, batch_size=32):
    dataset = LogDataset(log_messages, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state

            # 取 [CLS] 位置的 Token 作为嵌入
            cls_embedding = token_embeddings[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embedding)

    return np.array(embeddings)


# 处理日志数据并计算窗口 embedding
def process_logs(data, tokenizer, model, batch_size=32, embedding_method='mean'):
    """
    处理日志数据并计算窗口 embedding

    :param data: 日志数据 (dict)
    :param tokenizer: BERT Tokenizer
    :param model: BERT 模型
    :param batch_size: 处理批量大小
    :param method: 计算 embedding 方法 ('mean' 或 'concat')
    :return: DataFrame，包含 ['embedding', 'content', 'label']
    """
    contents = []
    labels = []
    embeddings = []
    block_mapping = {}

    # 组织日志数据
    pbar = tqdm(total=len(data.values()), desc="Processing")

    for block_data in data.values():
        content = ""
        for k in block_data['Content']:
            content = content + " - " + k + ';\n'
        labels.append(block_data["label"])

        embedding = get_bert_embeddings_batch(block_data['Content'],tokenizer,model)
        if embedding_method == 'concat':
            window_embedding = np.concatenate(embedding, axis=0)
        elif embedding_method == 'individual':
            window_embedding = np.vstack(embedding)
        elif embedding_method == 'mean':
            window_embedding = np.mean(np.vstack(embedding), axis=0)  # 计算均值 embedding
        else:
            raise ValueError("Invalid embedding_method. Choose 'individual' or 'mean'.")
        embeddings.append(window_embedding)
        contents.append(content)
        pbar.update(1)

    pbar.close()


    # 生成 DataFrame
    df = pd.DataFrame({
        'embedding': list(embeddings),
        'content': contents,
        'label': labels
    })

    return df


def load_evlog_data(file_path="./hadoop2/",use_cache=True,embedding_method='mean'):
    results = {}
    cache_file = os.path.join(file_path,f"cache_results_{embedding_method}.pkl")
    if use_cache and os.path.exists(cache_file):
        print(f"use_cache... ")
        with open(cache_file,'rb') as f:
            results = pickle.load(f)
    else:
        for file in ['test','train','valid']:
            with open(os.path.join(file_path,f"session_{file}.pkl"), 'rb') as f:
                data = pickle.load(f)
            print(f"process logs {os.path.join(file_path,f'session_{file}.pkl')} ")
            df = process_logs(data, tokenizer, bert_model, embedding_method=embedding_method)
            results[file] = df
            df[0:2000].to_csv(os.path.join(file_path,f"{file}_df_2000.csv"))
            with open(os.path.join(file_path,f"{file}_df.pkl"),'wb') as f:
                pickle.dump(df,f)
        with open(cache_file,'wb') as f:
            pickle.dump(results,f)
        # process_logs(hadoop_data)
    return results

# results
if __name__ == '__main__':
    # 示例使用
    spark_data = {
        "Block_1": {
            "label": 0,
            "templates": [],
            "Content": ["Task 1 started", "Task 1 completed", "Task 2 started"],
            "rac_label": []
        },
        "Block_2": {
            "label": 1,
            "templates": [],
            "Content": ["Error detected in node", "Restarting system"],
            "rac_label": []
        }
    }

    # df_mean = process_logs(spark_data, tokenizer, bert_model, embedding_method='mean')
    # df_individual = process_logs(spark_data, tokenizer, bert_model, embedding_method='individual')
    # df_concat = process_logs(spark_data, tokenizer, bert_model, embedding_method='concat')
    results = load_evlog_data(file_path="/home/xiaopei/XPLog/Dataset/Logevol/hadoop3",use_cache=True)

    # results = load_evlog_data(file_path="/home/xiaopei/XPLog/Dataset/Logevol/hadoop2",use_cache=True)

    # results = load_evlog_data(file_path="/home/xiaopei/XPLog/Dataset/Logevol/hadoop2",use_cache=True)
    # results = load_evlog_data(file_path="/home/xiaopei/XPLog/Dataset/Logevol/hadoop3",use_cache=True)
    # results = load_evlog_data(file_path="/home/xiaopei/XPLog/Dataset/Logevol/hadoop3",use_cache=True)
    # results = load_evlog_data(file_path="/home/xiaopei/XPLog/Dataset/Logevol/hadoop3",use_cache=True)
