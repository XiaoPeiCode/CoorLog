# 简单的日志异常检测 Pipeline 使用 BERT 和 KNN 或孪生网络（启用多 GPU 加速）
    
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
import argparse
import torch.nn as nn
import torch.optim as optim
from modules import llm_utils
# 读取配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# 是否启用缓存
use_cache = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 封装数据加载函数
def load_structured_log(file_path, dataset_name,sample_size=None):
    # 创建快速读取文件路径 (例如 .parquet)
    fast_path = file_path.replace('.csv', '.parquet')
    
    # 检查是否存在加速读取的文件
    if os.path.exists(fast_path):
        # 如果加速文件存在，直接读取
        structured_df = pd.read_parquet(fast_path)
    else:
        # 如果加速文件不存在，首次读取 CSV
        structured_df = pd.read_csv(file_path)
        
        # 存储为快速读取格式
        structured_df.to_parquet(fast_path, index=False)


    # structured_df = pd.read_csv(file_path)

    if sample_size:
        structured_df = structured_df.iloc[:sample_size, :]
    if dataset_name == "Zookeeper":
        structured_df["Label"] = structured_df["Level"].apply(lambda x: int(x == 'ERROR'))
#       df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    else:
        structured_df["Label"] = structured_df["Label"].apply(lambda x: int(x != '-'))
    return structured_df

# 封装数据分割函数
def split_dataset(log_df, method='random', test_size=0.2):
    print(f"split_method:{method}")
    if method == 'random':
        return train_test_split(log_df, test_size=test_size, random_state=42)
    elif method == 'sequential':
        split_index = int(len(log_df) * (1 - test_size))
        return log_df.iloc[:split_index], log_df.iloc[split_index:]
    else:
        raise ValueError("Invalid split method. Choose 'random' or 'sequential'.")

# 使用滑动窗口处理数据集
def apply_sliding_window(df, window_size=10, step_size=10):
    windowed_embeddings = []
    windowed_labels = []
    contents = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        # 将每个日志事件的嵌入向量堆叠起来，形成窗口的嵌入向量
        window_embedding = np.concatenate(window['embedding'].values, axis=0)
        content = ""
        for k in window['log_message'].values:
            content =   content  + " - " +  k + '；\n'
        windowed_embeddings.append(window_embedding)
        windowed_labels.append(window['label'].max())  # 如果窗口中有异常，则标记为异常
        contents.append(content)
    return pd.DataFrame({'embedding': windowed_embeddings, 'label': windowed_labels,'content':contents})

# 封装 BERT 嵌入生成函数
def generate_embeddings(log_messages, tokenizer, model, batch_size=32, desc="Generating Embeddings", method='concat'):
    log_dataset = LogDataset(log_messages, tokenizer)
    log_loader = DataLoader(log_dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in tqdm(log_loader, desc=desc):
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

class LogDataset(Dataset):
    def __init__(self, log_messages, tokenizer):
        self.log_messages = log_messages
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.log_messages)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.log_messages[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return tokens

def balanced_sampling_to_1(df, label_column):
    """
    从一个 DataFrame 中根据指定的标签列，采样出所有值为 1 的样本，
    同时随机采样出数量相等的值为 0 的样本。
    
    Parameters:
    - df: pd.DataFrame, 包含数据的 DataFrame
    - label_column: str, 指定的标签列名称
    
    Returns:
    - pd.DataFrame, 平衡采样后的 DataFrame
    """
    # if isinstance(sample_size, (int, float, complex)):

    # 获取所有值为 1 的样本
    ones = df[df[label_column] == 1]

    # 随机采样出与值为 1 的样本数量相等的值为 0 的样本
    zeros = df[df[label_column] == 0].sample(n=len(ones), random_state=42)

    # 合并采样结果
    sampled_df = pd.concat([ones, zeros])

    # 打乱结果顺序
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return sampled_df

# full_embedding_path
def get_train_test_df_with_index(config):
    result_dir = config['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    # 检查是否有可用的 GPU
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # 读取结构化日志文件
    structured_df = load_structured_log(config['structured_log_path'],config['dataset_name'])
    
    # 创建 DataFrame，确保每条日志都有相应的标签
    log_df = pd.DataFrame()
    log_df["log_message"] = structured_df["Content"]
    log_df["label"] = structured_df["Label"]
    
    count_labels_in_intervals(log_df["label"].values,2500)

    # 加载预训练的 BERT 分词器和模型
    tokenizer = BertTokenizer.from_pretrained(config['bert_model'], cache_dir=config['cache_dir'])
    model = BertModel.from_pretrained(config['bert_model'], cache_dir=config['cache_dir'])
    model = torch.nn.DataParallel(model)  # 启用多 GPU 加速
    model = model.to(device)

    train_df = log_df[config["train_start"]:config['train_end']]
    test_df = log_df[config["test_start"]:config['test_end']]

    train_embeddings = generate_embeddings(train_df['log_message'].tolist(), tokenizer, model, method=config['embedding_method'],batch_size=config['emb_batch_size'])
    test_embeddings = generate_embeddings(test_df['log_message'].tolist(), tokenizer, model, method=config['embedding_method'],batch_size=config['emb_batch_size'])


    # 使用滑动窗口处理训练集和测试集
    train_df['embedding'] = train_embeddings
    test_df['embedding'] = test_embeddings

    train_df = apply_sliding_window(train_df, window_size=config['window_size'],step_size=config['step_size'])
    test_df = apply_sliding_window(test_df, window_size=config['window_size'],step_size=config['step_size'])
    
    
    train_embeddings = np.vstack(train_df['embedding'].values)
    test_embeddings = np.vstack(test_df['embedding'].values)

    return train_df,test_df,train_embeddings,test_embeddings

def count_labels_in_intervals(data, interval_size,result_dir=''):
    with open("BGL.txt","w") as f: 
        # 统计每个区间内 1 的数目
        counts = []
        for i in range(0, len(data), interval_size):
            interval = data[i:i + interval_size]
            counts.append(sum(interval))  # 统计 1 的数量
            f.write(f"{i}_{i + interval_size}:{sum(interval)}\n")
    return counts

# full_embedding_path
def main(config):
    result_dir = config['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    # 检查是否有可用的 GPU
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 读取结构化日志文件
    structured_df = load_structured_log(config['structured_log_path'],config['dataset_name'], sample_size=config.get('sample_size'))
    
    # 创建 DataFrame，确保每条日志都有相应的标签
    log_df = pd.DataFrame()
    log_df["log_message"] = structured_df["Content"]
    log_df["label"] = structured_df["Label"]
    count_labels_in_intervals(log_df["label"].values,10000,config['structured_log_path'])
    # 加载预训练的 BERT 分词器和模型
    tokenizer = BertTokenizer.from_pretrained(config['bert_model'], cache_dir=config['cache_dir'])
    model = BertModel.from_pretrained(config['bert_model'], cache_dir=config['cache_dir'])
    model = torch.nn.DataParallel(model)  # 启用多 GPU 加速
    model = model.to(device)

    # 生成并缓存完整数据集的嵌入向量
    full_embedding_path = os.path.join(result_dir, 'full_log_embeddings.pkl')
    if use_cache and os.path.exists(full_embedding_path):
        print("load(full_embedding_path)")
        log_df = joblib.load(full_embedding_path)
    else:
        embeddings = generate_embeddings(log_df['log_message'].tolist(), tokenizer, model, method=config['embedding_method'],batch_size=1024)
        log_df['embedding'] = embeddings
        joblib.dump(log_df, full_embedding_path)

    # 划分训练集和测试集
    train_df, test_df = split_dataset(log_df, method=config['split_method'], test_size=config['test_size'])

    # 使用滑动窗口处理训练集和测试集
    train_df = apply_sliding_window(train_df, window_size=config['window_size'],step_size=config['step_size'])
    test_df = apply_sliding_window(test_df, window_size=config['window_size'],step_size=config['step_size'])
    print(test_df)
    # 嵌入已经生成，可以直接使用，无需重复生成
    train_embeddings = np.vstack(train_df['embedding'].values)
    test_embeddings = np.vstack(test_df['embedding'].values)

    return train_embeddings,test_embeddings


# full_embedding_path
def get_train_test_df(config):
    result_dir = config['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    # 检查是否有可用的 GPU
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # 读取结构化日志文件
    structured_df = load_structured_log(config['structured_log_path'],config['dataset_name'])
    
    # 创建 DataFrame，确保每条日志都有相应的标签
    log_df = pd.DataFrame()
    log_df["log_message"] = structured_df["Content"]
    log_df["label"] = structured_df["Label"]
    
    count_labels_in_intervals(log_df["label"].values,2500)

    # 加载预训练的 BERT 分词器和模型
    tokenizer = BertTokenizer.from_pretrained(config['bert_model'], cache_dir=config['cache_dir'])
    model = BertModel.from_pretrained(config['bert_model'], cache_dir=config['cache_dir'])
    model = torch.nn.DataParallel(model)  # 启用多 GPU 加速
    model = model.to(device)

    train_df = log_df[config["train_start"]:config['train_end']]
    test_df = log_df[config["test_start"]:config['test_end']]

    train_embeddings = generate_embeddings(train_df['log_message'].tolist(), tokenizer, model, method=config['embedding_method'],batch_size=config['emb_batch_size'])
    test_embeddings = generate_embeddings(test_df['log_message'].tolist(), tokenizer, model, method=config['embedding_method'],batch_size=config['emb_batch_size'])


    # 使用滑动窗口处理训练集和测试集
    train_df['embedding'] = train_embeddings
    test_df['embedding'] = test_embeddings

    train_df = apply_sliding_window(train_df, window_size=config['window_size'],step_size=config['step_size'])
    test_df = apply_sliding_window(test_df, window_size=config['window_size'],step_size=config['step_size'])
    
    
    train_embeddings = np.vstack(train_df['embedding'].values)
    test_embeddings = np.vstack(test_df['embedding'].values)

    return train_df,test_df,train_embeddings,test_embeddings
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/config_zoo.yaml', help="Path to the configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    # prompt_config = load_config('./config/prompt/prompt.yaml')

    main(config)

    
