import faiss
import numpy as np
import hdbscan
import joblib
import os

def apply_faiss_hdbscan(train_embeddings, train_labels, num_neighbors=10, min_cluster_size=15, min_samples=1, cache_path=None):
    """
    结合 FAISS 和 HDBSCAN 进行聚类降维，加速大规模数据集处理。
    
    :param train_embeddings: 训练数据的嵌入 (numpy array)
    :param train_labels: 训练数据的标签 (numpy array)
    :param num_neighbors: FAISS 近邻搜索的邻居数量
    :param min_cluster_size: HDBSCAN 最小簇大小
    :param min_samples: HDBSCAN 最小样本数
    :param cache_path: 结果缓存路径，避免重复计算
    :return: (降维后的 train_embeddings, train_labels)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached FAISS + HDBSCAN results from {cache_path}")
        return joblib.load(cache_path)

    print("Applying FAISS for fast nearest neighbor search...")

    # 初始化 FAISS Index（L2 归一化 + 余弦相似度）
    d = train_embeddings.shape[1]  # 维度
    index = faiss.IndexFlatL2(d)
    index.add(train_embeddings.astype(np.float32))

    # 搜索最近邻
    distances, indices = index.search(train_embeddings.astype(np.float32), num_neighbors)

    # 计算每个点的平均最近邻距离
    avg_distances = np.mean(distances, axis=1)

    # 设定阈值（Q10，即最小 10% 的距离）
    distance_threshold = np.percentile(avg_distances, 10)
    high_confidence_mask = avg_distances <= distance_threshold

    # 仅对高置信度样本运行 HDBSCAN
    print(f"Running HDBSCAN on {np.sum(high_confidence_mask)} selected samples...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(train_embeddings[high_confidence_mask])

    # 过滤 HDBSCAN 识别出的核心点
    hdbscan_mask = cluster_labels != -1
    final_mask = high_confidence_mask
    final_mask[high_confidence_mask] = hdbscan_mask  # 更新 mask，仅保留核心点

    # 筛选最终数据
    reduced_embeddings = train_embeddings[final_mask]
    reduced_labels = train_labels[final_mask]

    # 缓存结果
    if cache_path:
        joblib.dump((reduced_embeddings, reduced_labels), cache_path)

    print(f"Reduced dataset from {train_embeddings.shape[0]} → {reduced_embeddings.shape[0]} samples")
    return reduced_embeddings, reduced_labels


def apply_hdbscan(train_embeddings, train_labels, min_cluster_size=15, min_samples=1, cache_path=None):
    """
    使用 HDBSCAN 进行聚类，并同步更新 train_labels
    :param train_embeddings: 原始训练集的嵌入 (numpy array)
    :param train_labels: 原始训练集的标签 (numpy array)
    :param min_cluster_size: HDBSCAN 最小簇大小
    :param min_samples: HDBSCAN 最小样本数
    :param cache_path: 如果提供，则尝试从缓存加载 HDBSCAN 结果，避免重复计算
    :return: (HDBSCAN 处理后的 train_embeddings, train_labels)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached HDBSCAN results from {cache_path}")
        return joblib.load(cache_path)

    print("Applying HDBSCAN clustering for downsampling...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(train_embeddings)

    # 仅保留核心点
    mask = (cluster_labels != -1) | (train_labels == 1)
    filtered_embeddings = train_embeddings[mask]
    filtered_labels = train_labels[mask]  # 确保标签也对应减少

    if cache_path:
        joblib.dump((filtered_embeddings, filtered_labels), cache_path)
    
    # if config['cluster_detail']:
        # 统计聚类前后样本数量
        # print(f"Original samples: {train_embeddings.shape[0]}")
        # print(f"Reduced samples: {filtered_embeddings.shape[0]} ({100 * filtered_embeddings.shape[0] / train_embeddings.shape[0]:.2f}% retained)")
        
    # 统计正常/异常样本比例
    normal_before = np.sum(train_labels == 0)
    abnormal_before = np.sum(train_labels == 1)
    normal_after = np.sum(filtered_labels == 0)
    abnormal_after = np.sum(filtered_labels == 1)

    print(f"Before HDBSCAN: Normal={normal_before}, Abnormal={abnormal_before}")
    print(f"After HDBSCAN: Normal={normal_after}, Abnormal={abnormal_after}")

    print(f"Reduced training embeddings from {train_embeddings.shape[0]} to {filtered_embeddings.shape[0]} ({100 * filtered_embeddings.shape[0] / train_embeddings.shape[0]:.2f}% using HDBSCAN")
    return filtered_embeddings, filtered_labels

