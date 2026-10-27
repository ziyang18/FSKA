import os
from typing import List, Tuple, Optional

def load_rt_polarity(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    加载 RT Polarity 数据集
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        texts: 文本列表
        labels: 标签列表 (0=neg, 1=pos)
    """
    pos_path = os.path.join(data_dir, "rt-polaritydata", "rt-polaritydata", "rt-polarity.pos")
    neg_path = os.path.join(data_dir, "rt-polaritydata", "rt-polaritydata", "rt-polarity.neg")
    
    if not os.path.isfile(pos_path):
        pos_path = os.path.join(data_dir, "rt-polaritydata", "rt-polarity.pos")
        neg_path = os.path.join(data_dir, "rt-polaritydata", "rt-polarity.neg")

    # 尝试不同的编码
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    texts = []
    labels = []
    
    for encoding in encodings:
        try:
            with open(pos_path, 'r', encoding=encoding) as f:
                pos_texts = [line.strip() for line in f if line.strip()]
            with open(neg_path, 'r', encoding=encoding) as f:
                neg_texts = [line.strip() for line in f if line.strip()]
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法读取 RT Polarity 数据文件")
    
    texts = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    
    return texts, labels

def load_sst(data_dir: str, binary: bool = True) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]]
]:
    """
    加载 Stanford Sentiment Treebank (SST) 数据集
    
    Args:
        data_dir: 数据目录路径
        binary: True 为 SST-2（二分类），False 为 SST-5（五分类）
        
    Returns:
        (train_texts, train_labels),
        (val_texts, val_labels),
        (test_texts, test_labels)
    """
    sst_dir = os.path.join(data_dir, "stanfordSentimentTreebank")
    
    # 加载句子
    sentences = {}
    with open(os.path.join(sst_dir, "datasetSentences.txt"), 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                sent_id = int(parts[0])
                sentence = parts[1]
                sentences[sent_id] = sentence
    
    # 加载 phrase -> sentiment 映射
    phrase_to_sentiment = {}
    
    # 加载字典 (phrase -> phrase_id)
    phrase_to_id = {}
    with open(os.path.join(sst_dir, "dictionary.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2:
                phrase = parts[0]
                phrase_id = int(parts[1])
                phrase_to_id[phrase] = phrase_id
    
    # 加载 sentiment labels
    sentiment_labels = {}
    with open(os.path.join(sst_dir, "sentiment_labels.txt"), 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2:
                phrase_id = int(parts[0])
                sentiment = float(parts[1])
                sentiment_labels[phrase_id] = sentiment
    
    # 加载数据集划分
    splits = {}  # sentence_id -> split (1=train, 2=test, 3=dev)
    with open(os.path.join(sst_dir, "datasetSplit.txt"), 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                sent_id = int(parts[0])
                split_id = int(parts[1])
                splits[sent_id] = split_id
    
    # 构建数据集
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    test_texts, test_labels = [], []
    
    def get_label(sentiment: float, binary: bool) -> Optional[int]:
        """将连续的情感值转换为离散标签"""
        if binary:
            # SST-2: 只保留正面和负面，丢弃中性
            if sentiment <= 0.4:
                return 0  # negative
            elif sentiment > 0.6:
                return 1  # positive
            else:
                return None  # neutral, 丢弃
        else:
            # SST-5: 五分类
            if sentiment <= 0.2:
                return 0  # very negative
            elif sentiment <= 0.4:
                return 1  # negative
            elif sentiment <= 0.6:
                return 2  # neutral
            elif sentiment <= 0.8:
                return 3  # positive
            else:
                return 4  # very positive
    
    for sent_id, sentence in sentences.items():
        if sent_id not in splits:
            continue
        
        # 获取句子的情感值
        if sentence in phrase_to_id:
            phrase_id = phrase_to_id[sentence]
            if phrase_id in sentiment_labels:
                sentiment = sentiment_labels[phrase_id]
                label = get_label(sentiment, binary)
                
                if label is not None:
                    split_id = splits[sent_id]
                    if split_id == 1:
                        train_texts.append(sentence)
                        train_labels.append(label)
                    elif split_id == 2:
                        test_texts.append(sentence)
                        test_labels.append(label)
                    elif split_id == 3:
                        val_texts.append(sentence)
                        val_labels.append(label)
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
