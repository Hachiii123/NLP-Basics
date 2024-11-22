class MaskedLmInstance:
    def __init__(self, index, label):
        """
        index: 被掩码的位置
        label: 掩码位置对应的原始词
        """
        self.index = index
        self.label = label

    def __repr__(self):
        return f"MaskedLmInstance(index={self.index}, label={self.label})"


# 生成 MLM 训练数据
def create_mlm_lm_prediction(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """
    Create MLM training data for language modeling.
    tokens:输入文本
    masked_lm_prob:掩码语言模型的概率
    max_prediction_per_seq:每个序列的允许的最大掩码数目（控制掩码数目不会太多，避免信息丢失严重）
    vocab_words:词汇表
    rng:随机数生成器
    """

    cnd_indexes = []   # 存储可以参与掩码的token下标
    
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":   # 忽略[CLS]和[SEP]
            continue
        cand_indexes = []

        rng.shuffle(cand_indexes)     # 随机打乱所有下标
        output_tokens = list(tokens)  # 存储掩码后的输入序列，初始化为原始输入
        num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))   # 计算预测数目

        masked_lms = []        # 存储掩码实例
        covered_indexes = set()   # 存储已经处理过的下标
    
    for index in cand_indexes:
        if(len(masked_lms) >= num_to_predict):
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        
        masked_token = None
        # 80%的概率替换为[MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10%的概率保持不变
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10%的概率替换为随机词
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
            
        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        
    masked_lms = sorted(masked_lms, key=lambda x: x.index)  
    masked_lm_positions = []   # 存储需要掩码的下标
    masked_lm_labels = []      # 存储掩码前的原词
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)
        





