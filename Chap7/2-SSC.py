import numpy as np
import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

# 加载训练数据、分词器、预训练模型、评价指标
dataset = load_dataset('glue', 'sst2')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)
metric = evaluate.load('accuracy')

# 对训练集进行分词
def tokenize(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

dataset = dataset.map(tokenize, batched=True)
encoded_dataset = dataset.map(lambda examples:{'labels':examples['label']}, batched=True)

# 将数据格式化为torch.Tensor类型以训练pytorch模型
columns =  ['input_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns)

# 定义评价指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)

# 定义训练参数
args = TrainingArguments(
    'ft-sst2',                           # 输出路径
    evaluation_strategy="epoch",         # 评价策略:每轮结束后进行评价
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
)

# 定义训练器
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
