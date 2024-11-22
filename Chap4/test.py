import spacy
from torchtext.data.utils import get_tokenizer

# 加载spaCy的英语模型
nlp = spacy.load("en_core_web_sm")

# 使用模型的名称来初始化tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# 测试分词
text = "This is a test sentence."
tokens = tokenizer(text)
print(tokens)
