import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

print(pos_tag(word_tokenize('They sat by the fire.')))
