from goose3 import Goose
from goose3.text import StopWordsChinese
# 初始化，设置中文分词
g = Goose()
# 文章地址
url = 'https://medium.com/georgian-impact-blog/state-of-computer-vision-cvpr-2021-7c02b60e70e2'
# 获取文章内容
article = g.extract(url=url)

# 显示正文
article_text = article.cleaned_text
print(article_text)


from flair.data import Sentence
from flair.models import SequenceTagger

# make a sentence
sentence = Sentence(article_text)
# print(sentence)

# load the NER tagger
tagger = SequenceTagger.load('ner')

# run NER over sentence
tagger.predict(sentence)

# print(sentence)
print('The following NER tags are found:')

all_entity = []
# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)
    all_entity.append(entity.text)

all_entity = list(set(all_entity))


from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['wiki_es']
collection = db['wiki_es_dup']

for term in all_entity:
    myquery = {"page_title": term}

    mydoc = collection.find(myquery)
    for x in mydoc:
        print(x)


