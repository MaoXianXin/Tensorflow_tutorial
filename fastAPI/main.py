from fastapi import FastAPI
from pymongo import MongoClient
from goose3 import Goose
from goose3.text import StopWordsChinese
from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd

word_100 = pd.read_csv('/home/csdn/Documents/word_100.csv')

keyword_list = []

for index, row in word_100.iterrows():
    print(row['keyword'], row['desc'])
    keyword_list.append({row['keyword']: row['desc']})

# load the NER tagger
tagger = SequenceTagger.load('ner')

# 初始化，设置中文分词
g = Goose()

client = MongoClient('localhost', 27017)
db = client['wiki_es']
collection = db['wiki_es_dup']

title2class = {'Loss_function': 'Artificial_intelligence'}

app = FastAPI()


@app.get("/title/{title_name}")
async def get_model(title_name: str):
    if title2class[title_name] == 'Artificial_intelligence':
        return {'title_class': {
            'Deep_learning': 'Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.',
            'Machine_learning': 'Machine learning (ML) is the study of computer algorithms that can improve automatically through experience and by the use of data.[1] It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.[2] Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.'}}


# @app.get("/{url}")
# async def get_model(url: str):
#     return {'name': url}


# url参数可以和param重合
@app.get("/{url}")
def read_root(url: str, parms_1: str, parms_2: str = None):
    # 获取文章内容
    article = g.extract(url=parms_1)
    # 显示正文
    article_text = article.cleaned_text
    print(article_text)

    # make a sentence
    sentence = Sentence(article_text)

    # run NER over sentence
    tagger.predict(sentence)

    all_entity = []
    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        # print(entity)
        all_entity.append(entity.text)

    all_entity = list(set(all_entity))
    print(all_entity)

    x = 'init'
    all_glossary = []
    for term in all_entity:
        # print('---------------')
        myquery = {"page_title": term}

        mydoc = collection.find(myquery)
        for x in mydoc:
            print(x)
            all_glossary.append([x['page_title'], x['page_url'], x['context_page_description']])
    print(all_glossary)
    return {'url地址是: ': url, "parms_1参数是 ": parms_1,
            "Glossary": all_glossary}


@app.get("/keyword/{keyword_request}")
async def get_keyword(keyword_request: str):
    return keyword_list

