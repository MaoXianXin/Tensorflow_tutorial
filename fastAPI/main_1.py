from fastapi import FastAPI
import pandas as pd

word_100 = pd.read_csv('./word_100.csv')

keyword_list = []

for index, row in word_100.iterrows():
    # print(row['keyword'], row['desc'])
    keyword_list.append({row['keyword']: row['desc']})

title2class = {'Loss_function': 'Artificial_intelligence'}

app = FastAPI()


@app.get("/title/{title_name}")
async def get_model(title_name: str):
    if title2class[title_name] == 'Artificial_intelligence':
        return {'title_class': {
            'Deep_learning': 'Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.',
            'Machine_learning': 'Machine learning (ML) is the study of computer algorithms that can improve automatically through experience and by the use of data.[1] It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.[2] Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.'}}


@app.get("/keyword/{keyword_request}")
async def get_keyword(keyword_request: str):
    return keyword_list
