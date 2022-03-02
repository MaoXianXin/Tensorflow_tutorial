from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd
import csv
from nltk.tokenize import sent_tokenize
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--term_csv', default='cv_termv0.1.csv', type=str)
parser.add_argument("--data_source", default="/home/csdn/Downloads/Tensorflow_tutorial/spaCy/domain/cs.CV.csv", type=str)
args = parser.parse_args()

# load the trained model
model = SequenceTagger.load(
    '/home/csdn/Downloads/Tensorflow_tutorial/flair/resources/taggers/sota-ner-flert/final-model.pt')

# 使用的数据源是Arxiv论文的CV领域
abstract_csv = pd.read_csv(args.data_source)
abstract_list = list(abstract_csv['abstract'])
abstract_list = [abstract.replace('\n', ' ').strip() for abstract in abstract_list]

# 每个论文摘要是一个段落，需要拆分成句子
# sentences_list = []
# for abstract in abstract_list:
#     sentences_list += sent_tokenize(abstract)

# 收集预测出来的术语
entities = []
for abstract in abstract_list:
    # create example sentence
    sentence = Sentence(abstract)
    # predict the tags
    model.predict(sentence)

    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        entities.append(entity.text)

# 将预测出来的术语存入csv中
entities_dup = list(set(entities))
fields = ['keyword']
rows = [[sub] for sub in list(entities_dup)]

with open(args.term_csv, 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)
