import pandas as pd
from nltk.tokenize import sent_tokenize
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans
from pandas import DataFrame
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--term_csv', default='cv_termv0.1.csv', type=str)
parser.add_argument("--data_source", default="/home/csdn/Downloads/Tensorflow_tutorial/spaCy/domain/cs.CV.csv", type=str)
args = parser.parse_args()

# 使用的数据源是Arxiv论文的CV领域
abstract_csv = pd.read_csv(args.data_source)
abstract_list = list(abstract_csv['abstract'])
abstract_list = [abstract.replace('\n', ' ').strip() for abstract in abstract_list]

# 每个论文摘要是一个段落，需要拆分成句子
sentences_list = []
for abstract in abstract_list:
    sentences_list += sent_tokenize(abstract)

nlp = spacy.blank('en')

# 读取csv中存储的术语
ai_csv = pd.read_csv(args.term_csv)
ai_term_list = list(ai_csv['keyword'])
ai_term_list = list(set(ai_term_list))
ai_term_list = [str(term) for term in ai_term_list]
ai_term_patterns = list(nlp.pipe(ai_term_list))
matcher = PhraseMatcher(nlp.vocab)
matcher.add("tech", ai_term_patterns)

# 在每个句子中进行术语的匹配
train_samples = []
for doc in nlp.pipe(sentences_list):
    matches = matcher(doc)
    if len(matches) != 0:
        spans = [Span(doc, start, end, label='TECH') for match_id, start, end in matches]
        spans = filter_spans(spans)
        cnt = 0
        for span in spans:
            if cnt == 0:
                sample = [(span.text, span.label_)]
                cnt += 1
            else:
                sample.append((span.text, span.label_))
        train_samples.append([doc.text, sample])

data = pd.DataFrame(train_samples, columns=['text', 'annotation'])
data.to_csv('entity.csv', index=False)

# 统计每个术语出现的次数
term_dict_cnt = {}
for train_sample in train_samples:
    for annotation in train_sample[1]:
        term, label = annotation
        if term not in term_dict_cnt:
            term_dict_cnt[term] = 1
        else:
            term_dict_cnt[term] = term_dict_cnt[term] + 1

# 对统计结果按照降序排
keyword_sorted = sorted(term_dict_cnt.items(), key=lambda item: item[1], reverse=True)

# 将排序结果存入csv中
a = [term for term, cnt in keyword_sorted]  # 列表a
b = [cnt for term, cnt in keyword_sorted]  # 列表b
e = {"keyword": a,
     "keyword_cnt": b}  # 将列表a，b转换成字典
data = DataFrame(e)  # 将字典转换成为数据框
data.to_csv(args.term_csv, index=False)
