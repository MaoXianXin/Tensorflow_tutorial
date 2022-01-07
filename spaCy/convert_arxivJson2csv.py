# 第一步，把json格式的arxiv数据转成csv
# 第二步，从csv中提取出AI领域的论文
# 第三步，获取AI领域的术语列表，并保存成csv
# 导入所需的package
import json  # 读取数据，我们的数据为json格式
import pandas as pd  # 数据处理，数据分析
import spacy
import time
import csv
from nltk.corpus import stopwords


def readArxivFile(path, columns=['id', 'submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi', 'report-no',
                                 'categories', 'license', 'abstract', 'versions', 'update_date', 'authors_parsed'],
                  count=None):
    data = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if idx == count:
                break

            d = json.loads(line)
            d = {col: d[col] for col in columns}
            data.append(d)

    data = pd.DataFrame(data)
    return data


def convert_df2csv(data_file):
    data = readArxivFile(data_file)
    print(data.categories.nunique())
    data.to_csv('data.csv', index=False)


def get_domain_data(domain, data_file):
    data = pd.read_csv(data_file)

    ai_index = []
    for i in range(len(data)):
        if domain in data['categories'][i]:
            ai_index.append(i)

    ai_data = data.iloc[ai_index, :]
    ai_data.to_csv('ai_data.csv', index=False)


def get_keyword_list(data_file):
    data_ai = pd.read_csv(data_file)
    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')

    keyword_list = []
    start = time.time()
    for i in range(int(len(data_ai))):
        text = data_ai['abstract'][i].replace('\n', ' ')
        doc = nlp(text)

        # Analyze syntax
        for chunk in doc.noun_chunks:
            keyword_list.append(chunk.text)

        # Find named entities, phrases and concepts
        for entity in doc.ents:
            keyword_list.append(entity.text)
    print("Elapsed time: ", time.time() - start)
    print(len(set(keyword_list)))

    keyword_list = list(set(keyword_list))
    stopword = stopwords.words('english')
    fields = ['keyword']
    rows = [[keyword] for keyword in keyword_list if keyword.lower() not in stopword]
    with open('ai_keyword.csv', 'w', encoding='utf8') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(rows)


data_file = '/home/csdn/Downloads/arxiv-metadata-oai-snapshot.json'
convert_df2csv(data_file)

domain = 'cs.AI'
data_file = 'data.csv'
get_domain_data(domain, data_file)

data_file = 'ai_data.csv'
get_keyword_list(data_file)
