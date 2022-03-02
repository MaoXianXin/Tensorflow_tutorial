import pandas as pd
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--term_csv', default='all.csv', type=str)
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

# 对数据集进行shuffle操作
random.shuffle(sentences_list)
nlp = spacy.blank('en')

# 读取csv中存储的术语
ai_csv = pd.read_csv(args.term_csv)
ai_term_list = list(ai_csv['keyword'])
ai_term_list = list(set(ai_term_list))
ai_term_patterns = list(nlp.pipe(ai_term_list))
matcher = PhraseMatcher(nlp.vocab)
matcher.add("tech", ai_term_patterns)

# 在每个句子中进行术语的匹配. 同时规定，每个术语的样本量为 200
train_samples = []
term_dict = {}
for term in ai_term_list:
    term_dict[term] = 0

for doc in nlp.pipe(sentences_list):
    matches = matcher(doc)
    if len(matches) != 0:
        spans = [Span(doc, start, end, label='TECH') for match_id, start, end in matches]
        spans = filter_spans(spans)
        cnt = 0
        for span in spans:
            if term_dict[span.text] <= 200:
                term_dict[span.text] = term_dict[span.text] + 1
                if cnt == 0:
                    sample = [(span.text, span.label_)]
                    cnt += 1
                else:
                    sample.append((span.text, span.label_))

        if len(sample) != 0:
            train_samples.append([doc.text, sample])
            sample = []

data = pd.DataFrame(train_samples, columns=['text', 'annotation'])
data.to_csv('entity.csv', index=False)


# 用于创建BIO schema，也就是模型需要的训练数据格式
def matcher(string, pattern):
    '''
    Return the start and end index of any pattern present in the text.
    '''
    match_list = []
    pattern = pattern.strip()
    seqMatch = SequenceMatcher(None, string, pattern, autojunk=False)
    match = seqMatch.find_longest_match(0, len(string), 0, len(pattern))
    if match.size == len(pattern):
        start = match.a
        end = match.a + match.size
        match_tup = (start, end)
        string = string.replace(pattern, "X" * len(pattern), 1)
        match_list.append(match_tup)

    return match_list, string


def mark_sentence(s, match_list):
    '''
    Marks all the entities in the sentence as per the BIO scheme.
    '''
    word_dict = {}
    for word in s.split(' '):
        word_dict[word] = 'O'

    for start, end, e_type in match_list:
        temp_str = s[start:end]
        tmp_list = temp_str.split()
        if len(tmp_list) > 1:
            word_dict[tmp_list[0]] = 'B-' + e_type
            for w in tmp_list[1:]:
                word_dict[w] = 'I-' + e_type
        else:
            word_dict[temp_str] = 'B-' + e_type
    return word_dict


def clean(text):
    '''
    Just a helper fuction to add a space before the punctuations for better tokenization
    '''
    filters = ["!", "#", "$", "%", "&", "(", ")", "/", "*", ".", ":", ";", "<", "=", ">", "?", "@", "[",
               "\\", "]", "_", "`", "{", "}", "~", "'", ","]
    for i in text:
        if i in filters:
            text = text.replace(i, " " + i)

    return text


def create_data(df, filepath):
    '''
    The function responsible for the creation of data in the said format.
    '''
    with open(filepath, 'w') as f:
        for text, annotation in zip(df.text, df.annotation):
            text = clean(text)
            match_list = []
            for i in annotation:
                a, text_ = matcher(text, i[0])
                try:
                    match_list.append((a[0][0], a[0][1], i[1]))
                except IndexError:
                    print(text, match_list, a)
            d = mark_sentence(text, match_list)

            for i in d.keys():
                f.writelines(i + ' ' + d[i] + '\n')
            f.writelines('\n')


def main():
    # An example dataframe.
    # data = pd.DataFrame([["Horses are too tall and they pretend to care about your feelings", [("Horses", "ANIMAL")]],
    #                      ["Who is Shaka Khan?", [("Shaka Khan", "PERSON")]],
    #                      ["I like London and Berlin.", [("London", "LOCATION"), ("Berlin", "LOCATION")]],
    #                      ["There is a banyan tree in the courtyard", [("banyan tree", "TREE")]]],
    #                     columns=['text', 'annotation'])
    # path to save the txt file.
    filepath = 'train/train.txt'
    # creating the file.
    create_data(data, filepath)


if __name__ == '__main__':
    main()
