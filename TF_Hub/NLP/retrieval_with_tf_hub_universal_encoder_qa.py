import json
import nltk
import os
import pprint
import random
import simpleneighbors
import urllib
from IPython.display import HTML, display
from tqdm.notebook import tqdm

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

nltk.download('punkt')


def download_squad(url):
    return json.load(urllib.request.urlopen(url))


def extract_sentences_from_squad_json(squad):
    all_sentences = []
    for data in squad['data']:
        for paragraph in data['paragraphs']:
            sentences = nltk.tokenize.sent_tokenize(paragraph['context'])
            all_sentences.extend(zip(sentences, [paragraph['context']] * len(sentences)))
    return list(set(all_sentences))  # remove duplicates


def extract_questions_from_squad_json(squad):
    questions = []
    for data in squad['data']:
        for paragraph in data['paragraphs']:
            for qas in paragraph['qas']:
                if qas['answers']:
                    questions.append((qas['question'], qas['answers'][0]['text']))
    return list(set(questions))


def output_with_highlight(text, highlight):
    output = "<li> "
    i = text.find(highlight)
    while True:
        if i == -1:
            output += text
            break
        output += text[0:i]
        output += '<b>' + text[i:i + len(highlight)] + '</b>'
        text = text[i + len(highlight):]
        i = text.find(highlight)
    return output + "</li>\n"


def display_nearest_neighbors(query_text, answer_text=None):
    query_embedding = model.signatures['question_encoder'](tf.constant([query_text]))['outputs'][0]
    search_results = index.nearest(query_embedding, n=num_results)

    if answer_text:
        result_md = '''
    <p>Random Question from SQuAD:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    <p>Answer:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    ''' % (query_text, answer_text)
    else:
        result_md = '''
    <p>Question:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    ''' % query_text

    result_md += '''
    <p>Retrieved sentences :
    <ol>
  '''

    if answer_text:
        for s in search_results:
            result_md += output_with_highlight(s, answer_text)
    else:
        for s in search_results:
            result_md += '<li>' + s + '</li>\n'

    result_md += "</ol>"
    with open("data.html", "w") as file:
        file.write(result_md)

    display(HTML(result_md))


squad_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'

squad_json = download_squad(squad_url)
sentences = extract_sentences_from_squad_json(squad_json)
questions = extract_questions_from_squad_json(squad_json)
print("%s sentences, %s questions extracted from SQuAD %s" % (len(sentences), len(questions), squad_url))

print("\nExample sentence and context:\n")
sentence = random.choice(sentences)
print("sentence:\n")
pprint.pprint(sentence[0])
print("\ncontext:\n")
pprint.pprint(sentence[1])
print()

module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"
model = hub.load(module_url)

batch_size = 100

encodings = model.signatures['response_encoder'](
    input=tf.constant([sentences[0][0]]),
    context=tf.constant([sentences[0][1]]))
index = simpleneighbors.SimpleNeighbors(
    len(encodings['outputs'][0]), metric='angular')

print('Computing embeddings for %s sentences' % len(sentences))
slices = zip(*(iter(sentences),) * batch_size)
num_batches = int(len(sentences) / batch_size)
for index_num, s in enumerate(slices):
    print('index_num:', index_num)
    response_batch = list([r for r, c in s])
    context_batch = list([c for r, c in s])
    encodings = model.signatures['response_encoder'](
        input=tf.constant(response_batch),
        context=tf.constant(context_batch)
    )
    for batch_index, batch in enumerate(response_batch):
        index.add_one(batch, encodings['outputs'][batch_index])

index.build()
print('simpleneighbors index for %s sentences built.' % len(sentences))

num_results = 25

query = random.choice(questions)
display_nearest_neighbors(query[0], query[1])
