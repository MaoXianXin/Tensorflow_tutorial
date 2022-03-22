import bokeh
import bokeh.models
import bokeh.plotting
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import sklearn.metrics.pairwise

from simpleneighbors import SimpleNeighbors
from tqdm import tqdm
from tqdm import trange


def visualize_similarity(embeddings_1, embeddings_2, labels_1, labels_2,
                         plot_title,
                         plot_width=1200, plot_height=600,
                         xaxis_font_size='12pt', yaxis_font_size='12pt'):
    assert len(embeddings_1) == len(labels_1)
    assert len(embeddings_2) == len(labels_2)

    # arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
    sim = 1 - np.arccos(
        sklearn.metrics.pairwise.cosine_similarity(embeddings_1,
                                                   embeddings_2)) / np.pi

    embeddings_1_col, embeddings_2_col, sim_col = [], [], []
    for i in range(len(embeddings_1)):
        for j in range(len(embeddings_2)):
            embeddings_1_col.append(labels_1[i])
            embeddings_2_col.append(labels_2[j])
            sim_col.append(sim[i][j])
    df = pd.DataFrame(zip(embeddings_1_col, embeddings_2_col, sim_col),
                      columns=['embeddings_1', 'embeddings_2', 'sim'])

    mapper = bokeh.models.LinearColorMapper(
        palette=[*reversed(bokeh.palettes.YlOrRd[9])], low=df.sim.min(),
        high=df.sim.max())

    p = bokeh.plotting.figure(title=plot_title, x_range=labels_1,
                              x_axis_location="above",
                              y_range=[*reversed(labels_2)],
                              plot_width=plot_width, plot_height=plot_height,
                              tools="save", toolbar_location='below', tooltips=[
            ('pair', '@embeddings_1 ||| @embeddings_2'),
            ('sim', '@sim')])
    p.rect(x="embeddings_1", y="embeddings_2", width=1, height=1, source=df,
           fill_color={'field': 'sim', 'transform': mapper}, line_color=None)

    p.title.text_font_size = '12pt'
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 16
    p.xaxis.major_label_text_font_size = xaxis_font_size
    p.xaxis.major_label_orientation = 0.25 * np.pi
    p.yaxis.major_label_text_font_size = yaxis_font_size
    p.min_border_right = 300

    bokeh.io.output_notebook()
    bokeh.io.show(p)


# The 16-language multilingual module is the default but feel free
# to pick others from the list and compare the results.
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'

model = hub.load(module_url)


def embed_text(input):
    return model(input)


# Some texts of different lengths in different languages.
arabic_sentences = ['كلب', 'الجراء لطيفة.', 'أستمتع بالمشي لمسافات طويلة على طول الشاطئ مع كلبي.']
chinese_sentences = ['狗', '小狗很好。', '我喜欢和我的狗一起沿着海滩散步。']
english_sentences = ['dog', 'Puppies are nice.', 'I enjoy taking long walks along the beach with my dog.']
french_sentences = ['chien', 'Les chiots sont gentils.',
                    'J\'aime faire de longues promenades sur la plage avec mon chien.']
german_sentences = ['Hund', 'Welpen sind nett.', 'Ich genieße lange Spaziergänge am Strand entlang mit meinem Hund.']
italian_sentences = ['cane', 'I cuccioli sono carini.',
                     'Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.']
japanese_sentences = ['犬', '子犬はいいです', '私は犬と一緒にビーチを散歩するのが好きです']
korean_sentences = ['개', '강아지가 좋다.', '나는 나의 개와 해변을 따라 길게 산책하는 것을 즐긴다.']
russian_sentences = ['собака', 'Милые щенки.', 'Мне нравится подолгу гулять по пляжу со своей собакой.']
spanish_sentences = ['perro', 'Los cachorros son agradables.',
                     'Disfruto de dar largos paseos por la playa con mi perro.']

# Multilingual example
multilingual_example = ["Willkommen zu einfachen, aber", "verrassend krachtige", "multilingüe",
                        "compréhension du langage naturel", "модели.", "大家是什么意思", "보다 중요한", ".اللغة التي يتحدثونها"]
multilingual_example_in_en = ["Welcome to simple yet", "surprisingly powerful", "multilingual",
                              "natural language understanding", "models.", "What people mean", "matters more than",
                              "the language they speak."]

# Compute embeddings.
ar_result = embed_text(arabic_sentences)
en_result = embed_text(english_sentences)
es_result = embed_text(spanish_sentences)
de_result = embed_text(german_sentences)
fr_result = embed_text(french_sentences)
it_result = embed_text(italian_sentences)
ja_result = embed_text(japanese_sentences)
ko_result = embed_text(korean_sentences)
ru_result = embed_text(russian_sentences)
zh_result = embed_text(chinese_sentences)

multilingual_result = embed_text(multilingual_example)
multilingual_in_en_result = embed_text(multilingual_example_in_en)

visualize_similarity(multilingual_in_en_result, multilingual_result,
                     multilingual_example_in_en, multilingual_example,
                     "Multilingual Universal Sentence Encoder for Semantic Retrieval (Yang et al., 2019)")
