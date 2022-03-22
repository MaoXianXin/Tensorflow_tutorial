# import json
# import pickle
#
# input_file = "/home/csdn/Downloads/arxiv-metadata-oai-snapshot.json"
# output_file = "./cs_papers.json"
#
# with open(output_file, 'rb') as fp:
#     TRAIN_DATA = pickle.load(fp)
#
# from spacy.lang.en import English
#
# nlp = English()
# ruler = nlp.add_pipe("entity_ruler")
# patterns = [{"label": "CLS", "pattern": [{"LOWER": "image"}, {"LOWER": "classification"}]},
#             {"label": "OBJ", "pattern": [{"LOWER": "object"}, {"LOWER": "detection"}]},
#             {"label": "SEG", "pattern": [{"LOWER": "image"}, {"LOWER": "segmentation"}]},
#             {"label": "NN", "pattern": [{"LOWER": "neural"}, {"LOWER": "network"}]}, ]
# ruler.add_patterns(patterns)
#
# training_data = []
#
# for sentence in TRAIN_DATA[:100]:
#     sentence = json.loads(sentence)['abstract'].replace('\n', ' ')
#     doc = nlp(sentence)
#     entities = []
#     for ent in doc.ents:
#         if len(ent) > 0:
#             entities.append((ent.start_char, ent.end_char, ent.label_))
#     # [training_data.append((sentence, {"entities": [(ent.start_char, ent.end_char, ent.label_)]})) for ent in doc.ents if
#     #  len(ent) > 0]


# # Import and load the spacy model
# import spacy
#
# nlp = spacy.load("en_core_web_sm")
# print(nlp.pipe_names)
#
# # Getting the ner component
# ner = nlp.get_pipe('ner')
#
# # Training examples in the required format
# TRAIN_DATA = training_data
#
# # Adding labels to the `ner`
# for _, annotations in TRAIN_DATA[:1000]:
#     for ent in annotations.get("entities"):
#         ner.add_label(ent[2])
#
# # Resume training
# optimizer = nlp.resume_training()
# move_names = list(ner.move_names)
#
# # List of pipes you want to train
# pipe_exceptions = ["tok2vec", "ner"]
#
# # List of pipes which should remain unaffected in training
# unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
#
# # Import requirements
# import random
# from spacy.training.example import Example
#
# # TRAINING THE MODEL
# with nlp.disable_pipes(*unaffected_pipes):
#     # Training for 30 iterations
#     for iteration in range(30):
#         # shuffling examples before every iteration
#         random.shuffle(TRAIN_DATA)
#         losses = {}
#         for batch in spacy.util.minibatch(TRAIN_DATA, size=8):
#             for text, annotations in batch:
#                 # create Example
#                 doc = nlp.make_doc(text)
#                 example = Example.from_dict(doc, annotations)
#                 # Update the model
#                 nlp.update([example], losses=losses, drop=0.3)
#                 print("Losses", losses)
#
# # Testing the model
# doc = nlp("I ate Sushi yesterday. Maggi is a common fast food")
# print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
#
# from pathlib import Path
# # Save the model to directory
# output_dir = Path('./content/')
# nlp.to_disk(output_dir)
# print("Saved model to", output_dir)

# import pandas as pd
#
# csv = pd.read_csv('data_merge.csv')
#
# csv = csv.drop_duplicates(subset=['categories'])
#
# csv.to_csv('domain_categories1.csv', index=False, columns=['categories', 'category_name', 'category_description'])


# !/usr/bin/env python
# _*_ coding:utf-8 _*_

# import os
# import pandas as pd
#
# csv_merge = pd.read_csv('roles.csv')
# csv_merge[:10000].to_csv('roles1.csv', index=False)
# csv_merge[':LABEL'] = 'Keyword'
# csv_merge['personId:ID'] = [i for i in range(len(csv_merge))]
# csv_merge.to_csv('csv_merge.csv', columns=['personId:ID', 'keyword', 'role', ':LABEL'], index=False)

# def merge_csv(data_dir):
#     data_files = os.listdir(data_dir)
#     init_csv = pd.read_csv(data_dir + data_files[0])
#     for i in range(1, len(data_files)):
#         init_csv = pd.concat([init_csv, pd.read_csv(data_dir + data_files[i])], axis=0)
#
#     init_csv.to_csv('csv_merge.csv', index=False)
#
#
# data_dir = './keyword/'
# merge_csv(data_dir)

# csv_files = os.listdir('keyword')
# for file in csv_files:
#     print(file)
#     csv = pd.read_csv('./keyword/' + file)
#     csv['role'] = file.split('_')[0]
#     csv.to_csv('./keyword/' + file, index=False, columns=['role', 'keyword'])
