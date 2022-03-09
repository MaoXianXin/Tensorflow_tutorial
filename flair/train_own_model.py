from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import TransformerWordEmbeddings
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_model', default='bert-base-uncased', type=str)
parser.add_argument("--model_save_dir", default="resources/taggers/", type=str)
parser.add_argument("--batch_size", default="16", type=int)
args = parser.parse_args()

# define columns
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = './train'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')
# corpus = corpus.downsample(0.1)
print(len(corpus.train))

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize fine-tuneable transformer embeddings WITH document context
embeddings = TransformerWordEmbeddings(model=args.train_model,
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       )

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        use_crf=True,
                        use_rnn=False,
                        reproject_embeddings=False,
                        )

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. run fine-tuning
trainer.fine_tune(args.model_save_dir + args.train_model,
                  learning_rate=5.0e-6,
                  max_epochs=20,
                  mini_batch_size=args.batch_size,
                  # mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
                  )
