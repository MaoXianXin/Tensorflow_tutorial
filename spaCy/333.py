import spacy
from spacy import displacy
import pandas as pd

df = pd.read_csv('/home/csdn/Downloads/Tensorflow_tutorial/spaCy/domain/cs.AI.csv')
df = df[df['abstract'].str.contains("regression ", na=False)]
print(df.head())
text = df['abstract'][4].replace('\n', ' ').strip()

nlp = spacy.load("./output/model-best")
doc = nlp(text)
displacy.serve(doc, style="ent")
