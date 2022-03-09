from pymongo import MongoClient
import pandas as pd
import csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--term_csv', default='cv_termv0.1.csv', type=str)
parser.add_argument("--mongo_db", default="arxiv", type=str)
parser.add_argument("--mongo_collection", default="cv_termv0.5", type=str)
args = parser.parse_args()

term_csv = pd.read_csv(args.term_csv)
term_list = list(term_csv['keyword'])
term_list = [str(term) for term in term_list]

client = MongoClient('localhost', 27017)
db = client[args.mongo_db]
collection = db[args.mongo_collection]

no_exist_term = []
for term in term_list:
    myquery = {"keyword": term}

    collection.insert(myquery)