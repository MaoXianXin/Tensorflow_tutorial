# 第一步，合并多个csv
# 第二步，按照索引切分dataframe，并分别保存成csv
# 第三步，把csv转成markdown，用于仓库协作
import os
import pandas as pd


def merge_csv(data_dir):
    data_files = os.listdir(data_dir)
    init_csv = pd.read_csv(data_dir + data_files[0])
    for i in range(1, len(data_files)):
        init_csv = pd.concat([init_csv, pd.read_csv(data_dir + data_files[i])], axis=0)

    init_csv.drop_duplicates(subset=['术语'], inplace=True)
    init_csv.to_csv('csv_merge.csv', index=False)


def split_df_by_index(data_file):
    csdn_so = pd.read_csv(data_file)

    start = 0

    while start < len(csdn_so):
        end = start + 100
        csdn_so.iloc[start:end, :].to_csv('./csv_save/' + str(start) + '.csv', index=False)
        start = start + 100


def convert_csv2md(save_dir, csv_dir):
    file_names = os.listdir(csv_dir)
    for file in file_names:
        file_id = file.split('.')[0]
        cmd = 'mdtable -s ' + save_dir + file_id + '.md ' + 'csv_save/' + file_id + '.csv'
        os.system(cmd)


data_dir = '/home/csdn/Downloads/Tensorflow_tutorial/spaCy/csv_merge/'
merge_csv(data_dir)

data_file = './csv_merge.csv'
split_df_by_index(data_file)

save_dir = '/home/csdn/Downloads/Github_repo/Open_source_project_termbase/test/'
csv_dir = './csv_save'
convert_csv2md(save_dir, csv_dir)
