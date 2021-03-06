import pandas as pd
from baidu_api import translate
from baidu_api import custom_translate
import shutil
import time
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from collections import Iterator
import re
import unicodedata
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--project_folder', default='/home/csdn/Pictures/spring/docs/en/spring-cloud/', type=str)
# parser.add_argument("--save_folder_name", default='/home/csdn/Pictures/translate_folder/test_folder/', type=str)
parser.add_argument("--tech_term_csv", default='/home/csdn/Documents/data_center/dict_project/tech_term.csv', type=str)
args = parser.parse_args()

start = time.time()
project_folder = args.project_folder
# save_folder_name = args.save_folder_name
md_files = glob.glob(project_folder + '**/*.md', recursive=True)
tech_term = pd.read_csv(args.tech_term_csv)
tech_term_list = list(set(tech_term['origin']))
tech_term_dict = {}
for term in tech_term_list:
    tech_term_dict[term] = len(term.split(' '))

tech_term_dict = sorted(tech_term_dict.items(), key=lambda item: item[1], reverse=True)
tech_term_list = [term for term, cnt in tech_term_dict]
term2integer = {}
integer2term = {}
integer_list = []
for i in range(len(tech_term_list)):
    term2integer[tech_term_list[i]] = str(i + 10000)
    integer2term[str(i + 10000)] = tech_term_list[i]
    integer_list.append(str(i + 10000))


def preprocess_text(line):
    for term in tech_term_list:
        if line.find(term) != -1:
            line = re.sub(term, term2integer[term], line)
    return line


def post_preprocess_text(translate_result):
    for integer_str in integer_list:
        if translate_result.find(integer_str) != -1:
            translate_result = re.sub(integer_str, integer2term[integer_str], translate_result)
    return translate_result


def translate_task(md_file):
    print(md_file)
    with open(md_file) as f:
        en_lines = f.readlines()

    en_lines = [line.replace('\n', '') for line in en_lines]

    ch_lines = []
    flag = 0
    # ???????????????
    with ThreadPoolExecutor(max_workers=50) as pool:
        for line in en_lines:
            # ??????????????????
            if line.startswith('```') and flag == 0:
                ch_lines.append(line)
                flag = 1
                continue
            if flag == 1 and line != '```':
                ch_lines.append(line)
                continue
            if line.startswith('```') and flag == 1:
                ch_lines.append(line)
                flag = 0
                continue

            if flag == 0:
                if len(line) != 0:
                    # ????????????????????????????????????????????????????????????????????????
                    if line.strip().startswith('*') or line.strip().startswith('[') or line.strip().startswith(
                            '#') or line.strip().startswith('-'):
                        line = preprocess_text(line)
                        ch_lines.append(pool.map(custom_translate, [line]))
                        continue
                    elif '`' in line:
                        line = preprocess_text(line)
                        ch_lines.append(pool.map(custom_translate, [line]))
                        continue
                    line = preprocess_text(line)
                    ch_lines.append(pool.map(translate, [line]))
                else:
                    ch_lines.append(line)

    # text_dir = save_folder_name + project_folder.split('/')[-1] + '/'
    # if not os.path.exists(text_dir):
    #     os.makedirs(text_dir)
    text_dir = '/'.join(md_file.split('/')[:-1]) + '/'

    textfile = open(text_dir + md_file.split('/')[-1].split('.')[0] + ".txt", "w")
    for element in ch_lines:
        if isinstance(element, Iterator):
            try:
                translate_result = post_preprocess_text(next(element))
                # ?????????????????????????????????
                translate_result = unicodedata.normalize('NFKC', translate_result)
                textfile.write(translate_result + "\n")
            except:
                pass
        else:
            textfile.write(element + "\n")
    textfile.close()

    shutil.copyfile(text_dir + md_file.split('/')[-1].split('.')[0] + ".txt",
                    text_dir + md_file.split('/')[-1].split('.')[0] + ".md")
    os.remove(text_dir + md_file.split('/')[-1].split('.')[0] + ".txt")
    print('elapsed time: ', time.time() - start)


for i in range(len(md_files)):
    translate_task(md_files[i])
