from baidu_api import translate
from baidu_api import custom_translate
import shutil
import time
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from collections import Iterator

start = time.time()
project_folder = '/home/csdn/Pictures/spring/docs/en/spring-cloud/'
save_folder_name = '/home/csdn/Downloads/Tensorflow_tutorial/flair/translate_folder1/'
md_files = glob.glob(project_folder + '**/*.md', recursive=True)


def translate_task(md_file):
    # md_files[1] = '/home/csdn/Pictures/spring/docs/en/consuming-rest.md'
    # md_file = md_files[1]
    print(md_file)
    with open(md_file) as f:
        en_lines = f.readlines()

    en_lines = [line.replace('\n', '') for line in en_lines]

    ch_lines = []
    flag = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in en_lines:
            # 代码块不翻译
            if line == '```' and flag == 0:
                ch_lines.append(line)
                flag = 1
                continue
            if flag == 1 and line != '```':
                ch_lines.append(line)
                continue
            if line == '```' and flag == 1:
                ch_lines.append(line)
                flag = 0
                continue

            if flag == 0:
                if len(line) != 0:
                    if line.strip().startswith('*') or line.strip().startswith('[') or line.strip().startswith(
                            '#') or line.strip().startswith('-'):
                        ch_lines.append(pool.map(custom_translate, [line]))
                        continue
                    elif '`' in line:
                        # line = line.replace('`', '$')
                        # translate_line = translate(line).replace('$', '`')
                        ch_lines.append(pool.map(translate, [line]))
                        continue
                    # ch_lines.append(translate(line))
                    ch_lines.append(pool.map(translate, [line]))
                else:
                    ch_lines.append(line)

    text_dir = save_folder_name + project_folder.split('/')[-1] + '/'
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    textfile = open(text_dir + md_file.split('/')[-1].split('.')[0] + ".txt", "w")
    for element in ch_lines:
        if isinstance(element, Iterator):
            try:
                textfile.write(next(element) + "\n")
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
