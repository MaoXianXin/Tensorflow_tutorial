import pandas as pd

tech_term = pd.read_csv('./non_tech_term.csv')
print(len(tech_term))
tech_term = tech_term.drop_duplicates()
tech_term = tech_term.dropna()
print(len(tech_term))
tech_term.to_csv('./non_tech_term.csv', index=False)



# import html2text
#
#
# md_text = open('/home/csdn/Pictures/翻译/a.html', 'r', encoding='utf-8').read()
#
# markdown = html2text.html2text(md_text)
# with open('make2.md', 'w', encoding='utf-8') as file:
#     file.write(markdown)


# from tomd import Tomd
#
# md_text = open('/home/csdn/Pictures/翻译/a.html', 'r', encoding='utf-8').read()
# markdown = Tomd(md_text).markdown
# with open('make.md', 'w', encoding='utf-8') as file:
#     file.write(markdown)
#
