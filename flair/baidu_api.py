import requests
import json
import re

# 设置access token，以及请求的api地址
token = '24.ad76439c7722c0c0680048421fd4c2a1.2592000.1649331835.282335-25728895'
url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + token

# 设置翻译的源语言，以及目标语言
from_lang = 'en'
to_lang = 'zh'
term_ids = ''  # 术语库id，多个逗号隔开

headers = {'Content-Type': 'application/json'}


def preprocess_text(line, tech_term_list, term2integer):
    for term in tech_term_list:
        if line.find(term) != -1:
            line = re.sub(term, term2integer[term], line)
    return line


def post_preprocess_text(translate_result, integer_list, integer2term):
    for integer_str in integer_list:
        if translate_result.find(integer_str) != -1:
            translate_result = re.sub(integer_str, integer2term[integer_str], translate_result)
    return translate_result


# 把一段文本传送给百度翻译接口进行翻译，并取回翻译结果
def translate(word):
    payload = {'q': word, 'from': from_lang, 'to': to_lang, 'termIds': term_ids}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    js_data = json.loads(json.dumps(result, indent=4, ensure_ascii=False))
    content = js_data['result']['trans_result'][0]['dst']  # 提取json数据里面的dst
    return content


# 需要经过特殊处理的文本串，再转送翻译
def custom_translate(text):
    if text.strip().startswith('**'):
        return translate(text)
    elif text.strip().startswith('*'):
        text = text[1:]
        text = translate(text)
        text = '* ' + text
        return text

    if text.strip().startswith('['):
        return text

    if text.strip().startswith('#'):
        text = translate(text)
        for i in range(len(text)):
            if text[i] != '#':
                break
        text = text[:i] + ' ' + text[i:]
        if text.find('#u') != -1:
            text = re.sub('#u', '# ', text)
        return text

    if text.strip().startswith('-'):
        text = translate(text)
        text = '- ' + text[1:]
        return text

    if '`' in text:
        p1 = re.compile(r'[`](.*?)[`]', re.S)  # 最小匹配
        tech_term_list = re.findall(p1, text)
        tech_term_list = ['`' + term + '`' for term in tech_term_list]
        term2integer = {}
        integer2term = {}
        integer_list = []
        for i in range(len(tech_term_list)):
            term2integer[tech_term_list[i]] = str(i + 10000)
            integer2term[str(i + 10000)] = tech_term_list[i]
            integer_list.append(str(i + 10000))
        text = preprocess_text(text, tech_term_list, term2integer)
        text = translate(text)
        text = post_preprocess_text(text, integer_list, integer2term)
        if text.find('`*`*`*`') != -1:
            text = text.replace('`*`*`*`', '`*`')
        return text


translate_result = custom_translate(
    text="### [Code Conventions and Housekeeping](#_code_conventions_and_housekeeping)")
print(translate_result)
