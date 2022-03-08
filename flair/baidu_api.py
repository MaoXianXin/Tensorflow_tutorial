import requests
import json  # 安装相应的库

token = '24.e8e885ceca73fb790397209393485cf3.2592000.1649314375.282335-25725808'
url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + token

# For list of language codes, please refer to `https://ai.baidu.com/ai-doc/MT/4kqryjku9#语种列表`
from_lang = 'en'  # example: en
to_lang = 'zh'  # example: zh
term_ids = ''  # 术语库id，多个逗号隔开

headers = {'Content-Type': 'application/json'}


def translate(word):
    payload = {'q': word, 'from': from_lang, 'to': to_lang, 'termIds': term_ids}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    js_data = json.loads(json.dumps(result, indent=4, ensure_ascii=False))
    content = js_data['result']['trans_result'][0]['dst']  # 提取json数据里面的dst
    return content


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
        index = text.rfind('#')
        text = text[:index + 1] + ' ' + text[index + 1:]
        return text

    if text.strip().startswith('-'):
        text = translate(text)
        text = '- ' + text[1:]
        return text


# translate_result = translate(word="Most %Spring Boot% applications need very little Spring configuration.")
# print(translate_result.replace('%', ' '))
