import random
import hashlib
import urllib
import requests
import json  # 安装相应的库

src = 'en'  # 翻译的源语言
obj = 'zh'  # 翻译的目标语言
appid = '20220304001110302'  # 这里输入你注册后得到的appid
secretKey = 'nT75jNiwQylE1jAwkcTT'  # 这里输入你注册后得到的密匙

myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'  # 必须加上的头


def translate(word):
    salt = random.randint(31256, 66253)  # 产生随计数

    sign = appid + word + str(salt) + secretKey  # 文档的step1拼接字符串
    m1 = hashlib.md5()
    m1.update(sign.encode('utf-8'))
    sign = m1.hexdigest()  # 文档的step2计算签名
    myur1 = myurl + '?q=' + urllib.parse.quote(
        word) + '&from=' + src + '&to=' + obj + '&appid=' + appid + '&salt=' + str(salt) + '&sign=' + sign
    print(myur1)  # 生成的url并打印出来
    english_data = requests.get(myur1)  # 请求url
    js_data = json.loads(english_data.text)  # 下载json数据
    content = js_data['trans_result'][0]['dst']  # 提取json数据里面的dst
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
