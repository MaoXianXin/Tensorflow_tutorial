import urllib.request
import urllib.parse
from lxml import etree
import pandas as pd
import time
import hashlib
import urllib3
import sys

csdn_so = pd.read_csv('/home/csdn/Pictures/csdn_so.csv')
term_len = len(csdn_so)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_version = sys.version_info

is_python3 = (_version[0] == 3)

orderno = "ZF20219881550F64bH"
secret = "4a1536a5560b4459bcdf6b46e9928b47"

ip = "forward.xdaili.cn"
port = "80"

ip_port = ip + ":" + port

timestamp = str(int(time.time()))
string = ""
string = "orderno=" + orderno + "," + "secret=" + secret + "," + "timestamp=" + timestamp

if is_python3:
    string = string.encode()

md5_string = hashlib.md5(string).hexdigest()
sign = md5_string.upper()
# print(sign)
auth = "sign=" + sign + "&" + "orderno=" + orderno + "&" + "timestamp=" + timestamp

# print(auth)
proxy = "http://" + ip_port
# proxy = {"http": "http://" + ip_port, "https": "http://" + ip_port}
headers = {"Proxy-Authorization": auth,
           "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.75 Safari/537.36"}


def query(content):
    # 请求地址
    url = 'https://baike.baidu.com/item/' + urllib.parse.quote(content)
    # 利用请求地址和请求头部构造请求对象
    req = urllib.request.Request(url=url, headers=headers, method='GET')
    req.set_proxy(proxy, "http")
    # 发送请求，获得响应
    response = urllib.request.urlopen(req)
    # 读取响应，获得文本
    text = response.read().decode('utf-8')
    # 构造 _Element 对象
    html = etree.HTML(text)
    # 使用 xpath 匹配数据，得到匹配字符串列表
    sen_list = html.xpath('//div[contains(@class,"lemma-summary") or contains(@class,"lemmaWgt-lemmaSummary")]//text()')
    # 过滤数据，去掉空白
    sen_list_after_filter = [item.strip('\n') for item in sen_list]
    # 将字符串列表连成字符串并返回
    return ''.join(sen_list_after_filter)


if __name__ == '__main__':
    for i in range(term_len):
        content = csdn_so['术语'][i]
        print(content)
        result = query(content)
        csdn_so['术语描述'][i] = result
        print("查询结果：%s" % result)
        if i == 20:
            break
    csdn_so.to_csv('desc.csv', index=False)
