import json
from urllib import request

import pandas

url = 'https://movie.douban.com/j/chart/top_list?type=5&interval_id=100%3A90&action=&start=0&limit=1000'
headers = {
    "Cookie": "ll=\"118282\"; bid=30ymjogOeXY; __utma=30149280.693240395.1699886886.1699886886.1699886886.1; __utmc=30149280; __utmz=30149280.1699886886.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utmt=1; __utmt_douban=1; __utma=223695111.81652095.1699886924.1699886924.1699886924.1; __utmb=223695111.0.10.1699886924; __utmc=223695111; __utmz=223695111.1699886924.1.1.utmcsr=douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1699886924%2C%22https%3A%2F%2Fwww.douban.com%2F%22%5D; _pk_id.100001.4cf6=96f74d7c85486979.1699886924.; _pk_ses.100001.4cf6=1; ap_v=0,6.0; __yadk_uid=so973t5zc1leta8YHS1gyqdgbXDtU5j6; _vwo_uuid_v2=DF550B566F08FB9DF7F4132EEAAB888A5|ecc98bbf59e2f4dd70191fc9fe6c5f39; __utmb=30149280.6.10.1699886886",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}
# request.Request
req = request.Request(url=url, headers=headers)
# urlopen
response = request.urlopen(req)
# response.read()
content = response.read().decode('utf-8')
# transfer to obj
obj = json.loads(content)
# transfer to pd
df = pandas.DataFrame(obj)
# 持久化
df.to_excel(pandas.ExcelWriter("douban_action_movies.xlsx"), engine="xlsxwriter")
