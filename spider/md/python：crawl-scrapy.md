在AIGC（人工智能生成内容）的背景下，爬虫依然是非常重要的工具，虽然AIGC本身具备生成内容的能力，但它仍然依赖于高质量、多样化的训练数据。  

在RAG系统中，爬虫依然是非常重要的组成部分，尤其是在需要实时、个性化、领域特定或私有数据的场景下。
1、**构建知识库**：RAG依赖于一个高质量、结构化或半结构化的外部知识库。这个知识库可以是：企业内部文档、互联网公开数据、行业数据库、用户行为日志。而这些数据的获取，往往需要通过爬虫来完成。  
在企业内部，RAG系统可能需要基于内部文档、邮件、会议记录、产品手册等私有数据进行回答。这些数据无法从互联网获取，但可以通过**内部爬虫**或数据采集工具进行整理和索引。  

2、**个性化与定制化**：在企业或行业应用中，RAG系统可能需要基于特定领域的数据进行回答。例如：医疗领域的RAG系统需要抓取医学文献、金融领域的RAG系统需要抓取财经新闻、法律领域的RAG系统需要抓取法律条文。这些数据通常无法通过通用模型直接获取，必须通过爬虫来定制采集。  

# official website
scrapy 默认只能爬取静态页面  

https://docs.scrapy.net.cn/en/latest/intro/overview.html  

# 技术原理
## 架构
见官方文档  

# 使用
## 快速使用：
### 创建project
进入你要创建项目的父目录下，Terminal终端执行如下命令，创建脚手架  
```python
scrapy startproject [projectName]
```
eg:   
```python
scrapy startproject scrapy_baidu  
```

### 在生成的spider目录下创建爬虫文件
```python
scrapy genspider [fileName] [网页]
```
eg:   
```python
scrapy genspider baidu www.baidu.com
```

### 运行爬虫代码
Name就是生成的爬虫文件中的name值  
```python
scrapy crawl [Name]
```
eg:  
```python
scrapy crawl baidu
```
### 案例1：crawl quotes
**创建项目**
```cmd
scrapy startproject quotes
```
**生成文件**
```cmd
scrapy genspider quote https://quotes.toscrape.com/page/1/
```
**自定义spider代码**
```python
import scrapy


class QuoteSpider(scrapy.Spider):
    name = "quote"
    allowed_domains = ["quotes.toscrape.com"]
    start_urls = ["https://quotes.toscrape.com/page/1/"]

    def parse(self, response):
        quotes = response.xpath('//div[@class="col-md-8"]//span[@class="text"]/text()')
        print(quotes)
        for quote in quotes:
            print(quote.extract())
```
**运行与测试结果**  
略   

## items封装 & pipeline下载
### 案例1：quotes下载并写入json文件
在**初始化项目**的案例中，我们只是控制台打印了爬取内容，更多的情况是，我们需要将内容写入到本地文件中，然后用pandas进行数据清洗过滤等处理  

**settings.py**  
找到ITEM_PIPELINES,并取消注释，这一步就是做配置，打开pipeline管道
```python
ITEM_PIPELINES = {
   "quotes.pipelines.QuotesPipeline": 300,
}
```

**items.py**  
```python
import scrapy


class QuotesItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    quote_text = scrapy.Field()
    quote_author = scrapy.Field()
```

**spiders目录下的quote.py**  
```python
import scrapy

from quotes.items import QuotesItem


class QuoteSpider(scrapy.Spider):
    name = "quote"
    allowed_domains = ["quotes.toscrape.com"]
    start_urls = ["https://quotes.toscrape.com/page/1/"]

    def parse(self, response):
        quote_selectors = response.xpath('//div[@class="col-md-8"]/div[@class="quote"]')
        print('--------------------')
        # print(quote_selectors[0])
        # print(quote_selectors[0].extract())
        for quote_selector in quote_selectors:
            quote_text = quote_selector.xpath('.//span[@class="text"]/text()').extract_first()
            quote_author = quote_selector.xpath('.//small[@class="author"]/text()').extract_first()
            quote = QuotesItem(quote_text = quote_text,quote_author = quote_author)
            yield quote
```

**pipelines.py**
```python
import json

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class QuotesPipeline:
    def open_spider(self,spider):
        print('open_spider')
        self.fp = open('quote.json','w',encoding='utf-8')

    def process_item(self, item, spider):
        data = json.dumps(dict(item), ensure_ascii=False)+ '\n'
        self.fp.write(data)
        return item

    def close_spider(self,spider):
        print("close_spider")
        self.fp.close()
```

**测试结果**
**quote.json**
```json
{"quote_text": "“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”", "quote_author": "Albert Einstein"}
{"quote_text": "“It is our choices, Harry, that show what we truly are, far more than our abilities.”", "quote_author": "J.K. Rowling"}
{"quote_text": "“There are only two ways to live your life. One is as though nothing is a miracle. The other is as though everything is a miracle.”", "quote_author": "Albert Einstein"}
{"quote_text": "“The person, be it gentleman or lady, who has not pleasure in a good novel, must be intolerably stupid.”", "quote_author": "Jane Austen"}
{"quote_text": "“Imperfection is beauty, madness is genius and it's better to be absolutely ridiculous than absolutely boring.”", "quote_author": "Marilyn Monroe"}
{"quote_text": "“Try not to become a man of success. Rather become a man of value.”", "quote_author": "Albert Einstein"}
{"quote_text": "“It is better to be hated for what you are than to be loved for what you are not.”", "quote_author": "André Gide"}
{"quote_text": "“I have not failed. I've just found 10,000 ways that won't work.”", "quote_author": "Thomas A. Edison"}
{"quote_text": "“A woman is like a tea bag; you never know how strong it is until it's in hot water.”", "quote_author": "Eleanor Roosevelt"}
{"quote_text": "“A day without sunshine is like, you know, night.”", "quote_author": "Steve Martin"}
```

## pipeline：多管道下载
### 案例1：当当下载图片和保存书籍信息
**items.py**  
```python
# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrapyDangdangItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    book_img_src = scrapy.Field()
    book_name = scrapy.Field()
    book_price = scrapy.Field()

```

**spiders目录下dangdang.py**  
```python
import scrapy

from scrapy_dangdang.items import ScrapyDangdangItem


class DangdangSpider(scrapy.Spider):
    name = "dangdang"
    allowed_domains = ["category.dangdang.com"]
    start_urls = ["https://category.dangdang.com/cp01.03.32.00.00.00.html"]

    def parse(self, response):
        book_selectors = response.xpath('//ul[@id="component_59"]/li')
        for book_selector in book_selectors:
            book_img_src = book_selector.xpath('.//img/@data-original').extract_first() or book_selector.xpath('.//img/@src').extract_first()
            book_name = book_selector.xpath('.//img/@alt').extract_first()
            book_price = book_selector.xpath('.//p[@class="price"]/span[@class="search_now_price"]/text()').extract_first()
            item = ScrapyDangdangItem(book_img_src=book_img_src, book_name=book_name, book_price=book_price)

            yield item
```

**pipelines.py**  
```python
# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import json
import urllib.request

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class ScrapyDangdangPipeline:
    def open_spider(self, spider):
        print('open_spider')
        self.fp = open('book.json', 'w', encoding='utf-8')

    def process_item(self, item, spider):
        data = json.dumps(dict(item), ensure_ascii=False) + '\n'
        self.fp.write(data)
        return item

    def close_spider(self, spider):
        print("close_spider")
        self.fp.close()


class ScrapyDangdangDownloadImgPipeline:
    def process_item(self, item, spider):
        url = 'http:' + item.get('book_img_src')
        filename = item.get('book_name').strip().replace(":", "_").replace(" ", "_")
        filename = './books/' + filename + '.jpg'
        urllib.request.urlretrieve(url=url, filename=filename)
        return item

```

**settings.py**
```python

BOT_NAME = "scrapy_dangdang"

SPIDER_MODULES = ["scrapy_dangdang.spiders"]
NEWSPIDER_MODULE = "scrapy_dangdang.spiders"




# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
   "scrapy_dangdang.pipelines.ScrapyDangdangPipeline": 300,
   "scrapy_dangdang.pipelines.ScrapyDangdangDownloadImgPipeline":301,
}

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

from fake_useragent import UserAgent
USER_AGENT = UserAgent().chrome
```
**测试结果**

略  


## spider：分页下载  
### 方式1：页面url有规律变化  
**spider下的dangdang.py**   
主要使用如下方法：**yield scrapy.Request(url=url, callback=self.parse)**   
其他文件同上  
```python
import scrapy

from scrapy_dangdang.items import ScrapyDangdangItem


class DangdangSpider(scrapy.Spider):
    name = "dangdang"
    allowed_domains = ["category.dangdang.com"]
    start_urls = ["https://category.dangdang.com/cp01.03.32.00.00.00.html"]
    base_url = 'https://category.dangdang.com/pg'
    page = 1

    def parse(self, response):
        book_selectors = response.xpath('//ul[@id="component_59"]/li')
        for book_selector in book_selectors:
            book_img_src = book_selector.xpath('.//img/@data-original').extract_first() or book_selector.xpath(
                './/img/@src').extract_first()
            book_name = book_selector.xpath('.//img/@alt').extract_first()
            book_price = book_selector.xpath(
                './/p[@class="price"]/span[@class="search_now_price"]/text()').extract_first()
            item = ScrapyDangdangItem(book_img_src=book_img_src, book_name=book_name, book_price=book_price)
            yield item
        if self.page < 2:
            self.page = self.page + 1
            url = self.base_url + str(self.page) + '-cp01.03.32.00.00.00.html'
            yield scrapy.Request(url=url, callback=self.parse)
```

## spider：多层级页面数据爬取  
### 案例1：读书网  
爬取读书网**首页**的图书名称和图书**详情页**的图书内容简介   

**middlewares.py**   
新增如下中间件   
```python
class RandomUserAgentMiddleware:
    def __init__(self):
        self.user_agent = UserAgent()

    def process_request(self, request, spider):
        # 为每个请求随机设置 User-Agent
        request.headers['User-Agent'] = self.user_agent.random
        print(request.headers['User-Agent'])
```

**pipelines.py**  
```python
import json

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class ScrapyDushuPipeline:
    def open_spider(self, spider):
        print('open_spider')
        self.fp = open('dushu_book.json', 'w', encoding='utf-8')

    def process_item(self, item, spider):
        data = json.dumps(dict(item), ensure_ascii=False) + '\n'
        self.fp.write(data)
        return item

    def close_spider(self, spider):
        print("close_spider")
        self.fp.close()
```

**settings.py**  
打开下载中间件  
```python
DOWNLOADER_MIDDLEWARES = {
   # "scrapy_dushu.middlewares.ScrapyDushuDownloaderMiddleware": 543,
   "scrapy_dushu.middlewares.RandomUserAgentMiddleware":543
}
```
关闭rotbot协议   
```python
# ROBOTSTXT_OBEY = True
```

打开itemPipeLine  
```python
ITEM_PIPELINES = {
   "scrapy_dushu.pipelines.ScrapyDushuPipeline": 300,
}
```

**items.py**  
```python
import scrapy

class ScrapyDushuItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    book_name = scrapy.Field()
    book_summary = scrapy.Field()
```

**spiders\dushu.py**  
```python
import scrapy

from scrapy_dushu.items import ScrapyDushuItem

class DushuSpider(scrapy.Spider):
    name = "dushu"
    allowed_domains = ["www.dushu.com"]
    start_urls = ["https://www.dushu.com/book/1188.html"]

    def parse(self, response):
        selectors = response.xpath('//div[@class="bookslist"]//div[@class="book-info"]')
        for selector in selectors:
            book_name = selector.xpath('./h3/a/text()').extract_first()
            # print(attr1)
            book_detail_url = 'https://www.dushu.com' + selector.xpath('./h3/a/@href').extract_first()
            # print(attr2)
            yield scrapy.Request(url=book_detail_url, callback=self.parse_next,meta={"book_name":book_name})

    def parse_next(self, response):
        book_summary = response.xpath('//div[@class="book-summary"][1]//div[@class="text txtsummary"]/text()').extract_first()
        # print(book_summary)
        item = ScrapyDushuItem(book_name = response.meta["book_name"], book_summary=book_summary)
        yield item
```

**测试结果**  
略  

## CrawlSpider：规则匹配爬取
### 案例1：读书网分页爬取
**创建项目**  
```cmd
\scrapy>  scrapy startproject scrapy_read_book

\scrapy\scrapy_read_book\scrapy_read_book\spiders>  scrapy genspider -t crawl read_book www.dushu.com
```

**spiders\read_book.py**   

其他的items和settings和pipelines同上  

```python
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from scrapy_read_book.items import  ScrapyReadBookItem

class ReadBookSpider(CrawlSpider):
    name = "read_book"
    allowed_domains = ["www.dushu.com"]
    start_urls = ["https://www.dushu.com/book/1188_1.html"]

    rules = (
        Rule(
            LinkExtractor(allow=r'/book/1188_\d+\.html'),
            callback="parse_item",
            follow=False
        ),
    )

    def parse_item(self, response):
        selectors = response.xpath('//div[@class="bookslist"]//div[@class="book-info"]')
        for selector in selectors:
            book_name = selector.xpath('.//img/@alt').extract_first()
            book = ScrapyReadBookItem(book_name=book_name)
            yield book
```
