# 准备信息
## selenium版本
基于4.38.0   
```cmd
pip show selenium
Name: selenium
Version: 4.38.0
```
## chrome驱动
自行下载  

# find_element(self,by=By.ID/NAME/XPATH/CSS_SELECTOR...,value: Optional[str] = None) -> WebElement
```python
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

driver_path = 'D:\Learning\code\pythonLearning\crawler\selenium\chromedriver.exe'
service = Service(executable_path=driver_path)
driver = webdriver.Chrome(service=service)

driver.get('https://www.baidu.com')

time.sleep(2)

# 贺据id采找到对象
chat_submit_button = driver.find_element(By.ID, 'su')
print(chat_submit_button)

# 想据标签属性的属性值来获取对象的
login_button = driver.find_element(By.NAME, 'wd')
print(login_button)

# 想据xpath语句来获取对象
button = driver.find_elements(By.XPATH, '//input[@id="su"]')
print(button)

# 想据标签的召字来获取对象
button = driver.find_elements(By.TAG_NAME, 'input')
print(button)

# 使用的bs4的语法来获取对象
button = driver.find_elements(By.CSS_SELECTOR, '#su')
print(button)

# 使用超链接名称
button = driver.find_element(By.LINK_TEXT, '文库')
print(button)
```

# readElement
```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

driver_path = 'D:\Learning\code\pythonLearning\crawler\selenium\chromedriver.exe'
service = Service(executable_path=driver_path)
driver = webdriver.Chrome(service=service)

driver.get('https://www.baidu.com')

input = driver.find_element(By.ID, 'su')
print(input.get_attribute('class'))
print(input.tag_name)

a = driver.find_element(By.LINK_TEXT, '新闻')
print(a.text)
```

# operateElement
## 源码方法
**执行js脚本**  
```python
class WebDriver(BaseWebDriver) :
  def execute_script(self, script: str, *args):  
```

**元素响应事件**  
```python
class WebElement(BaseWebElement):
  # 点击
  def click(self) -> None:
  
  # def send_keys(self, *value: str) -> None:
  def send_keys(self, *value: str) -> None:
```

**等等**  

## 示例
```python
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver_path = 'D:\Learning\code\pythonLearning\crawler\selenium\chromedriver.exe'
service = Service(executable_path=driver_path)
driver = webdriver.Chrome(service=service)

driver.get('https://www.baidu.com')

# 1.打开百度，输入关键字，点击搜索
input = driver.find_element(By.ID,'chat-textarea')
input.send_keys('AI')

submit = driver.find_element(By.ID,'chat-submit-button')
submit.click()

# 2.等待搜索内容加载
content_left = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, 'content_left'))
)

# 3. 滑动
scroll_bottom = 'document.documentElement.scrollTop=10000'
driver.execute_script(scroll_bottom)

# 4. 点击下一页按钮
next_page_button = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//a[@class="n "]'))
)
next_page_button.click()

# 5. 回到上一页
driver.back()
time.sleep(6)

# 6. 回到下一页
driver.forward()


time.sleep(50)
```
