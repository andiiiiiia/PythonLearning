# 1. 准备信息
## 1.1 officialWebsite
https://www.selenium.dev/zh-cn/documentation/webdriver/getting_started/  

## 1.2 selenium版本
基于4.38.0   
```cmd
pip show selenium
Name: selenium
Version: 4.38.0
```
## 1.3 chrome驱动
自行下载  

# 2. wait 等待 
## 2.1 显示等待 WebDriverWait + expected_conditions
```python
next_page_button = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//a[@class="n "]'))
)
```
# 3. 元素WebElement(BaseWebElement)
## 3.1 元素定位
### 3.1.1 class name
定位class属性与搜索值匹配的元素（不允许使用复合类名）  
### 3.1.2 css selector	
定位 CSS 选择器匹配的元素   
### 3.1.3 id	
定位 id 属性与搜索值匹配的元素   
### 3.1.4 name	
定位 name 属性与搜索值匹配的元素   
### 3.1.5 link text	
定位link text可视文本与搜索值完全匹配的锚元素    
### 3.1.6 partial link text	
定位link text可视文本部分与搜索值部分匹配的锚点元素。如果匹配多个元素，则只选择第一个元素。    
### 3.1.7 tag name	
定位标签名称与搜索值匹配的元素    
### 3.1.8 xpath	
定位与 XPath 表达式匹配的元素   
### 3.1.9 示例
```python
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

driver_path = 'xxx\chromedriver.exe'
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

## 3.2 获取元素信息
### 3.2.1 是否显示
检查连接的元素是否正确显示在网页上    
```python
driver.get("https://www.selenium.dev/selenium/web/inputs.html")

# isDisplayed
is_email_visible = driver.find_element(By.NAME, "email_input").is_displayed()
```
### 3.2.2 是否启用
检查所连接的元素在网页上是启用还是禁用状态。  
```python
is_enabled_button = driver.find_element(By.NAME, "button_input").is_enabled()
```
### 3.2.3 是否被选定
相关的元素是否 已选定，常用于复选框、单选框、输入框和选择元素中。  
```python
is_selected_check = driver.find_element(By.NAME, "checkbox_input").is_selected()
```
### 3.2.4 获取元素标签名
```python
tag_name_inp = driver.find_element(By.NAME, "email_input").tag_name
```
### 3.2.5 位置和大小
元素左上角的X轴位置  
元素左上角的y轴位置  
元素的高度  
元素的宽度  
```python
rect = driver.find_element(By.NAME, "range_input").rect
```
### 3.2.6 获取元素CSS值
```python
css_value = driver.find_element(By.NAME, "color_input").value_of_css_property("font-size")
```
### 3.2.7 文本内容
```python
text = driver.find_element(By.TAG_NAME, "h1").text
```
### 3.2.8 获取特性或属性
```python
email_txt = driver.find_element(By.NAME, "email_input")
value_info = email_txt.get_attribute("value")
```

### 3.2.9 示例
```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

driver_path = 'xxx\chromedriver.exe'
service = Service(executable_path=driver_path)
driver = webdriver.Chrome(service=service)

driver.get('https://www.baidu.com')

input = driver.find_element(By.ID, 'su')
print(input.get_attribute('class'))
print(input.tag_name)

a = driver.find_element(By.LINK_TEXT, '新闻')
print(a.text)
```

## 3.3 元素交互
### 3.3.1 点击 (适用于任何元素)
```python
# Navigate to URL
driver.get("https://www.selenium.dev/selenium/web/inputs.html")

# Click on the checkbox
check_input = driver.find_element(By.NAME, "checkbox_input")
check_input.click()
```
### 3.3.2 发送键位 (仅适用于文本字段和内容可编辑元素)
```python
# Handle the email input field
email_input = driver.find_element(By.NAME, "email_input")
email_input.clear()  # Clear field

email = "admin@localhost.dev"
email_input.send_keys(email)  # Enter text
```
### 3.3.3 清除 (仅适用于文本字段和内容可编辑元素)
```python
email_input.clear()
```
### 3.3.4 示例
```python
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver_path = 'xxxx\chromedriver.exe'
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

# 4. 浏览器交互 WebDriver(BaseWebDriver)
获取浏览器标题
```python
title = driver.title
```
获取浏览器当前 URL
```python
url = driver.current_url
```
## 4.1 导航
### 4.1.1 打开网站
```python
driver.get("https://www.selenium.dev/selenium/web/index.html")
```
### 4.1.2 后退
```python
driver.back()
```
### 4.1.3 前进
```python
driver.forward()
```
### 4.1.4 刷新当前页面
```python
driver.refresh()
```

# 5. headless
## 示例
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# 配置
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--disable-dev-shm-usage')  # 优化内存使用

# 2. chromeDriver路径
chromedriver_path = r'xxx\chromedriver.exe'
service = Service(executable_path=chromedriver_path)

# 3. 初始化浏览器
browser = webdriver.Chrome(service=service, options=chrome_options)

# 示例：打开百度
browser.get('https://www.baidu.com')
print("✅ 当前页面标题：", browser.title)

# 6. 关闭浏览器
browser.quit()
print("✅ 浏览器已关闭")
```
**测试结果**
```text
✅ 当前页面标题： 百度一下，你就知道
✅ 浏览器已关闭
```
