from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import pickle
from random import randint 

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'
opt = webdriver.ChromeOptions()
opt.add_argument('--user-agent=%s' % user_agent)
browser = webdriver.Chrome(executable_path="chromedriver.exe")


# 要先登入是因為登入後搜尋結果會比較好，執行cookie.py後可以紀錄登入狀態
url = "https://shopee.tw/search?keyword="
browser.get(url)
cookies = pickle.load(open("cookies.pkl", "rb"))
for cookie in cookies:
    browser.add_cookie(cookie)


keyword_list = ['提提研新極輕絲角鯊烷面膜', '姍拉娜治痘洗面乳']
brand_list = []
for keyword in keyword_list:
    try: 
        # go to search page
        browser.get(url)
        time.sleep(1)

        # search product
        search_bar = browser.find_elements_by_xpath("/html/body/div[1]/div/div[2]/div[2]/div/div[1]/div[1]/div/form/input")[0]
        search_bar.send_keys(keyword)
        search_button = browser.find_elements_by_xpath("/html/body/div[1]/div/div[2]/div[2]/div/div[1]/div[1]/button")[0]
        search_button.click()
        time.sleep(1)

        # proucts layout
        prouduct_href = browser.find_elements_by_xpath('/html/body/div[1]/div/div[3]/div/div/div[2]/div[2]/div[2]/div[6]/a')[0].get_attribute("href")
        browser.get(prouduct_href)
        time.sleep(1)

        # product page
        brand = browser.find_element_by_class_name('kQy1zo').text
        brand_list.append(brand)
    except:
        brand_list.append(None)
    time.sleep(randint(1,3))
    
print(brand_list)