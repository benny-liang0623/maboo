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


import pandas as pd
from tqdm import tqdm
other = pd.read_csv("Brand/BCE/other.csv", encoding="utf-8", index_col="Unnamed: 0")

keyword_list = ['韓國abib口香糖面膜27ml積雪草']
brand_list = []
for keyword in tqdm(keyword_list):
    try: 
        # go to search page
        browser.get(url)
        time.sleep(1)

        # search product
        search_bar = browser.find_elements_by_class_name("shopee-searchbar-input__input")[0]
        search_bar.send_keys(keyword)
        time.sleep(1)
        search_button = browser.find_elements_by_class_name("shopee-searchbar__search-button")[0]
        search_button.click()
        time.sleep(1)

        # proucts layout
        prouduct_href = browser.find_elements_by_xpath('//a[@data-sqe="link"]')[4]
        prouduct_href.click()
        time.sleep(1)

        # product page
        brand = browser.find_element_by_class_name('kQy1zo').text
        brand_list.append(brand)
    except:
        brand_list.append(None)
    time.sleep(randint(1,2))
    
# other["shopee"] = brand_list
# other.to_csv("Brand/BCE/other_shopee.csv", encoding="utf-8")
print(brand_list)