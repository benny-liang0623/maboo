# import pandas as pd

# df = pd.read_csv('other2.csv', encoding="utf-8", index_col="Unnamed: 0")
# df1 = df.iloc[:1453,:]
# df1.to_csv('other2_1.csv', encoding="utf-8")
# df2 = df.iloc[1453:,:]
# df2.to_csv('other2_2.csv', encoding="utf-8")


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


# log in FB first
url = "https://zh-tw.facebook.com/"
browser.get(url) 


username_input = browser.find_elements_by_name('email')[0]
password_input = browser.find_elements_by_name('pass')[0]
username_input.send_keys("qir20458@jeoce.com")  #輸入FB信箱 
password_input.send_keys("shopee123")  #輸入FB密碼 
login_click = browser.find_elements_by_name('login')[0]
login_click.click()
time.sleep(10)

# go to shopee
url = "https://shopee.tw/buyer/login?keyword=&next=https%3A%2F%2Fshopee.tw%2Fsearch%3Fkeyword%3D"
browser.get(url) 
WebDriverWait(browser, 30).until(EC.presence_of_element_located((By.XPATH,'/html/body/div[1]/div/div[2]/div/div/form/div/div[2]/div[5]/div[2]/button[1]')))
fb_button = browser.find_elements_by_xpath("/html/body/div[1]/div/div[2]/div/div/form/div/div[2]/div[5]/div[2]/button[1]")[0]
fb_button.click()
time.sleep(10)



# 要先登入是因為登入後搜尋結果會比較好，執行cookie.py後可以紀錄登入狀態
url = "https://shopee.tw/search?keyword="
browser.get(url)
# cookies = pickle.load(open("cookies.pkl", "rb"))
# for cookie in cookies:
#     browser.add_cookie(cookie)


import pandas as pd
from tqdm import tqdm
import pickle

other = pd.read_csv("other2_1.csv", encoding="utf-8", index_col="Unnamed: 0")

keyword_list = other['name']
brand_list = []
for keyword in tqdm(keyword_list):
    brand=None

    try: 
        # go to search page
        browser.get(url)
        time.sleep(0.5)

        # search product
        search_bar = browser.find_elements_by_class_name("shopee-searchbar-input__input")[0]
        search_bar.send_keys(keyword)
        time.sleep(0.5)
        search_button = browser.find_elements_by_class_name("shopee-searchbar__search-button")[0]
        search_button.click()
        time.sleep(1)

        # proucts layout
        prouduct_href = browser.find_elements_by_xpath('//a[@data-sqe="link"]')[4:7]
        prouduct_href = [href.get_attribute('href') for href in prouduct_href]

        for href in prouduct_href:
            try:
                browser.get(href)
                time.sleep(1)
                brand = browser.find_element_by_class_name('kQy1zo').text
                break
            except:
                pass
    except:
        pass

    brand_list.append(brand)

    with open('brand_list_1.pkl', 'wb') as f:
        pickle.dump(brand_list, f)
    time.sleep(randint(1,2))
    
other["shopee"] = brand_list
other.to_csv("other_shopee_1.csv", encoding="utf-8")
print(brand_list)