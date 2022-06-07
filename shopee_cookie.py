from selenium.webdriver.support import expected_conditions as EC  #pip install selenium
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import pickle


user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'
opt = webdriver.ChromeOptions()
opt.add_argument('--user-agent=%s' % user_agent)
browser = webdriver.Chrome(executable_path="chromedriver.exe")


# log in FB first
url = "https://zh-tw.facebook.com/"
browser.get(url) 


username_input = browser.find_elements_by_name('email')[0]
password_input = browser.find_elements_by_name('pass')[0]
username_input.send_keys("")  #輸入FB信箱
password_input.send_keys("")  #輸入FB密碼
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


pickle.dump(browser.get_cookies(), open("cookies.pkl","wb"))
