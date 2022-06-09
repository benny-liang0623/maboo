from bs4 import BeautifulSoup
import requests as rq
import time


def get_brand_name(keyword):

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
    try:
        url = 'https://m.momoshop.com.tw/search.momo?searchKeyword={}&couponSeq=&cpName=&searchType=1&cateLevel=-1&cateCode=-1&ent=k&_imgSH=fourCardStyle'.format(keyword)
        response = rq.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, features="html.parser")
            table =  soup.find_all('li' ,attrs={"class":"goodsItemLi" })[0]
            products = table.find_all("a")
            for product in products:
                try:
                  href = product.get("href")
                  if "http" not in href:
                      in_url  = 'https://m.momoshop.com.tw'+href
                      response = rq.get(in_url, headers=headers)
                      if response.status_code == 200:
                          soup = BeautifulSoup(response.text, features="html.parser")
                          brand_name = soup.find_all('a' ,attrs={"class":"brandNameTxt" })[0]
                          return brand_name.get_text()
                except:
                    pass
                time.sleep(0.1)
        return None
    except:
        return None


import pandas as pd
from tqdm import tqdm
other = pd.read_csv("Brand/BCE/other.csv", encoding="utf-8", index_col="Unnamed: 0")
other["momo"] = None
for i in tqdm(other.index):
    other["momo"][i] = get_brand_name(other["name"][i])
    time.sleep(0.3)

other.to_csv("Brand/BCE/other_momo.csv", encoding="utf-8")