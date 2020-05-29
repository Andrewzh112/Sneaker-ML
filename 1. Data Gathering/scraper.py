from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import re
import requests
import os
import random
bad_links = ['//stockx-assets.imgix.net/svg/icons/list-gray.svg?auto=compress,format',
             '//stockx-assets.imgix.net/emails/the-outsole/youtube-black.png?auto=compress,format',
             '//stockx-assets.imgix.net/media/New-Product-Placeholder-Default.jpg?auto=compress,format']

def html_parse():
    # get url and expand every page
    url = "https://www.goat.com/sneakers"
    opts = Options()
    opts.add_argument("user-agent=Google")
    driver = webdriver.Chrome(options=opts)
    driver.get(url)
    while True:
        time.sleep(10)
        try:
            driver.find_element_by_xpath("//button/span[contains(.,'See More')]").click()
        except:
            break
    time.sleep(30)
    html = driver.page_source.encode('utf-8')
    return html

def image_links(html,hype=True):
    soup = BeautifulSoup(html, 'lxml')
    im_links = soup.find_all('img', src=True)
    pair_links={}
    for link in im_links:
        try:
            if hype:
                pair_links[link['alt'].split("alt=")[-1]]=link["src"].split("src=")[-1]
            else:
                shoelink=link["src"].split("src=")[-1]
                if not re.findall(r'(http://|https://)',shoelink):
                    shoelink='http:'+shoelink
                pair_links[int(random.random()*1000000)]=shoelink
        except:
            continue
    return pair_links

def download_images(img_links,folder):
    rand_num=int(random.random()*10000000000000)
    for name,link in img_links.items():
        if link in bad_links:
            continue
        ext = re.findall(r'(.png|.jpg)',link)
        if ext:
            name=re.sub(r'(\"|\')','',name)
            filename=os.path.join(folder,str(rand_num)+str(name)+ext[0])
        else:
            continue
        r = requests.get(link, allow_redirects=True)
        try:
            open(filename, 'wb').write(r.content)
        except:
            continue

def get_stockx_html(start_url, page_num, amazon=False):
    # get url and expand every page
    url = f"{start_url}{page_num}"
    if amazon:
        url = f'https://www.amazon.com/s?rh=n%3A7141123011%2Cn%3A7147441011%2Cn%3A679255011&page={page_num}&qid=1590093336&ref=lp_679255011_pg_2'
    opts = Options()
    opts.add_argument("user-agent=Google")
    driver = webdriver.Chrome(options=opts)
    driver.get(url)
    time.sleep(20)
    html = driver.execute_script("return document.documentElement.innerHTML;").encode('utf-8')
    driver.close()
    return html

if __name__=='__main__':
    goat_html = html_parse()
    goat_img_links = image_links(goat_html)
    download_images(goat_img_links,'goat')
    
    for page in range(1,26):
        stockx_html = get_stockx_html('https://stockx.com/sneakers?page=',page)
        stockx_img_links = image_links(stockx_html)
        download_images(stockx_img_links,'stockx')

    for page in range(1,35):
        fc_html = get_stockx_html('https://www.flightclub.com/men?page=',page)
        fc_img_links = image_links(fc_html)
        download_images(fc_img_links,'flight_club')
        
    for page_num in range(2,100):
        amazon_html = get_stockx_html('',page_num,True)
        amazon_img_links = image_links(amazon_html,False)
        download_images(amazon_img_links,'non-hype')