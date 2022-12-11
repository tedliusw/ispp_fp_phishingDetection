import scrapy
import pandas as pd
import os
class TestSpider(scrapy.Spider):
    name = "test"
    ds = pd.read_csv("../dataset/dataset_phishing.csv")
    start_urls = list(ds.iloc[:, 0])[:2]
    # start_urls = [
    #     "http://stackoverflow.com/questions/38233614/download-a-full-page-with-scrapy",
    # ]
    

    def parse(self, response):
        file = "./" + response.url.split("/")[2]
        if not os.path.exists(file) and len(response.body.strip()) != 0:
            os.mkdir(file)
            filename = file + '/html.txt'
            with open(filename, 'wb') as f:
                f.write(response.body)
            urlfile = file + '/info.txt'
            with open(urlfile, 'w') as g:
                g.write(response.url)
