# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from bs4 import BeautifulSoup
import re
import json
import nltk

class AccreLinksSpider(CrawlSpider):
  name = "gh_links"
  allowed_domains = ["github.com/accre"]
  start_urls = ['https://github.com/accre/']
  rules = (
      Rule(LinkExtractor(), callback='parse_page', follow=True), 
  )

  def parse_page(self, response):

    md_content = response.css('body article.markdown-body.entry-content').extract_first()
    if md_content is None:
      md_content = response.css('body div.markdown-body').extract_first()

    if md_content is not None:
      soup = BeautifulSoup(md_content, 'lxml') 
      
      # Filter out the script tags
      [s.extract() for s in soup('script')]
     
      yield {'page': response.url,
          'title': response.xpath("//head//title//text()").extract_first(), 
          'content': re.sub(r'\s+', ' ', soup.get_text(" "))}
    
