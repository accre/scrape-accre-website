# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from bs4 import BeautifulSoup
import re
import json


class AccreLinksSpider(CrawlSpider):
  name = "accre_links"
  allowed_domains = ["www.accre.vanderbilt.edu"]
  start_urls = ['http://www.accre.vanderbilt.edu/']
  rules = (
      Rule(LinkExtractor(), callback='parse_page', follow=True), 
  )

  re_tag = re.compile(r'\S+')

  @staticmethod
  def get_content(soup):
    return re.sub(r'\s+', ' ', soup.get_text(" "))

  def parse_page(self, response):
    page_title = response.css('html').select("//head//title//text()")\
        .extract_first()
    
    soup = BeautifulSoup(
        response.css('body div.pf-content').extract_first(), 
        'lxml') 
    
    # Filter out the script tags
    [s.extract() for s in soup('script')]

    # Log any section that has an id, i.e. linkable
    for section in soup.find_all(attrs={'id': re.compile(r'\S+')}):
      hash_tag = "#" + section['id']
      yield {
          'url': response.url + hash_tag,
          'title': page_title + " " + hash_tag, 
          'content': self.get_content(section)
          }
      # Remove the content
      section.extract()
   
    # Log whatever is left over
    yield {
        'url': response.url,
        'title': page_title, 
        'content': self.get_content(soup)
        }
    
