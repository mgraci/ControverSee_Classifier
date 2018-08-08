'''
    I used Newspaper 3k for further scraping data from the URL's collected in the reddit post scraping.
    Newspaper 3k does a great job at scraping the relevant info from news articles.
    
    For more info on Newspaper 3k, check out this link:
    https://github.com/codelucas/newspaper
    
    This script primarily uses the URLs combined in data_merger.py to further scrape
    news websites for the content of the articles.
    
'''

import newspaper
from newspaper import Article
import pandas as pd
import time


def news_scraper(df):
    '''
        This function takes in a dataframe, takes the URL for each article, then scraps the news website for
        info on the article, including its text, post title, article title, date of publication and author.
        
        '''
    content = []
    for index, row in df.iterrows():
        url = row['URL']
        news=Article(url)
        try:
            # need to first download the article
            news.download()
            news.parse()
            content.append({'post_title': row['post_title'],'Article_Title':news.title, 'Text': news.text.replace('\n',' '),
                     'Date_of_pub':news.publish_date})
        except:
            print('could not download', news)
    return content

# importing dataset from merging.py
df = pd.read_csv('reddit_data.csv')
#scraping the newspapers
news_content = news_scraper(df)
#building newspaper dataframe to merge with reddit data
news_data = pd.DataFrame.from_dict(news_content)
# merging df with news content
all_reddit_data = pd.merge(df, news_data, how='inner', on=['post_title'])
#dropping duplicates
all_reddit_data = all_reddit_data.drop_duplicates('post_title')
#saving out CSV to use for classification
all_reddit_data.to_csv('clean_reddit_data.csv')
