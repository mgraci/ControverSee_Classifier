'''
    Using the Reddit_API_scraper.py, this script does the following things:
     1) Loads in data from Reddit_API_scraper.py
     2) Labels controversial data
     3) Merges all data together for later Newspaper_scraper.py
'''

import pandas as pd
### 1) importing all datasets from Reddit API scraping
df1 = pd.read_csv('alltime_con_worldnews.csv')
df2 = pd.read_csv('alltime_con_news.csv')
df3 = pd.read_csv('alltime_con_politics.csv')
df4 = pd.read_csv('alltime_top_worldnews.csv')
df5 = pd.read_csv('alltime_top_news.csv')
df6 = pd.read_csv('alltime_top_politics.csv')

# Combining them all together
high_traffic_data = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
# dropping duplicates across datasets
high_traffic_data = high_traffic_data.drop_duplicates(['URL'], keep = 'first')

'''
    Controversy label creation explanation:

    Controversial content, by this programs definition is high traffic in nature. Through an optimization process
    and visualizing some distributions, I found that a .75 upvote ratio cutoff was most appropriate,
    but results never wavered that much when going plus/minus .5, which is a nice sanity check. Thus, controversial
    content is labeled as a 2, and popular content is labeled as a 1.
    
    Something is not controversial if one person likes it and one person dislikes it...
    that's just low traffic in nature. Thus, if a news article was not high on reddit (i.e., bottom 400 in the ranking &
    not commented on much ( below 100 comments) then it was low traffic in nature and labeled as a 0.
'''
### 2a) making labels for controversial content
def label_controversy(lst):
    #labeling it as more controversial b/c upvote ratio is below 3/4
    if lst  < .75:
        out = int(2)
    #labeling it as more popular b/c upvote ratio is above 3/4
    else:
        out = int(1)
    return out

con_labels = []
for row in high_traffic_data['upvote_ratio']:
    lst = label_controversy(row)
    #print([a,b])
    con_labels.append(lst)
    high_traffic_data['controversy_label'] = con_labels


# loading in rest of data
df7 = pd.read_csv("month_con_worldnews.csv")
df8 = pd.read_csv("month_con_news.csv")
df9 = pd.read_csv("month_con_politics.csv")
df10 = pd.read_csv("month_top_worldnews.csv")
df11 = pd.read_csv("month_top_news.csv")
df12 = pd.read_csv("month_top_politics.csv")

# Combining low traffic stuff together
low_traffic_data  = pd.concat([df7, df8,df9, df10, df11, df12], ignore_index=True)
# dropping duplicates across datasets
low_traffic_data = high_traffic_data.drop_duplicates(['URL'], keep = 'first')

### 2b) making 0's for controversy class
low_traffic_data['controversy_label'] = low_traffic_data[['rank','total_number_comments']].apply(lambda x: (x[0]>400 and x[1] <100),axis=1).map({True: 0})

'''
    Traffic Label creation explanation:
    
    Need to officially label the low_traffic of the dataset. Through an optimization process
    and visualizing some distributions, I found that a .75 upvote ratio cutoff was good,
    but results never wavered that much when going plus/minus .5, which is a nice sanity check.
'''

### 3) putting it all together
full_df = pd.concat([high_traffic_data, low_traffic_data], ignore_index=False)
full_df.head()
full_df.to_csv('reddit_data.csv')
