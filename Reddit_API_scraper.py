'''
This program utilizes reddit's API to pull some of the most popular and controversial content all time, as well as from this month
'''

import praw
import pandas as pd
import time


### login credentials for using Reddit API
# check of this link for how to set up your own reddit API:
# http://www.storybench.org/how-to-scrape-reddit-with-python/
PERSONAL_USE_SCRIPT_14_CHARS = 'AAAAAAAAAAAAAA'
SECRET_KEY_27_CHARS = 'BBBBBBBBBBBBBBBBBBBBBBBBBBB'
YOUR_APP_NAME = 'insight'
YOUR_REDDIT_USER_NAME = 'CCCC'
YOUR_REDDIT_LOGIN_PASSWORD = 'DDDD'

reddit = praw.Reddit(client_id=PERSONAL_USE_SCRIPT_14_CHARS, \
                     client_secret=SECRET_KEY_27_CHARS, \
                     user_agent=YOUR_APP_NAME, \
                     username=YOUR_REDDIT_USER_NAME, \
                     password=YOUR_REDDIT_LOGIN_PASSWORD)
### double-checking you are logged in
print(reddit.user.me())

### Building CSV for top world news of all time

### selecting the relevant news subreddits
subreddits = ('worldnews', 'news', 'politics')
### pulling the relevant data from these reddits
names = ('post_title', 'rank', 'upvotes', 'upvote_ratio', 'total_number_comments', 'traffic_label', 'URL')

### collecting the most controversial articles from the news subreddits of all time
start_time = time.time()
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    lst_con = []
    rank_con = 0
    for submission in subreddit.controversial('all', limit=1500):
        rank_con +=1
        lst_con.append ([submission.title, rank_con, submission.ups, submission.upvote_ratio, submission.num_comments,
                         1, submission.url])
        if rank_con%100==1:
            print (rank_con)
            print ("My controversial data is at:", time.time() - start_time)
            df_con = pd.DataFrame(lst_con, columns = names)
            df_con.to_csv('./alltime_con_' + str(sub)+ '.csv')

    df_con = pd.DataFrame(lst_con, columns = names)
    df_con.to_csv('./alltime_con_' + str(sub)+ '.csv')
### collecting the most top performing articles from the news subreddits of all time
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    lst_top = []
    rank_top = 0
    for submission in subreddit.top('all', limit=1500):
        rank_top +=1
        lst_top.append([submission.title, rank_top, submission.ups, submission.upvote_ratio, submission.num_comments,
                        0, submission.url])
        if rank_top%100==1:
            print (rank_top)
            print ("My top data is at:", time.time() - start_time)
            df_top = pd.DataFrame(lst_top, columns = names)
            df_top.to_csv('./alltime_top_' + str(sub)+ '.csv')


    df_top = pd.DataFrame(lst_top, columns = names)
    df_top.to_csv('./alltime_top_' + str(sub)+ '.csv')

### collecting the most controversial articles from the news subreddits of the past month
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    lst_con = []
    rank_con = 0
    for submission in subreddit.controversial('month', limit=1500):
        rank_con +=1
        lst_con.append ([submission.title, rank_con, submission.ups, submission.upvote_ratio, submission.num_comments,
                         1, submission.url])
        if rank_con%100==1:
            print (rank_con)
            print ("My controversial data is at:", time.time() - start_time)
            df_con = pd.DataFrame(lst_con, columns = names)
            df_con.to_csv('./month_con_' + str(sub)+ '.csv')

    df_con = pd.DataFrame(lst_con, columns = names)
    df_con.to_csv('./month_con_' + str(sub)+ '.csv')
### collecting the most top performing articles from the news subreddits of the past month
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    lst_top = []
    rank_top = 0
    for submission in subreddit.top('month', limit=1500):
        rank_top +=1
        lst_top.append([submission.title, rank_top, submission.ups, submission.upvote_ratio, submission.num_comments,
                        0, submission.url])
        if rank_top%100==1:
            print (rank_top)
            print ("My top data is at:", time.time() - start_time)
            df_top = pd.DataFrame(lst_top, columns = names)
            df_top.to_csv('./month_top_' + str(sub)+ '.csv')

    df_top = pd.DataFrame(lst_top, columns = names)
    df_top.to_csv('./month_top_' + str(sub)+ '.csv')

