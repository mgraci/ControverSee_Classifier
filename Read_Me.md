Controver.See

## This folder contains files necessary in a pipeline for classifying the controversial vs popular vs low-traffic news articles on a social media platform (i.e., reddit's top news subreddits). 


### Included in the package:

### Reddit_API_scraper.py: module for pulling data from pre-specified top news subreddits, such as its URL, upvote ratio, rank 

### Data_merger.py: module combining data together and generating labels for classification

### News_scraper.py: module for further scraping news content from the URL's scraped from Reddit_API_scraper.py

### BoW_classifier.py: module for generating features from the news content

## Please see my website, Controver.See.xyz, for a live demonstration of the tool. 

### Contributors: Matt Graci
