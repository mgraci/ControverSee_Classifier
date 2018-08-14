'''
    This script does the following things:
    1) loads in the reddit data
    2) extracts basic word count features of text
    3) runs a logistic regression classifier using text counts as features to predict
    controversy vs popularity vs low traffic nature of posts
    4) plots a ROC curve to determine the performance of the classifier
'''

# importing packages for analyses
from nltk import word_tokenize
import numpy as np
import pandas as pd
import string
from sklearn import model_selection, preprocessing, linear_model
from sklearn.preprocessing import StandardScaler
import textblob

# import packages for visualizations
import scikitplot as skplt
import matplotlib.pyplot as plt

### 1) load the dataset
data = pd.read_csv("clean_reddit_data.csv")
data = data.fillna('', inplace=False)
data = data.drop_duplicates('URL',keep='first')

### 2a) setting up basic count features to use in classification

df = pd.DataFrame()
df['text'] = data.Text
# basic count of character in text
df['char_count'] = df['text'].apply(len)
# basic count of words in text
df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
# the density of words (i.e., characters divided by words)
df['word_density'] = np.round(df['char_count'] / df['word_count'],2)
# count of punctuation
df['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
# count of upper case words
df['upper_case_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

### 2b) setting up Part of Speech count features to use in classification

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

def check_pos_tag(x, flag):
    '''
        This function takes in text and then utilizes the pos_family to count if that part of speech
        is in the sentence, returning the counts. To do so, the function uses Textblob.
        '''
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt
# adding counts for POS tags as more features in the dataframe 
df['noun_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'pron'))

### 3a) Standardizing X
X = df.loc[:, 'char_count':'pron_count']
# scaling X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = data.controversy_label

### 3b) Spliting out train and test data for classification
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size = .30, random_state = 7)

### 3c) enabling classifier
classifier = linear_model.LogisticRegression()
clf = classifier.fit(X_train, y_train)
pred = classifier.predict_proba(X_test)

### 4) plotting out the classifiers performance
y_true = y_test
y_probas = pred
skplt.metrics.plot_roc_curve(y_test, y_probas)
# delete comment this back in if you want to save the figure out
#plt.savefig('POS_modeling_ROC_curve.png')
plt.show()
