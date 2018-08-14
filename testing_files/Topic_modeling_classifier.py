'''
    This script does the following things:
    1) loads in the reddit data
    2) cleans & vectorizes the text from the articles
    3) performs topic modeling on text
    4) hypertuning using gridsearch to get the best fitting model of topics for the text
    5) runs a logistic regression classifier using topic probabilities as features to predict
        controversy vs popularity vs low traffic nature of posts
    6) plots a ROC curve to determine the performance of the classifier
'''

# import packages for analyses
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# import packages for visualizations
import scikitplot as skplt
import matplotlib.pyplot as plt


### 1) load the dataset
data = pd.read_csv("clean_reddit_data.csv")
data = data.fillna('', inplace=False)
data = data.drop_duplicates('URL',keep='first')

data.head()

### 2a)  enabling vectorizer
vectorizer = CountVectorizer(analyzer='word',
                             #min_df=10,                    # minimum reqd occurences of a word
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                             )
### 2b) Vectorizing text
data_vectorized = vectorizer.fit_transform(data['Text'])

### 3)building the LDA Model
lda_model = LatentDirichletAllocation(n_topics=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',
                                      random_state=100,          # Random state
                                      batch_size=250,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default
                                      n_jobs = -1,               # Use all available CPUs
                                      )
lda_output = lda_model.fit_transform(data_vectorized)

# Printing out the models performance on first try

# Log Likelihood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))

# See model parameters
print(lda_model.get_params())


### 4) hypertuning the LDA Model

# Define Search Param
search_params = {'n_components': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 'learning_decay': [.5, .7, .9]}

# Intiate LDA
lda = LatentDirichletAllocation()

# Intiate Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Performing the Grid Search
model.fit(data_vectorized)

# if you want to customize things even further
'''
    GridSearchCV(cv=None, error_score='raise',
    estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
    evaluate_every=-1, learning_decay=0.7, learning_method=None,
    learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
    mean_change_tol=0.001, n_components=10, n_jobs=1,
    n_topics=None, perp_tol=0.1, random_state=None,
    topic_word_prior=None, total_samples=1000000.0, verbose=0),
    fit_params=None, iid=True, n_jobs=1,
    param_grid={'n_topics': [2, 4, 6, 8, 10, 12], 'learning_decay': [0.5, 0.7, 0.9]},
    pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
    scoring=None, verbose=0)
    '''
# Selecting out the best model
best_lda_model = model.best_estimator_

# Printing out the best models metrics
# Model Parameters
print("Best Model's Params: ", model.best_params_)
# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)
# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

### 5a) Contructing features (i.e., topic probablities) to use in classification
# Create Document - Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)


###
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

### 5b) Spliting out train and test data for classification
# split the dataset into training and validation datasets
X = df_document_topic[df_document_topic.columns[0:best_lda_model.n_components]]
y = data.controversy_label
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y, test_size = .30, random_state = 7)

### 5c) running classifier
classifier = linear_model.LogisticRegression()
clf = classifier.fit(train_x, train_y)
pred = classifier.predict_proba(valid_x)

### 6) plotting out the classifiers performance
y_true = valid_y
y_probas = pred
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.savefig('Topic_modeling_ROC_curve.png')
plt.show()
