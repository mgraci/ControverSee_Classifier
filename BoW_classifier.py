'''
    Using the data from Newspaper_scraper.py, this script does the following things:
    1) Cleans text for vectorization
    2) Vectorizes text for classification
    3) Classifies text using a L2 Logistic Regression
    4) Saves results in a pickle file for prediction
    '''
# importing all related packages
import pandas as pd
import pickle
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer

# importing relevant packages from text cleaning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import string

# load the dataset
# make sure to drop duplicates, as well as fill na's, which will be important for running the classifiers
data = pd.read_csv("clean_reddit_data.csv")
data = data.fillna('', inplace=False)

### 1) text cleaning function
def text_cleaner(mess):
    """
        Takes in a string of text, then performs the following:
        1. Removes all punctuation
        2. Removes all stopwords
        3. Lowercases the words
        4. Lemmatizes the words
        5. Gets rid of non-alphabetical symbols
        6. Returns clean text
        """
    nopunc = [char for char in words if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    #remove stop words
    no_stops = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #remove lower cases
    lower_tokens = [toks.lower() for toks in no_stops]
    
    lemma_tokens = [wordnet_lemmatizer.lemmatize(toks) for toks in lower_tokens]
    
    # Now only return alphabetical stuff ### might be optional, as numbers might be informative
    alphas = [t for t in lemma_tokens if t.isalpha()]
    
    #make it one string
    clean_text = ' '.join(alphas)
    
    return clean_text

### 2) vectorizing text for classification

# enables count vectorizer for fitting text data
vectorizer = CountVectorizer(analyzer=text_cleaner, token_pattern=r'\w{1,}')
# setting X for classifier
vectorized_text = count_vect.fit_transform(data['Text'])

X_dense = vectorized_text.todense()

### 3) training model for classification
# Split-out validation dataset
X = X_dense
y = data.controversy_label
# test size
validation_size = 0.70
# setting a random seed
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

# After using Naive Bayes, KNN, LDA, Random Forest, and SVM, Logistic Regression performed the best (i.e. highest F1 and accuracy scores)
classifier = linear_model.LogisticRegression(penalty='l2')
BoW_classifier = classifier.fit(X_train, Y_train)

### 4) saving vectorizer and trained classifier for predicting new text
three_class_vec_fit = open("three_class_vec_fit.pickle", "wb")
pickle.dump(vectorizer, three_class_vec_fit)
three_class_vec_fit.close()

three_class_BoW = open("three_class_classifier.pickle", "wb")
pickle.dump(BoW_classifier, three_class_BoW)
three_class_BoW.close()

