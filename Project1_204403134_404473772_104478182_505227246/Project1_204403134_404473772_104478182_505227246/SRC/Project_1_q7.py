from Project_1_q1_q2 import question2 as getData
from Project_1_q3 import LSI_
from Project_1_q4 import fit_predict_and_plot_roc1, plot_confusion_matrix
from Project_1_q5 import print_metrics
from Project_1_q6 import Naive_Bae

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, f1_score, auc
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB
import random
from sklearn.pipeline import Pipeline

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from string import punctuation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD, NMF
# used to cache results
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from functools import partial
from sklearn.externals.joblib import dump

import pandas as pd
from pandas import ExcelWriter

np.random.seed(42)
random.seed(42)

categories = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']

remove = [(),('headers', 'footers')]

train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, 
                                   random_state = None, remove=remove[0])
test_dataset = fetch_20newsgroups(subset = 'test'  , categories = categories, shuffle = True, 
                                   random_state = None, remove=remove[0])
train_dataset_remove = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, 
                                   random_state = None, remove=remove[1])
test_dataset_remove = fetch_20newsgroups(subset = 'test'  , categories = categories, shuffle = True, 
                                   random_state = None, remove=remove[1])

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

wnl = nltk.wordnet.WordNetLemmatizer()

stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))
analyzer = CountVectorizer().build_analyzer()
def lemmatize_sent(list_word):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
        for word, tag in pos_tag(list_word)]

def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in combined_stopwords and not word.isdigit())

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)


# Pipeline options
MIN_DF_OPTIONS = [3,5]
ANALYZER_OPTIONS = ['word', stem_rmv_punc] #with and without lemm
### Dimensionality Reduction: LSI vs NMF
REDUCE_DIM_POSS = [TruncatedSVD(n_components=50, random_state=0), NMF(n_components=50, init='random', random_state=0)]
l1 = ['l1']
l2 = ['l2']
c_10 = 10
c_100 = 100
gamma_10 = 10

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(random_state=0)),
    ('classifier', GaussianNB()),
],
memory=memory
)
param_grid = [{
    'vectorizer__min_df': MIN_DF_OPTIONS,
    'vectorizer__analyzer': ANALYZER_OPTIONS,
    'reduce_dim': REDUCE_DIM_POSS,
    'classifier': [LinearSVC(C=gamma_10)]
             },{
    'vectorizer__min_df': MIN_DF_OPTIONS,
    'vectorizer__analyzer': ANALYZER_OPTIONS,
    'reduce_dim': REDUCE_DIM_POSS,
    'classifier': [LogisticRegression(C=c_10, solver='liblinear')],
    'classifier__penalty':l1
             },{
    'vectorizer__min_df': MIN_DF_OPTIONS,
    'vectorizer__analyzer': ANALYZER_OPTIONS,
    'reduce_dim': REDUCE_DIM_POSS,
    'classifier': [LogisticRegression(C=c_100, solver='liblinear')],
    'classifier__penalty':l2
             },{
    'vectorizer__min_df': MIN_DF_OPTIONS,
    'vectorizer__analyzer': ANALYZER_OPTIONS,
    'reduce_dim': REDUCE_DIM_POSS,
    'classifier': [GaussianNB()]
}]
grid = GridSearchCV(pipeline, cv=5, n_jobs=1, param_grid=param_grid, scoring='accuracy')
Bin_Target_Train = [int(i > 3) for i in train_dataset.target]
Bin_Target_Test = [int(i > 3) for i in test_dataset.target]
q7_results = grid.fit(train_dataset.data, train_dataset.target)

q7_dataframe = pd.DataFrame(grid.cv_results_)

writer_1 = ExcelWriter('Project_1_q7_data_frame_results.xlsx')
q7_dataframe.to_excel(writer_1)
writer_1.save()