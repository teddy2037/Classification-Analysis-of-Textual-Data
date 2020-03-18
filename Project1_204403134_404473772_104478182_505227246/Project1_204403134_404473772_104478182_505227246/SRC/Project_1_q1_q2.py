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

# Question 1
def question1(flag = True):
	newsgroups_train = fetch_20newsgroups(subset='train')
	ax = plt.subplot()
	plt.hist(newsgroups_train['target'], bins=80)
	plt.xlabel('Category Number')
	plt.ylabel('Number of Training Documents')
	plt.title('20 Newsgroups Dataset')
	plt.axis([-1, 20, 0, 650])
	plt.grid(True)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	if flag:
		plt.show()

def build_vectorizer(min_df, print_stop_words = False):
    wnl = nltk.wordnet.WordNetLemmatizer()
    stop_words_skt = text.ENGLISH_STOP_WORDS
    stop_words_en = stopwords.words('english')
    combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))
    
    if print_stop_words:
        print("# of stop_words_skt:\t\t %s" % len(stop_words_skt))
        print("# of stop_words_en:\t\t %s" % len(stop_words_en))
        print("# of punctuation:\t\t %s" % len(punctuation))
        print("# of combined_stopwords:\t %s" % len(combined_stopwords))

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

    return CountVectorizer(min_df=min_df, analyzer=stem_rmv_punc, stop_words='english')

def vectorize_data(categories, min_df=3, binary=True, print_stop_words = False):
    train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = None)
    test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = None)
    
    if binary:
        Bin_Target_Train = [int(i > 3) for i in train_dataset.target]
        Bin_Target_Test = [int(i > 3) for i in test_dataset.target]
    else:
        Bin_Target_Train = train_dataset.target
        Bin_Target_Test = test_dataset.target

    count_vect = build_vectorizer(min_df=min_df, print_stop_words=print_stop_words)

    X_train_counts = count_vect.fit_transform(train_dataset.data)
    X_test_counts = count_vect.transform(test_dataset.data)

    # Report the shape of the TF-IDF matrices of the train and test subsets respectively**
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return X_train_tfidf, X_test_tfidf, Bin_Target_Train, Bin_Target_Test

def question2(print_stop_words = False, shape_of_IDF = True):
    
    np.random.seed(42)
    random.seed(42)
    
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
        
    categories = ['comp.graphics', 'comp.os.ms-windows.misc',
                      'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                      'rec.autos', 'rec.motorcycles',
                      'rec.sport.baseball', 'rec.sport.hockey']
    
    X_train_tfidf, X_test_tfidf, Bin_Target_Train, Bin_Target_Test = vectorize_data(categories, 3, print_stop_words = print_stop_words)
        
    if shape_of_IDF:
        print("X_train_tfidf size: ", X_train_tfidf.shape)
        print("X_test_tfidf  size: ", X_test_tfidf.shape)
        
    return X_train_tfidf, X_test_tfidf, Bin_Target_Train, Bin_Target_Test

if __name__ == "__main__":
    A, B, _, _ = question2()

