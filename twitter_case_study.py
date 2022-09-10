import numpy as np
import pandas as pd
import nltk
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from string import punctuation
from nltk import FreqDist
from sklearn import preprocessing
import spacy
import string
from sklearn.feature_selection import chi2,SelectKBest
nlp = spacy.load("en_core_web_sm")
stopwords = spacy.lang.en.stop_words.STOP_WORDS
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

train = pd.read_csv('C:/Users/allaka.satyasri/Downloads/train_E6oV3lV.csv')

test = pd.read_csv('C:/Users/allaka.satyasri/Downloads/test_tweets_anuFYb8.csv')

train.info()
test.info()

nlp.Defaults.stop_words = {"abc","aap","abba","yyc","yyj","yyc","yyyh","yyz","yul","yyg",
                          "yrs","vs","yhz","yqr","yvr","yqf","yqm",
                           "yqr","yfc","yr","yhz","ysj","yo","lol",
                          "yg","etx","ummmm","una","un"}

def preprocessing(tweet):
    x=' '.join(filter(lambda x:x[0] not in ('@'), tweet.split()))
    x=re.sub(r"http\S+", " ", x)
    x=re.sub('[^a-zA-Z]+',' ',x)
    x=x.lower()
    x=x.split()
    x=[i for i in x if i not in punctuation]
    x=[i for i in x if i not in stopwords]  
    x=nlp(' '.join(x))
    x=[words.lemma_ for words in x]
    x=' '.join(x)
    return x

train['filtered']=train['tweet'].apply(preprocessing)

test['filtered']=test['tweet'].apply(preprocessing)

#replacing short words
train['filtered'] = train['filtered'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
test['filtered'] = test['filtered'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


#wordcloud
all_words = ' '.join([text for text in train['filtered']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
wordcloud.to_file('word_Cloud.png')

#wordcloud for positive words
all_words = ' '.join([text for text in train['filtered'][train['label']==0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
wordcloud.to_file('positive_words.png')

#wordcloud for negative words
all_words = ' '.join([text for text in train['filtered'][train['label']==1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
wordcloud.to_file("negative_Words.png")

test_id = test['id']

X_train, X_test, y_train, y_test = train_test_split(train['filtered'], train['label'] , 
                                                    test_size = 0.2,shuffle=True)

test_data = test['filtered']

c_vectorizer = CountVectorizer(stop_words = 'english')

count = c_vectorizer.fit(list(X_train))
xtrain_cv= count.transform(X_train)
xtest_cv = count.transform(X_test) 
test_cv = count.transform(test_data)
feature_name = c_vectorizer.get_feature_names()
xtrain_cv.shape()
test_cv.shape()

vectorizer = TfidfVectorizer(max_features=None, 
            strip_accents='unicode', analyzer='word',
            ngram_range=(1, 3),
            stop_words = 'english',min_df=5, max_df=0.8)
X = vectorizer.fit(list(X_train))
xtrain_tfv=X.transform(X_train)
xtest_tfv=X.transform(X_test)
test_tfv = X.transform(test_data)
feature_name=vectorizer.get_feature_names()
xtrain_tfv.shape
xtest_tfv.shape

class_weight1 = class_weight.compute_class_weight('balanced',
                                                np.unique(train['label']),
                                                train['label'])
cls_wt_dict=pd.Series(list(class_weight1)).to_dict()

grid=SVC(C=1,gamma=1,kernel='rbf',class_weight=cls_wt_dict)
grid.fit(xtrain_tfv, y_train)
y_pred = grid.predict(xtest_tfv)

print(accuracy_score(y_test,y_pred))

confusion_matrix(y_pred,y_test)

y_pred_n = grid.predict(test_tfv)

submission = pd.read_csv('C:/Users/allaka.satyasri/Downloads/sample_submission_gfvA5FD.csv')
submission.head()

submission = pd.DataFrame({'id': test_id, 'label': y_pred_n})

try1 = submission[['id','label']]

try1.set_index('id',inplace = True)

try1.to_csv('SVC_tweet_casestudy.csv')



class_weight = dict({1:1.9, 2:35, 3:180})
rdf = RandomForestClassifier(class_weight=cls_wt_dict, 
            criterion='gini',
            max_depth=8, max_features='auto',
            n_estimators=300)


clf = RandomForestClassifier()
param_grid = {'n_estimators':[200,300,400], 'max_depth':[20,30,None], 'criterion':['gini','entropy']}
grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=5, scoring=make_scorer(f1_score))
cv_res = grid.fit(xtrain_tfv, y_train)
print(cv_res.cv_results_)
print(cv_res.best_params_)

best_clf = cv_res.best_estimator_
best_clf.fit(xtrain_tfv, y_train)
y_pred_t = best_clf.predict(test_tfv)

submission = pd.read_csv('C:/Users/allaka.satyasri/Downloads/sample_submission_gfvA5FD.csv')
submission.head()

submission = pd.DataFrame({'id': test_id, 'label': y_pred_t})

try1 = submission[['id','label']]

try1.set_index('id',inplace = True)

try1.to_csv('random_forest_best_para.csv')

#=========================================================================
#class weights
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

y_1 = np.ones(29720)     # dataset-1
y_2 = np.ones(2242) + 1 # dataset-2
y = np.concatenate([y_1, y_2])
len(y)
# 1197

classes=[1,2]
cw = compute_class_weight('balanced', classes, y)
cw

cls_wt_dict_new=pd.Series(list(cw)).to_dict()
#=========================================================================

