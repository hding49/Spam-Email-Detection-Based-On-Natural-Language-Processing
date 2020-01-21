import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.ensemble import VotingClassifier

data = pd.read_csv('/Users/achilles/Desktop/Digital Analytics/Project/spam_ham_dataset.csv')
data = data.iloc[:, 1:]
data.head()

print('Dataset include {} emails'.format(data.shape[0]))
print('Ham email: {}'.format(data['label_num'].value_counts()[0]))
print('Spam email: {}'.format(data['label_num'].value_counts()[1]))

plt.style.use('seaborn')
plt.figure(figsize=(6, 4), dpi=100)
plt.title('NUmber of ham adn spam email')
data['label'].value_counts().plot(kind='bar')

# text and label_num
new_data = data.iloc[:, 1:]
length = len(new_data)
new_data.head()

new_data['text'] = new_data['text'].str.lower()
new_data.head()

stop_words = set(stopwords.words('english'))
stop_words.add('subject')

def text_process(text):
    tokenizer = RegexpTokenizer('[a-z]+')
    lemmatizer = WordNetLemmatizer()
    token = tokenizer.tokenize(text)
    token = [lemmatizer.lemmatize(w) for w in token if lemmatizer.lemmatize(w) not in stop_words]
    return token

new_data['text'] = new_data['text'].apply(text_process)

new_data.head()

seed = 20190524
X = new_data['text']
y = new_data['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)  # 75% train set and 25% test set

train = pd.concat([X_train, y_train], axis=1)  # train set
test = pd.concat([X_test, y_test], axis=1)  # test set

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

print('Train set has {} emails，Test set has {} emails'.format(train.shape[0], test.shape[0]))

print(train['label_num'].value_counts())
plt.figure(figsize=(6, 4), dpi=100)
plt.title('NUmber of ham adn spam email in train set')
train['label_num'].value_counts().plot(kind='bar')

print(test['label_num'].value_counts())
plt.figure(figsize=(6, 4), dpi=100)
plt.title('NUmber of ham adn spam email')
test['label_num'].value_counts().plot(kind='bar')

print('0 is ham email and 1 is spam email in test set')
ham_train = train[train['label_num'] == 0]  # ham email
spam_train = train[train['label_num'] == 1]  # spam email

ham_train_part = ham_train['text'].sample(10, random_state=seed)  # select 10 ham email in random
spam_train_part = spam_train['text'].sample(10, random_state=seed)  # select 10 spam email in random

part_words = []  # part of word
for text in pd.concat([ham_train_part, spam_train_part]):
    part_words += text

part_words_set = set(part_words)
print('Word table has {} words'.format(len(part_words_set)))

train_part_texts = [' '.join(text) for text in np.concatenate((spam_train_part.values, ham_train_part.values))]
# word into sentence in train set
train_all_texts = [' '.join(text) for text in train['text']]
# word into sentence in test set
test_all_texts = [' '.join(text) for text in test['text']]

cv = CountVectorizer()
part_fit = cv.fit(train_part_texts)
train_all_count = cv.transform(train_all_texts)
test_all_count = cv.transform(test_all_texts)
tfidf = TfidfTransformer()
train_tfidf_matrix = tfidf.fit_transform(train_all_count)
test_tfidf_matrix = tfidf.fit_transform(test_all_count)

print('train set', train_tfidf_matrix.shape)
print('test set', test_tfidf_matrix.shape)

#Naive Bayesian model
mnb = MultinomialNB()
mnb.fit(train_tfidf_matrix, y_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

y_pred_prob = mnb.predict_proba(test_tfidf_matrix)
fpr1, tpr1, thresholds = roc_curve(y_test, y_pred_prob[:, 1])

y_pred = mnb.predict(test_tfidf_matrix)
print('Naive Bayesian_classification_report:')
print(classification_report(y_test, y_pred))
mnb_acc = accuracy_score(y_test, y_pred)
print('Naive bayesian accuracy:' , mnb_acc)

# diagram
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(fpr1, tpr1)
plt.title('Naive Bayesian Model Prediction Accuracy: {:.4f}'.format(mnb_acc))
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(train_tfidf_matrix, y_train)

y_RF_pred_prob = clf.predict_proba(test_tfidf_matrix)
fpr2, tpr2, thresholds = roc_curve(y_test, y_RF_pred_prob[:, 1])

y_RF_pred = clf.predict(test_tfidf_matrix)
print('RF_classification_report:')
print(classification_report(y_test,y_RF_pred))
rf_acc = accuracy_score(y_test, y_RF_pred)
print('Random Forest accuracy:' , rf_acc)

# diagram
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(fpr2, tpr2)
plt.title('Random Forest Model Prediction Accuracy: {:.4f}'.format(rf_acc))
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

#Decision Tree
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(train_tfidf_matrix, y_train)

y_tree_pred_prob = dt.predict_proba(test_tfidf_matrix)
fpr3, tpr3, thresholds = roc_curve(y_test, y_tree_pred_prob[:, 1])

y_tree_pred = dt.predict(test_tfidf_matrix)
print('tree_classification_report:')
print(classification_report(y_test, y_tree_pred))
dt_acc = accuracy_score(y_test, y_tree_pred)
print('Decision tree accuracy:' , dt_acc)

# diagram
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(fpr3, tpr3)
plt.title('Decision tree Model Prediction Accuracy: {:.4f}'.format(dt_acc))
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

#SVM
from sklearn.svm import LinearSVC
svm = LinearSVC()
svm_prob = CalibratedClassifierCV(svm)
svm_prob.fit(train_tfidf_matrix, y_train)

y_svm_pred_prob = svm_prob.predict_proba(test_tfidf_matrix)
fpr4, tpr4, thresholds = roc_curve(y_test, y_svm_pred_prob[:, 1])

y_svm_pred = svm_prob.predict(test_tfidf_matrix)
print('svm_classification_report:')
print(classification_report(y_test,y_svm_pred))
svm_acc = accuracy_score(y_test, y_svm_pred)
print('SVM accuracy:' , svm_acc)

# diagram
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(fpr4, tpr4)
plt.title('SVM Model Prediction Accuracy: {:.4f}'.format(svm_acc))
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()


# soft voting
soft_voting_clf = VotingClassifier(estimators=[('mnb_clf',MultinomialNB()),('rf_clf',RandomForestClassifier(n_estimators=10)),('dt_clf',dt), ('svc_clf',svm_prob)], voting='soft')
soft_voting_clf.fit(train_tfidf_matrix, y_train)

y_soft_pred_prob = soft_voting_clf.predict_proba(test_tfidf_matrix)
fpr5, tpr5, thresholds = roc_curve(y_test, y_soft_pred_prob[:, 1])

y_soft_pred = soft_voting_clf.predict(test_tfidf_matrix)
print('soft voting_classification_report:')
print(classification_report(y_test,y_soft_pred))
soft_voting_acc = accuracy_score(y_test, y_soft_pred)
print('soft voting accuracy:' , soft_voting_acc)

# diagram
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(fpr5, tpr5)
plt.title('soft voting Model Prediction Accuracy: {:.4f}'.format(soft_voting_acc))
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()
