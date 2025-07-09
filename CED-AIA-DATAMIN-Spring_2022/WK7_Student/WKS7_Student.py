#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:18:44 2022

@author: skciller
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

#set seed variable for reproducibility
random_seed = 25
np.random.seed(random_seed)

#NAIVE_BAYES spam filtering exercise
df = pd.read_csv("./enron_emails.csv")
df

#Lets Look at the distribution of spam and ham emails
#                       verify using cli
#df.label.value_counts()

#print(df[df["label"]=="ham"].text.iloc[109])

#print(df[df["label"]=="spam"].text.iloc[109])

'''bag of words presentation'''

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)   #fit_transform = to TRAINING DATA
print(X.shape)
'''end ========================bag of words presentation'''


#The CountVectorizer's `fit_transform` method returns a NxM matrix. 
#`N` is the number of documents (sentences) you have in your corpus,
# and `M` is the number of unique words in your corpus. 
#Item `n`x`m` is how many times word `m` appears in document `n`.

#vectorizer.get_feature_names_out()  -cli
#vectorizer.get_feature_names()

#prints vectorized X as an array, rather than a line with mulitiple breaks
#print(X.toarray())

'''output the corpus vectorized, as the header of the X.toarray output using a dictionary, 
    and zip joining function'''
#print(corpus[0])
#dict(zip(vectorizer.get_feature_names(),X.toarray()[0]))

'''output the corpus vectorized, as the header of the X.toarray output using a dictionary, 
    and zip joining function'''
#print(corpus[1])
#dict(zip(vectorizer.get_feature_names(),X.toarray()[0]))


'''Now, if you want to vectorize new data (e.g. test data), 
then you use the .transform function. 
If the vectorizer encounters a word it hasn't seen before, 
it will simply ignore it. '''

#ex:  vectorizer.transform(["This is the coolest document"]).toarray()
#array([[0, 1, 0, 1, 0, 0, 1, 0, 1]])  
#coolest isnt used, so no entry for coolest will be compared

'''
===1.3) Building and Running the Model (Group)

Now that you have all the required tools, build a Naive Bayes Classifier and evaluate it on a train and test set. In this instance, Multinomial Naive Bayes classifier, which is most useful for discrete features that use frequency counts (e.g. a bag of words vector).
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Create training and test splits - 20% split   #test_size=.20   random_state=random_seed
#no clarification of what is x.. using DF instead.
Train, Test = train_test_split(df, test_size=0.20, random_state=random_seed)
#not using x_train, x_test, y_train, y_test as we a using only df, to determine the train, and test data

# Vectorize on your training data using BoW
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Train.text)

print(X.toarray())
''' output : 
[[ 0 10  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]
 ...
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]]

'''
# Fit the classifier below
clf = MultinomialNB()  #references to line 83
clf.fit(X, Train.label_num) #fit to X, Y (which is now Train.label_num)

# Vectorize your test data using transform and then predict the test data
test_vector = vectorizer.transform(Test.text)  #transform to Test_data, and save to test_vector)
preds = clf.predict(test_vector)               #predict to test_vector and save to predictions)
'''
=========================== 
lets test out data for accuracy using a confusion_matrix
 and print out a classification matrix
=========================== 
'''
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Print a confusion matrix using confusion_matrix
confusion_matrix(Test.label_num, preds)
'''
array([[728,   6],
       [ 11, 290]])
'''

# Print a classification report using classification_report
print(classification_report(Test.label_num, preds))

'''
1.4
Technical Note: Log Probabilities:

When using probabilistic methods with large datasets, sometimes you get 
features with extremely small probabilities (e.g. 10^−10).

This becomes a problem, because computers aren't really good at doing operations 
    with numbers at this scale. Therefore, in most systems, operations are done on
    the log of the probabilities.

This makes calculations much more managable (e.g. log(10^−10)=−10). 
    As an added bonus, due to log rules (log(ab) = log(a) + log(b)), 
    all multiplications turn into additions, which are easier for the computer.

Some general rules of thumb: the closer to zero a log prob is, 
the more probabable it is, and each time a log prob decreases by one, 
it's an order of magnitude less probable.
feature_log_probs gives us the log probabilities for each word. 
In notation, each of these are P(word | class)
'''

# Given that a message is ham, how probable is it for the words to show up?
#output of cli command:   clf.feature_log_prob_[0]
'''
array([ -5.75023447,  -5.71398455, -11.28296498, ..., -13.07472445,
       -13.07472445, -13.07472445])

# Given that a message is SPAM, how probable is it for the words to show up?
#output of cli command:   clf.feature_log_prob_[1]

array([ -6.23363088,  -7.07284789,  -9.8691907 , ..., -11.74099287,
       -11.74099287, -11.74099287])

This code will sort all the words by log probability, so that all of the most probable words show up first...
'''
spam_args = np.argsort(clf.feature_log_prob_[1])
#spam_words = np.array(vectorizer.get_feature_names_out())[spam_args]
spam_words = np.array(vectorizer.get_feature_names())[spam_args]
spam_words = np.flip(spam_words)
print(spam_words)


#Is this flipped or is this wrong

ham_args = np.argsort(clf.feature_log_prob_[0])
#ham_words = np.array(vectorizer.get_feature_names_out())[ham_args]
ham_words = np.array(vectorizer.get_feature_names())[ham_args]
#ham_words = np.flip(spam_words) #this line is incorrect
ham_words = np.flip(ham_words) #this is what the line should be
print(ham_words)  

'''
#There is a single line that causes the issue.  See above.  When you make the change, 
#you should see "the, and..." being the most probable words to detect spam and ham.  

#The argsort function sorts the array from smallest to biggest probability.  
#When we want to know the words that most indicate spam or ham we need to use the
# flip function to put the biggest probability at the beginning of the array.  
#We could either use the flip on the indices or the array itself.  
#We chose the array here. 
output of cli command : spam_words[85:100]

array(['investment', 'contact', 'prices', 'computron', '000', '60',
       'online', '15', 'click', 'business', 'mail', 'been', 'best',
       'report', 'like'], dtype='<U24')

output of cli command : ham_words[50:65]

array(['bretz', 'lydecker', 'sladana', 'slagle', 'slideshow', 'slides',
       'fiskrob', 'slice', 'slewis', 'sleeves', 'fishero', 'breathitt',
       'breaux', 'bridgeline', 'lvuittonet', 'firstworld', 'firstname',
       'slc', 'lwood', 'firstam', 'slammed', 'brendyc', 'slakie', 'brent',
       'breedlove'], dtype='<U24')
'''

# Since we're operating on logs, division turns into subtraction
log_odds = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
spam_ham_args = np.argsort(log_odds)
#vectorizer.get_feature_names() SORTS the list... so we need a sorted ars list)
spam_ham_words = np.array(vectorizer.get_feature_names())[spam_ham_args]

spam_ham_words = np.flip(spam_ham_words)

'''
Here's some of the "spammiest" words...
# output of cli:  top_x=200
#spam_ham_words[0:top_x]
array(['td', 'nbsp', 'pills', 'width', 'computron', 'br', 'font', 'href',
       'viagra', 'height', 'xp', 'src', '2004', 'cialis', 'soft', 'meds',
       'paliourg', 'php', 'voip', 'drugs', 'oo', 'valign', 'bgcolor',
       'biz', 'hotlist', 'moopid', 'div', 'photoshop', 'mx', 'img',
       'knle', 'pharmacy', 'gr', 'intel', 'corel', 'prescription', 'iit',
       'demokritos', 'rolex', 'xanax', 'macromedia', 'dealer',
       'uncertainties', 'valium', 'htmlimg', 'darial', '000000',
       '0310041', 'lots', 'projections', 'jebel', 'adobe', 'rnd', 'color',
       'alt', '161', 'colspan', 'pain', 'readers', 'rx', 'canon',
       'export', 'draw', 'fontfont', 'gra', 'speculative', '1226030',
       'gold', 'pro', 'logos', 'wi', 'toshiba', 'china', '1933', 'spam',
       'vicodin', 'itoy', 'viewsonic', 'ooking', '1618', 'cellpadding',
       'weight', 'hewlett', '4176', 'pill', 'robotics', 'soma',
       'resellers', '8834464', '8834454', 'apc', 'intellinet', 'aopen',
       'iomega', 'enquiries', 'customerservice', 'targus', 'packard',
       'tr', 'uae', 'dealers', 'spain', 'nomad', '1934', 'drug', 'muscle',
       'abdv', 'zonedubai', 'eogi', 'aeor', 'doctors', 'inherent',
       'wysak', 'emirates', 'cheap', 'health', 'border', 'illustrator',
       'hottlist', 'oem', 'apple', 'ffffff', 'ce', 'verdana', 'sex',
       'gif', 'resuits', 'graphics', 'mining', 'studio', 'differ',
       'materia', 'predictions', 'arial', 'waste', 'cellspacing', 'yap',
       'male', 'phentermine', 'tirr', 'cf', 'wiil', 'construed', 'otcbb',
       'atleast', 'materially', 'kin', '2005', 'vi', 'anticipates',
       'erections', 'artprice', 'deciding', 'featured', 'prescriptions',
       'sofftwaares', 'ali', 'ur', 'sir', 'discreet', 'gains', 'dose',
       'cia', 'assurance', 'distributorjebel', 'nigeria', 'spur',
       'serial', 'ambien', 'creative', 'align', 'stocks', 'aerofoam',
       'der', 'penis', 'emerson', 'bingo', 'ffffffstrongfont', 'mai',
       'style', 'anxiety', 'brbr', 'prozac', 'undervalued', 'epson',
       'fontbr', 'notebook', 'levitra', 'es', 'iso', 'risks', 'alcohol',
       'xm', 'erection', 'lasts', 'effects', 'vlagra', 'technoiogies',
       '124', 'couid'], dtype='<U24')

Note that words like td, nbsp and br are all HTML tags (for tables, spaces and newlines, respectively. This suggests that SPAM is more likely to have fancy HTML formatting than HAM.

Reverse the list, and now we have the "hammiest" words... (words most indicative of a legitimate email)

np.flip(spam_ham_words)[0:top_x]

array(['enron', 'ect', 'meter', 'hpl', 'daren', 'mmbtu', 'xls', 'pec',
       'sitara', 'hou', 'volumes', 'ena', 'forwarded', 'melissa',
       'tenaska', 'teco', 'nom', '2001', 'pat', 'aimee', 'actuals',
       'noms', 'hsc', 'susan', 'cotten', 'chokshi', 'nomination', 'fyi',
       'pipeline', 'wellhead', 'eastrans', 'clynes', 'hplc', '713',
       'counterparty', 'pefs', 'bob', 'nominations', 'cec', 'gcs',
       'lannou', 'txu', 'farmer', 'hplno', 'rita', 'weissman', 'cc',
       'equistar', 'enronxgate', 'iferc', 'scheduled', 'spreadsheet',
       'wynne', 'allocated', 'entex', 'path', 'buyback', 'fuels', 'hplo',
       'lisa', 'scheduling', 'pops', 'anita', 'calpine', 'gco', 'darren',
       'clem', 'steve', 'aep', 'katy', 'tu', 'flowed', 'follows',
       'sherlyn', 'donna', 'lloyd', 'midcon', 'pm', 'redeliveries',
       'jackie', 'gary', 'vance', 'papayoti', 'meters', 'cornhusker',
       'luong', 'howard', 'pg', 'lsk', 'revision', 'julie', 'utilities',
       '281', 'bryan', 'dfarmer', 'ees', 'reinhardt', 'hplnl', 'cleburne',
       'valero', 'unify', 'outage', 'poorman', 'victor', 'methanol',
       '6353', 'tap', 'baumbach', 'devon', 'lsp', 'lamphier', 'herod',
       'liz', 'schumack', 'enserch', 'employee', '098', 'boas', 'megan',
       'meyers', 'allocation', 'deliveries', 'easttexas', 'ami',
       'enrononline', 'invoice', 'withers', 'taylor', 'robert', 'bellamy',
       'fred', 'gpgfin', 'avila', 'pathed', 'duke', 'spoke', 'mccoy',
       'cernosek', 'oasis', 'carlos', 'kevin', '1266', 'saturday', '853',
       'riley', 'tejas', 'waha', 'katherine', 'kcs', 'graves',
       'logistics', 'revised', 'paso', '345', 'eileen', 'hakemack', 'mm',
       'ponton', 'cdnow', 'hesco', 'cp', 'reliantenergy', 'sandi', 'btu',
       'mckay', 'gomes', 'chad', '0435', 'superty', 'lamadrid', '4179',
       'tisdale', 'neon', 'lauri', 'interconnect', 'aepin', 'neuweiler',
       'herrera', 'attached', 'panenergy', 'acton', 'tess', 'deal',
       'rodriguez', 'mops', 'holmes', 'coastal', 'imbalance', 'stacey',
       'availabilities', 'eol', 'pinion', 'heidi', 'camp', 'brenda',
       'mary', 'origination', 'charlene', 'billed', 'lee'], dtype='<U24')
'''