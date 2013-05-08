# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Hack Night Workshop
# ## May 7th 2013
# ## by Rob Zinkov (rob@zinkov.com)

# <headingcell level=1>

# Brief introduction to ipython notebook

# <markdowncell>

# First let's make sure we have plotting enabled

# <codecell>

%pylab inline

# <codecell>

import matplotlib.pylab as plt
import numpy as np

x = np.linspace(-10,10,100)
plt.plot(x, np.sin(0.75*x))
plt.show()

# <markdowncell>

# We also have the option to run commands from here

# <codecell>

!!ls

# <markdowncell>

# The cell output is interactive and live

# <codecell>

from time import sleep
import sys

for i in range(21):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
    sys.stdout.flush()
    sleep(0.25)

# <markdowncell>

# We also have the option of calling other languages

# <codecell>

%load_ext rmagic

# <codecell>

%R X=c(1,4,5,7); print(sd(X)); mean(X)

# <codecell>

%R data(mtcars); plot(mtcars$mpg, mtcars$hp)

# <codecell>

%%R
library(ggplot2)
p <- ggplot(mtcars, aes(x=mpg, y=hp)) + geom_point()
print(p);

# <headingcell level=1>

# Loading the data

# <markdowncell>

# For convience we will load the data using the pandas package

# <codecell>

import pandas as pd

# <markdowncell>

# We will use this package to import our data

# <codecell>

data = pd.read_csv("fb_attendees_cleaned.csv")
data

# <markdowncell>

# Pandas can be thought of like a data frame for python
# <br />
# It has lots of nice features we won't go into right now

# <codecell>

print data.Description[52]
print
print data.Category[52]

# <headingcell level=1>

# Transforming our Data using CountVectorizer

# <markdowncell>

# We start by

# <codecell>

import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer

# <codecell>

venue_vec = CountVectorizer(stop_words="english")
desc_vec = CountVectorizer(stop_words="english")

# <codecell>

venue_data = venue_vec.fit_transform(data.Venue)
desc_data = desc_vec.fit_transform(data.Description)

X = sparse.hstack((venue_data, desc_data))

# <codecell>

print X.shape

# <markdowncell>

# Unfortunately, we need to use less columns, so let's drop rare words

# <codecell>

venue_vec = CountVectorizer(stop_words="english", min_df=0.001)
desc_vec = CountVectorizer(stop_words="english", min_df=0.001)

venue_data = venue_vec.fit_transform(data.Venue)
desc_data = desc_vec.fit_transform(data.Description)

X = sparse.hstack((venue_data, desc_data))
print X.shape

# <codecell>

print data.Venue[3]
venue_vec.inverse_transform(venue_data)[:10]

# <codecell>

CountVectorizer?

# <headingcell level=1>

# Label Encoding

# <markdowncell>

# We need to encode the strings for our categories as numbers

# <codecell>

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(data.Category)

# <headingcell level=1>

# Creating a Classifier

# <markdowncell>

# We will create a stochastic gradient classifier to solve this

# <codecell>

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(alpha=0.001, n_iter=2)

# <markdowncell>

# Fit the data

# <codecell>

clf = clf.fit(X,y)

# <headingcell level=2>

# Predicting label

# <codecell>

X_pred = sparse.hstack((venue_vec.transform([data.Venue[12]]),
                        desc_vec.transform([data.Description[12]])))
print le.inverse_transform(clf.predict(X_pred))

# <headingcell level=1>

# Testing the Classifier

# <codecell>

from sklearn.metrics import classification_report

# <markdowncell>

# Let's make a test set

# <codecell>

venue_data_test = venue_vec.transform(data.Venue[:1000])
desc_data_test = desc_vec.transform(data.Description[:1000])

X_test = sparse.hstack((venue_data_test, desc_data_test))
y_test = le.transform(data.Category[:1000])

# <codecell>

y_pred = clf.predict(X_test)
print classification_report(y_test, y_pred, target_names=list(le.classes_))

# <headingcell level=1>

# Making better features for our Classifier

# <markdowncell>

# The trouble with our features is they don't overweigh words that occur in many documents
# 
# Stripping common words (stopwords) helps, but there are per-corpus words that can still
# <br>
# pollute the results
# 
# Let's use a Vectorizer that reviews some of the words

# <codecell>

from sklearn.feature_extraction.text import TfidfVectorizer

# <codecell>

venue_vec = TfidfVectorizer(stop_words="english", min_df=0.001)
desc_vec = TfidfVectorizer(stop_words="english", min_df=0.001)

# <codecell>

venue_data = venue_vec.fit_transform(data.Venue)
desc_data = desc_vec.fit_transform(data.Description)

X = sparse.hstack((venue_data, desc_data))

# <headingcell level=1>

# Trying different classifiers

# <markdowncell>

# Let's try another classifier

# <codecell>

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

# <headingcell level=2>

# Predicting label with probability

# <codecell>

clf = clf.fit(X,y)
print clf.predict_proba(X_pred)
print le.classes_

# <headingcell level=1>

# Recommendation Challenge

# <markdowncell>

# ## Data Details
# ### The full data has the following columns
# * Title
# * Description
# * Category
# * Location
# * Timezone
# * Number of RSVPs
# * Start Time
# * End Time
# 
# __Not all fields are guaranteed to exist for every venue__

# <markdowncell>

# ## Describe Evaluation Criteria
# 
# There are many categories and you won't be able to get all correct
# 
# 
# Hence we will use average [F1 score](http://en.wikipedia.org/wiki/F1_score) when measuring accuracy of a prediction
# 
# This means we will take the weighted average of all the precision 
# <br />
# and recall scores for each category, and use these to create 
# <br />
# a f1-score. The higher the score the better

# <markdowncell>

# ## Contest Details
# 
# * Teams can be no larger than four people
# * Judging will occur next week May 14th, 2013
# * There will be a first and second place for best accuracy
# * A prize will be made for unique solutions

# <headingcell level=1>

# Further Details

# <rawcell>

# I am available for tutoring and consultation
# 
# rob@zinkov.com

