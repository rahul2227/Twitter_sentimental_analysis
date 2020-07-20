# Twitter Sentiment Analysis

# Project Aimed:
Twitter sentiment analysis allows you to keep track of what's being said about a particular topic or trend on social media, and can help us detect angry people or negative mentions before they turn into a major crisis. The aim of the project is to analyze the sentiment of tweets. Through, that sentiment we are trying to analyze what is the mood of the person while tweeting. We can also check the product reviews by using this sentiment analysis.

# Pre-Requisites:
Basic about Python3
Some of the Python Libraries
1.  numpy           "https://numpy.org/"
2.  pandas          "https://pypi.org/project/pandas/"
3.  seaborn         "https://pypi.org/project/seaborn/"
4.  sklearn         "https://pypi.org/project/sklearn/"
5.  matplotlib      "https://pypi.org/project/matplotlib/"
6.  nltk            "https://pypi.org/project/nltk/"
7.  re              "https://pypi.org/project/regex/"

# Installation:
There are some steps for installation
1.  Install Python3.                        "https://www.python.org/downloads/"
2.  Download and Install Anaconda Toolkit   "https://www.anaconda.com/products/individual"  
3.  Install Spyder.                         
4.  Install all the libraries above mentioned by using "pip install library_name".
5.  Download the project. Run it in your system.


# Dataset:
In the given dataset there are following columns such as tweets and labels. In the dataset, those who positive response are mentioned with a label "1" and those who are not with"0".

Note: The dataset is downloaded from Kaggle: "https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech".

# Algorithm:
As We can us e a classifier in this project looking at the structure of the dataset provided and after testing with two - three classifiers I came to know the best classifier for this particular model would be Naive-Bayes. So the classifier used in this is Naive Bayes, using sklearn the naivebayes model is imported. In this seaborn library that uses matplotlib is used to plot graphs and pandas is used for reading the dataset. 
I am getting 93% accuracy in this project through Naive Bayes.
