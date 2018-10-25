import os
import math
import enchant
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob as tb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def make_dict():
    direc = "/home/adeep/PycharmProjects/safecity/metooTweets/metooDataset/"
    tweetfiles=os.listdir(direc)
    tweetlists = [direc + tweetlist for tweetlist in tweetfiles]#stores the csv file names

    word1 = []
    allwords = []
    bloblist = []
    dict = []
    flag = 0
    d = enchant.Dict("en_US")
    c= len(tweetlists)
    i=0
    cf = pd.read_csv("/home/adeep/PycharmProjects/safecity/metooTweets/metooDataset/MeTooCampaign.csv")
    category = cf.iloc[:, 5]
    for tweetlist in tweetlists:
        #f= open(tweetlist)
        #blob = f.read()
        df = pd.read_csv(tweetlist)
        tweets = df.iloc[:, 1]

        with open(tweetlist, 'r') as f:
            blob = f.read()
        for row in range(0, tweets.shape[0]):
            tweet = tweets[row]
            tweet = tweet.lower()
            word1 = word1 + tweet.split(" ")
        #words += blob.split(" ")
        for i in range(len(word1)):
            if not word1[i].isalpha() or len(word1[i]) <= 3 or d.check(word1[i]) == False:
                word1[i] = ""
        if flag == 1:
            word1.remove("")
        allwords = allwords + word1
        stop = stopwords.words('english')
        for x in range(len(word1)):
            if word1[x] in stop or len(word1[x]) <= 2:
                word1[x] = ""
        para = ' '.join(word1)
        para = para.lower()
        para = para.replace('[^\w\s]','')

        bloblist.append(tb(para))
        word1.clear()
        i = i+1
    for i, blob in enumerate(bloblist):
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:10]:
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
            dict.append(word)
        print(c)
        c -= 1

    print(dict)
    dictionary = Counter(word1)
    del dictionary[""]
    print(dictionary)
    #return dictionary.most_common(200), category
    return dict, category


def make_dataset(dictionary, category):
    feature_set = []
    labels = []

    twefile = "/home/adeep/PycharmProjects/safecity/metooTweets/metooDataset/MeTooCampaign.csv"
    data = []
    #with open(tweetlist, 'r') as f:
    #   blob = f.read()
    df = pd.read_csv(twefile)
    tweets=df.iloc[:,1]
    for row in range(0,tweets.shape[0]):
        data.clear()
        tweet = tweets[row]
        words = tweet.split(" ")
        for entry in dictionary:
            data.append(words.count(entry[0]))
        #creating feature vectors with 200 columns
        feature_set.append(data)

        if "rap" in words:
            labels.append(0)
        elif "grop" in words:
            labels.append(1)
        elif "assault" in words:
            labels.append(2)
        elif "harass" in words:
            labels.append(3)
        elif "touch" in words:
            labels.append(3)
        elif "comment" in words:
            labels.append(3)
        else:
            labels.append(4)#for other category
    return feature_set, category

def train(features, labels, d):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size = 0.2, random_state=1)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clf = DecisionTreeClassifier() #RandomForestRegressor(n_estimators=20, random_state=0) #KNeighborsClassifier(n_neighbors=30)  #MultinomialNB()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    print(accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    #adding new data to check the prediction
    feature1 = []
    inp = input(">")
    #creating the feature vector of this input
    for word in d:
        feature1.append(inp.count(word[0]))
    res = clf.predict([feature1])
    print(["Rape","Groping","Assault","Harassment","Inappropriate Touching","Unwanted Comments","Awareness/Support"][res[0]])

d,c = make_dict()
features, labels = make_dataset(d, c)
train(features, labels, d)
print(len(features))
print(len(labels))
