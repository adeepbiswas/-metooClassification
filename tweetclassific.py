import os
from collections import Counter
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

def make_dict():
    direc = "/home/adeep/PycharmProjects/safecity/#metootweets/metooDataset/"
    tweetfiles=os.listdir(direc)
    tweetlists = [direc + tweetlist for tweetlist in tweetfiles]#stores the csv file names

    words = []
    c= len(tweetlists)
    for tweetlist in tweetlists:
        #f= open(tweetlist)
        #blob = f.read()
        with open(tweetlist, 'r') as f:
            blob = f.read()
        words += blob.split(" ")
        print(c)
        c -= 1

    #print(words)

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""
    dictionary = Counter(words)
    del dictionary[""]
    return dictionary.most_common(3000)


def make_dataset(dictionary):
    direc = "/home/adeep/PycharmProjects/safecity/#metootweets/metooDataset/"
    tweetfiles = os.listdir(direc)
    tweetlists = [direc + tweetfiles[0]]

    feature_set = []
    labels = []
    c = len(tweetlists)
    for xyz in tweetlists:
        twefile = xyz
        data = []
        #with open(tweetlist, 'r') as f:
         #   blob = f.read()
        df = pd.read_csv(twefile)
        tweets=df.iloc[:,1]
        for row in range(0,tweets.shape[0]):
            tweet = tweets[row]
            words = tweet.split(" ")
            for entry in dictionary:
                data.append(words.count(entry[0]))
            feature_set.append(data)
            if "rap" in words:
                labels.append(0)
            elif "grop" in words:
                labels.append(1)
            elif "assault" in words:
                labels.append(2)
            elif "harass" in words:
                labels.append(3)
            else:
                labels.append(4)#for other category
    return feature_set, labels

d = make_dict()
features, labels = make_dataset(d)
#print(len(features))
#print(len(labels))

x_train, x_test, y_train, y_test = tts(features, labels, test_size = 0.2)
del d, features, labels
clf = MultinomialNB()
clf.fit(x_train, y_train)

preds = clf.predict(x_test)
print(accuracy_score(y_test, preds))