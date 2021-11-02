import numpy as numpy
import pandas as pd
import reticker
# import yfinance as yf
import stockquotes as stonks
import sys
from collections import Counter
# https://www.quora.com/Using-Python-whats-the-best-way-to-get-stock-data

# Negative and positive train
Negative_Train = []
Positive_Train = []


# negative and positive test

vocab = Counter()
neg_vocab = Counter()
pos_vocab = Counter()


# returns ticker of the stock - if unavailable, returns an empty string.
def findTicker(title):
    extractor = reticker.TickerExtractor()
    tickers = extractor.extract(title)
    if(len(tickers) > 0):
        return tickers[0]
    else:
        return ""

# CHECKING THE APIS.
# print(stock.current_price)
# history = stock.historical
# first_day = history[-1]
# print(type(first_day))
# print(first_day['adjusted_close'])


# GETTING THE DATA
# add in the files
print("Sorting the WSB JSON File. There are over 450,000 individual comments, so please be patient. :-)")
data = pd.read_json('data\WSB.json', lines=True)

dd_posts = data[data['link_flair_css_class'].isin(['dd'])]
# let's remove the rest to clear up some memory
del(data)
# print(dd_posts)
# For speed of testing, we will move 1000 down to 20 - changing back to higher should change accuracy
data = dd_posts.iloc[0:400]
# print(data)
text_data = (data[['title', 'selftext']])
print(text_data)


# Learn - BAG OF WORDS
# Iterate through the titles ->
for index, row in text_data.iterrows():
    # Find ticker values - want to change this to title probably
    ticker = findTicker(row['title'])
    print("TICKER: ", ticker)
    if(len(ticker) > 0):
        # if good, add the selftext to the good
        try:
            stock = stonks.Stock(ticker)
            currentPrice = stock.current_price
            history = stock.historical
            first_day = history[-1]
            earlier_price = (first_day['adjusted_close'])

            # if price has gone up, add to the positive train
            if(currentPrice > earlier_price):
                Positive_Train.append(row['selftext'])
            else:
                # else, add to the negative train.
                Negative_Train.append(row['selftext'])
        except Exception as inst:
            print(type(inst))    # the exception instance

            print(Exception)  # want to do exception.type
            print("Ticker: ", ticker, " Stock doesnt exist")


# NAIVE BAYES


neg_sample_count = len(Negative_Train)
pos_sample_count = len(Positive_Train)

# Sent in positive train
for sent in Positive_Train:
    word_list = sent.split()
    for word in word_list:
        pos_vocab[word] += 1
        vocab[word] += 1

# Sent in negative train
for sent in Negative_Train:
    word_list = sent.split()
    for word in word_list:
        neg_vocab[word] += 1
        vocab[word] += 1

count_pos_words = 0
for word in pos_vocab:
    count_pos_words += pos_vocab[word]

count_neg_words = 0
for word in neg_vocab:
    count_neg_words += neg_vocab[word]

print(vocab)
print(pos_vocab)
print(neg_vocab)


# ALPHA VALUE
alpha = .2
vocab_size = len(vocab)

# Find negative prob


def find_neg_prob(list_of_words):
    prob_neg = (neg_sample_count / (pos_sample_count+neg_sample_count))

    for word in list_of_words:
        if word in neg_vocab:
            numreator = neg_vocab[word] + alpha
        else:
            numreator = alpha
        denominator = count_neg_words + alpha*vocab_size
        current_word_prob = numreator/denominator
        prob_neg = prob_neg * current_word_prob
    return prob_neg


# Find pos probability
def find_pos_prob(list_of_words):
    prob_pos = (pos_sample_count / (pos_sample_count+neg_sample_count))

    for word in list_of_words:
        if word in neg_vocab:
            numreator = pos_vocab[word] + alpha
        else:
            numreator = alpha
        denominator = count_pos_words + alpha*vocab_size
        current_word_prob = numreator/denominator
        prob_pos = prob_pos * current_word_prob

    return prob_pos


correct_positive = 0
correct_neagative = 0
predictions = []
Positive_Test = []
Negative_Test = []

# Fill the test data
test_data = dd_posts.iloc[400:450]
test_data = (data[['title', 'selftext']])
# Iterate through the titles
for index, row in test_data.iterrows():
    # Find ticker values - want to change this to title probably
    ticker = findTicker(row['title'])
    print("TICKER: ", ticker)
    if(len(ticker) > 0):
        # if good, add the selftext to the good
        try:
            stock = stonks.Stock(ticker)
            currentPrice = stock.current_price
            history = stock.historical
            first_day = history[-1]
            earlier_price = (first_day['adjusted_close'])

            # if price has gone up, add to the positive train
            if(currentPrice > earlier_price):
                Positive_Test.append(row['selftext'])
            else:
                # else, add to the negative train.
                Negative_Test.append(row['selftext'])
        except Exception as inst:
            print(type(inst))    # the exception instance

            print(Exception)  # want to do exception.type
            print("Ticker: ", ticker, " Stock doesnt exist")


for sent in Positive_Test:
    word_list = sent.split()
    pos_prob = find_pos_prob(word_list)
    neg_prob = find_neg_prob(word_list)

    true_label = "Positive"

    if pos_prob > neg_prob:
        pred_label = "Positive"
    else:
        pred_label = "Negative"

    predictions.append((sent, true_label, pred_label))

for sent in Negative_Test:
    word_list = sent.split()
    pos_prob = find_pos_prob(word_list)
    neg_prob = find_neg_prob(word_list)

    true_label = "Negative"

    if pos_prob > neg_prob:
        pred_label = "Positive"
    else:
        pred_label = "Negative"

    predictions.append((sent, true_label, pred_label))


for pred in predictions:
    print(pred)
    print('\n')


correct = 0
total = 0
for pred in predictions:
    (sent, true_label, pred_label) = pred
    if true_label == pred_label:
        correct += 1
    total += 1


accuracy = correct/total

print("accuracy is: ", accuracy)
#     Check
#     If correct, correct += 1
