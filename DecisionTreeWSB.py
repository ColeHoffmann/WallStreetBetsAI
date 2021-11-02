import requests
import pandas as pd
import math
import numpy as numpy
import pandas as pd
import reticker
import stockquotes as stonks
import sys
from collections import Counter


class Node:

    def __init__(self, data):
        self.data = data
        self.children = []
        self.choices = []
        self.predict = None

    def addChild(self, childNode):
        self.children.append(childNode)

    def getChildren(self):
        print(self.children)

    def setChoices(self, attributeChoices):
        self.choices.append(attributeChoices)

    def getChoices(self):
        return self.choices

    def setPredict(self, data):
        self.predict = data

    def getPredict(self):
        return self.predict

# returns the entropy value for certain values.


def entropy_formula(e, p):
    entropy = 0
    total = e + p
    if(e > 0 and p > 0):
        entropy = (-((e/total) * math.log2(e/total)) -
                   ((p/total)*(math.log2(p/total))))

    return entropy


def calculate_entropy(decision, column):

    # First, we found out number of possibilities i.e. forcast has 3-> Sunny, Overcst, Rainy
    choices = column.unique()  # Array of choices
    print("CHOICES")
    print(choices)
    sum_entropy = 0

    # Now find out the count for each one of those values, and also, the times in which decision is edible or poisineous...
    for choice in choices:
        # probability
        num_choice_happens = ((column == choice).sum())
        total_events = len(column)
        # we have probability now of event, so lets find out the count of edible and poisonous for that value
        prob = num_choice_happens / total_events

        # will return the amount of times the column is dependent on that choice, and also it is edible. If this works its HUGE.
        e_count = (((column == choice) & (decision == 'e')).sum())
        # of course, same situation for poisnous
        p_count = (((column == choice) & (decision == 'p')).sum())
        sum_entropy = sum_entropy + prob * entropy_formula(e_count, p_count)
    # At this point, we should have Probability of that event occuring, and the times they are yes or no.. we could call a function such that Probabily * Entropy(number of edible, number of poisonous.)
    return(sum_entropy)


# returns a dataframe with select rows.
def select_rows(df, choice, column):
    # this should return a dataframe only in which the values of the column of the main node are equal to the choice we are looking for.
    return (df.loc[df[column] == (choice)])
    # Implementation of the ID3 Decision Tree creator.

# will return to bullish and bearish


def num_e(column):
    num = (column == 'e').sum()
    return num


# will return the amount of poisonous for a column - TODO (change to bullish and bearish)
def num_p(column):
    num = ((column == 'p').sum())
    return num


# Load The Data
def gatherData(start, end):
    data = pd.read_json('data\WSB.json', lines=True)
    dd_posts = data[data['link_flair_css_class'].isin(['dd'])]
    # let's remove the rest to clear up some memory
    del(data)
    # print(dd_posts)
    # For speed of testing, we will move 1000 down to 20 - changing back to higher should change accuracy
    data = dd_posts.iloc[start:end]
    # print(data)
    text_data = (data[['title', 'selftext', 'score', 'num_comments', ]])
    return text_data

# Completed ID3 Decision Tree


# Completed ID3 Decision Tree
def ID3(df):
    # Create root node - will be catergory with lowest entropy.
    entropyVals = []

    for column in df.columns[1:len(df)]:
        entropyVals.append(calculate_entropy(df[0], df[column]))

    # In this example, it will return 4, but it is really the 5th column, -1 for removal of decision column
    rootNode = entropyVals.index(min(entropyVals)) + 1

    # this will find a rootnode who hasent been used yet.
    while(rootNode in used_columns):
        entropyVals[rootNode-1] = 1
        rootNode = entropyVals.index(min(entropyVals)) + 1

    print('\n The current list of entropy values are as followed: \n', entropyVals)
    used_columns.append(rootNode)
    print("The next lowest entropy was at column", rootNode,
          ", which had an entropy value of ", entropyVals[rootNode-1])

    print(used_columns)
    # now we have a rootnode of
    node = Node(rootNode)

    # We want to go through the different chouices of the called attribute, if homogenous, make it a rootnode, if not
    choices_of_root = df[rootNode].unique()

    for choice in choices_of_root:
        # we want to check for homogenous groups
        e_count = (((df[rootNode] == choice) & (df[0] == 'e')).sum())
        p_count = (((df[rootNode] == choice) & (df[0] == 'p')).sum())

        # if e is 0, homogenous, add p leaf
        if(e_count == 0):
            # create the choice node, and the final leaf.
            choiceNode = Node(choice)
            leafNode = Node("p")

            # choice node and add the leaf to  the choice.
            choiceNode.addChild(leafNode)

            # add the child.
            node.addChild(choiceNode)

        elif(p_count == 0 or len(used_columns) == 3):  # if p count is 0, add e leaf - homogenous
            # create the choice node, and the leaf node
            choiceNode = Node(choice)
            leafNode = Node("e")

            # leaf is child to choice, choice is child to column node.
            choiceNode.addChild(leafNode)

            # add the child
            node.addChild(choiceNode)

        else:  # neither e or p, will want a splt.

            # call with respecect to the part of the data frame where
            # function call to

            #print(choice, "Not Homogenus, adding split. ")
            new_df = select_rows(df, choice, rootNode)
            node.addChild(ID3(new_df))

    return node


# Recursive function to return the expected attribute of the mushroom:
def findExpected(tree, row):
    # if this is a leaf node and it is an e or a p, return:
    if((tree.children == [])):
        return (tree.data)

    if(len(tree.children) == 1):
        tree = tree.children[0]
    else:
        # we now want to go through and find the correct path to go to.
        columnNumber = tree.data

        value = row[columnNumber]

        newColumn = 0
        for child in tree.children:
            # This is basiaclly finding the next column/node to go to. This SHOULD always give us a new node, either there is an attribute that works, OR it will find the number of the new column.
            if (isinstance(child.data, int)):
                tree = child
            if value == child.data:
                tree = child
                break

    return (findExpected(tree, row))


def testTree(tree, df_test):
    total = 0
    totalCorrect = 0

    for index, rows in df_test.iterrows():
        total += 1
        real = rows[0]
        expected = findExpected(tree, rows)
        if (expected == real):
            totalCorrect += 1

    accuracy = totalCorrect/total
    print("\nFINAL: ACCURACY OF DECISION TREE: ", accuracy, '\n')


# returns ticker of the stock - if unavailable, returns an empty string.
def findTicker(title):
    extractor = reticker.TickerExtractor()
    tickers = extractor.extract(title)
    if(len(tickers) > 0):
        return tickers[0]
    else:
        return ""

# Gonna save myself the hassle of changing the words, e means bullish and p means bearish.
# Aka, E = stock up, p = stock down.


# Returns a column of e,p, or ' ', whether a stock ticker went up down or doesnt exsist.
def addBearBull(text_data):
    bearBull = []
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
                    bearBull.append('e')
                else:
                    # else, add to the negative train.
                    bearBull.append('p')
            except Exception as inst:
                print(type(inst))    # the exception instance

                print(Exception)  # want to do exception.type
                print("Ticker: ", ticker, " Stock doesnt exist")
                bearBull.append("")
        else:
            bearBull.append("")

    upOrDown = bearBull
    return upOrDown


def changeVals(df):
    # ['title', 'selftext', 'score', 'num_comments'
    title = []
    selftext = []
    score = []
    comments = []

    for index, row in text_data.iterrows():
        # change title
        if(len(row['title']) > 35):
            title.append('l')
        else:
            title.append('s')

        # change selftext
        if(len(row['selftext']) > 200):
            selftext.append('l')
        else:
            selftext.append('s')

        # change score
        if((row['score']) > 1000):
            score.append('h')
        elif (row['score'] > 250):
            score.append('m')
        else:
            score.append('l')

        # change comments
        if((row['num_comments']) > 100):
            comments.append('h')
        elif (row['num_comments'] > 30):
            comments.append('m')
        else:
            comments.append('l')

        print(len(title))
        print(len(selftext))
        print(len(score))
        print(len(comments))

    df.drop(['title', 'selftext', 'score', 'num_comments'],
            axis=1, inplace=True)
    df.insert(1, 4, title, True)
    df.insert(1, 3, selftext, True)
    df.insert(1, 2, score, True,)
    df.insert(1, 1, comments, True)

    return df


# Recursive function to return the expected attribute of the mushroom:
def findExpected(tree, row):
    # if this is a leaf node and it is an e or a p, return:
    if((tree.children == [])):
        return (tree.data)

    if(len(tree.children) == 1):
        tree = tree.children[0]
    else:
        # we now want to go through and find the correct path to go to.
        columnNumber = tree.data

        value = row[columnNumber]

        newColumn = 0
        for child in tree.children:
            # This is basiaclly finding the next column/node to go to. This SHOULD always give us a new node, either there is an attribute that works, OR it will find the number of the new column.
            if (isinstance(child.data, int)):
                tree = child
            if value == child.data:
                tree = child
                break

    return (findExpected(tree, row))


def testTree(tree, df_test):
    total = 0
    totalCorrect = 0

    for index, rows in df_test.iterrows():
        total += 1
        real = rows[0]
        expected = findExpected(tree, rows)
        print("Expected is :", expected)
        print("Real is :", real)
        if (expected == real):
            totalCorrect += 1

    accuracy = totalCorrect/total
    print("\nFINAL: ACCURACY OF DECISION TREE: ", accuracy, '\n')


def gatherTestData(start, end):
    data = pd.read_json('data\WSB.json', lines=True)
    dd_posts = data[data['link_flair_css_class'].isin(['dd'])]
    # let's remove the rest to clear up some memory
    del(data)
    # print(dd_posts)
    # For speed of testing, we will move 1000 down to 20 - changing back to higher should change accuracy
    data = dd_posts.iloc[start:end]
    # print(data)
    test_data = (data[['title', 'selftext', 'score', 'num_comments', ]])
    return test_data


def changeTestVals(df):
    print(df)


print("Sorting the WSB JSON File. There are over 450,000 individual comments, so please be patient. :-)")
# 120-150
text_data = gatherData(0, 400)
print(text_data)

# Right now we dont have a decision column, we need to go through and adjust some of the data. Aka, if a stock price went up, make it bullish, if it went down, make it bearish, if it can not be found, throw it out..
insertArray = addBearBull(text_data)
text_data.insert(0, 0, insertArray, True)
print(text_data)

# drop columns for no ticker data:
index_names = text_data[text_data[0] == ""].index
text_data.drop(index_names, inplace=True)

# alterdata -# This will be the final ID3 data frame
# Global variable for array of columns used.
used_columns = []
print(text_data.columns.values)

ID3_data = changeVals(text_data)

print(ID3_data)

# TIME TO RUN ID3


decision_tree = ID3(ID3_data)
print(decision_tree)
print("Tree Made!")


#Now, testData
#print("Gathering Test Data. Again, Please be patient! :-) ")
#test_data = gatherTestData(0, 30)
# print("test_data")
#insertArray = addBearBull(test_data)
#test_data.insert(0, 0, insertArray, True)
# drop columns for no ticker data:
#index_names = test_data[test_data[0] == ""].index
#test_data.drop(index_names, inplace=True)
# print(test_data.columns.values)
#ID3_test = changeVals(test_data)


print("Creating the test data, please be patient! :-)")
#20 - 50
text_data = gatherData(400, 450)
print(text_data)

# Right now we dont have a decision column, we need to go through and adjust some of the data. Aka, if a stock price went up, make it bullish, if it went down, make it bearish, if it can not be found, throw it out..
insertArray = addBearBull(text_data)
text_data.insert(0, 0, insertArray, True)
print(text_data)

# drop columns for no ticker data:
index_names = text_data[text_data[0] == ""].index
text_data.drop(index_names, inplace=True)

# alterdata -# This will be the final ID3 data frame
# Global variable for array of columns used.
used_columns = []
print(text_data.columns.values)

ID3_data = changeVals(text_data)

print(ID3_data)


testTree(decision_tree, ID3_data)
