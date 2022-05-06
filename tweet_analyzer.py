import time

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import statistics

from nltk.util import everygrams
from nltk import collections
from os.path import exists



import emoji


def main():
    file_exists = exists('features.csv')
    if file_exists:
        df = pd.read_csv('features.csv', low_memory=False)
        df = df.iloc[1:, :]
        df = df.drop(columns="index")
    else:
        before = time.time()
        df = get_tweets()
        after = time.time()
        print('time to get tweets {0}'.format(after-before))
        # write dataframe to file as it takes around 30 min to prepare
        # can read in again if needed by adding name to index column in csv
        df.to_csv('features.csv')

    print(df.head(10))
    before = time.time()
    clf = DecisionTreeClassifier()
    y = df['viral']
    X = df[df.columns[1:len(df.columns)]]

    # single calcs

    scores = cross_val_score(clf, X, y, cv=10)  # times of validation
    after = time.time()
    print('time to get scores {0}'.format(after - before))
    print(scores)
    print('mean accuracy = {0}'.format(statistics.mean(scores)))

# gets tweets from datasheet and organizes for training input
def get_tweets():
    column_names = ['Tweet_text', 'viral']
    df = pd.read_csv('Tweet_set1.csv', names=column_names)
    tweets = df.Tweet_text.to_list()

    # initialize features lists
    total_1grams = []
    total_2grams = []
    total_3grams = []
    total_4grams = []
    total_5grams = []

    char_2gram = []
    char_3gram = []
    char_4gram = []
    char_5gram = []

    emoticons = []
    hashtags = []

    # iterate through the tweet
    for tweet in tweets:
        words_list = tweet.split()
        # get word grams in tweet
        wordgrams_intweet = list(everygrams(words_list, 1, 5))
        for ngram in wordgrams_intweet:
            if len(ngram) == 1:
                total_1grams.append(ngram)
            elif len(ngram) == 2:
                total_2grams.append(ngram)
            elif len(ngram) == 3:
                total_3grams.append(ngram)
            elif len(ngram) == 4:
                total_4grams.append(ngram)
            elif len(ngram) == 5:
                total_5grams.append(ngram)

        # character grams
        character_grams_intweet = list(everygrams(tweet, 2, 5))
        for c_gram in character_grams_intweet:
            if len(c_gram) == 2:
                char_2gram.append(c_gram)
            elif len(c_gram) == 3:
                char_3gram.append(c_gram)
            elif len(c_gram) == 4:
                char_4gram.append(c_gram)
            elif len(c_gram) == 5:
                char_5gram.append(c_gram)
        # emoticons hashtags
        for word in words_list:
            if '#' in word:
                if word not in hashtags:  # not stripping hashtag so checking for hashtag is easier
                    hashtags.append(word)
            if word in emoji.UNICODE_EMOJI_ENGLISH:
                if word not in emoticons:
                    emoticons.append(word)
    # end tweets

    # find most common features
    most_common_1grams = collections.Counter(total_1grams).most_common(50)
    most_common_2grams = collections.Counter(total_2grams).most_common(50)
    most_common_3grams = collections.Counter(total_3grams).most_common(50)
    most_common_4grams = collections.Counter(total_4grams).most_common(50)
    most_common_5grams = collections.Counter(total_5grams).most_common(50)

    most_common_2char = collections.Counter(char_2gram).most_common(50)
    most_common_3char = collections.Counter(char_3gram).most_common(50)
    most_common_4char = collections.Counter(char_4gram).most_common(50)
    most_common_5char = collections.Counter(char_5gram).most_common(50)

    # add columns to dataframe
    init_input = pd.read_csv('Tweet_set1.csv', names=column_names)

    # add columns to datafram
    new_columns = most_common_1grams + most_common_2grams + most_common_3grams + most_common_4grams + \
                  most_common_5grams + most_common_2char + most_common_3char + most_common_4char + most_common_5char + \
                  hashtags + emoticons
    columns_to_add = pd.DataFrame(columns=new_columns)
    learn_input = pd.concat([init_input, columns_to_add], axis=1)

    # iterate through tweets and populate features
    for index, row in learn_input.iterrows():
        tweet = row['Tweet_text']
        words_list = tweet.split()
        wordgrams_intweet = list(everygrams(words_list, 1, 5))
        character_grams_intweet = list(everygrams(tweet, 2, 5))
        for _1gram in most_common_1grams:
            if _1gram[0] in wordgrams_intweet:
                learn_input[_1gram][index] = 1
            else:
                learn_input[_1gram][index] = 0
        for _2gram in most_common_2grams:
            if _2gram[0] in wordgrams_intweet:
                learn_input[_2gram][index] = 1
            else:
                learn_input[_2gram][index] = 0
        for _3gram in most_common_3grams:
            if _3gram[0] in wordgrams_intweet:
                learn_input[_3gram][index] = 1
            else:
                learn_input[_3gram][index] = 0
        for _4gram in most_common_4grams:
            if _4gram[0] in wordgrams_intweet:
                learn_input[_4gram][index] = 1
            else:
                learn_input[_4gram][index] = 0
        for _5gram in most_common_5grams:
            if _5gram[0] in wordgrams_intweet:
                learn_input[_5gram][index] = 1
            else:
                learn_input[_5gram][index] = 0

        for _2char in most_common_2char:
            if _2char[0] in character_grams_intweet:
                learn_input[_2char][index] = 1
            else:
                learn_input[_2char][index] = 0
        for _3char in most_common_3char:
            if _3char[0] in character_grams_intweet:
                learn_input[_3char][index] = 1
            else:
                learn_input[_3char][index] = 0
        for _4char in most_common_4char:
            if _4char[0] in character_grams_intweet:
                learn_input[_4char][index] = 1
            else:
                learn_input[_4char][index] = 0
        for _5char in most_common_5char:
            if _5char[0] in character_grams_intweet:
                learn_input[_5char][index] = 1
            else:
                learn_input[_5char][index] = 0
        for hashtag in hashtags:
            if hashtag in words_list:
                learn_input[hashtag][index] = 1
            else:
                learn_input[hashtag][index] = 0
        for emote in emoticons:
            if emote in words_list:
                learn_input[emote][index] = 1
            else:
                learn_input[emote][index] = 0
        #print(learn_input.head())

    #drop tweet text as we dont want to train on the actual text
    return learn_input.drop(columns="Tweet_text")


if __name__ == '__main__':
    main()
