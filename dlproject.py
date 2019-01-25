import pandas as pd
import re
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import TimeDistributed
from keras import optimizers
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier

hash_replace = "HASHTAGHERE"
mention_replace = "MENTIONHERE"
pad_token = '_PAD_'


def read_tweets():
    """
    read in data and apply preprocessing
    """

    tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)

    df = pd.read_csv('germeval2018.training.txt',
                     sep='\t',
                     header=None,
                     names=['TWEET', 'LABEL-TASK-I', 'LABEL-TASK-II'],
                     encoding='utf-8')

    df_task_II = df[['TWEET', 'LABEL-TASK-II']]

    tweets = preprocessing(df_task_II['TWEET'], tokenizer)
    class_mapping = {"PROFANITY": 0, "INSULT": 1, "ABUSE": 1, "OTHER": 2}
    labels = transform_labels(df_task_II['LABEL-TASK-II'], class_mapping)

    return tweets.tolist(), labels.tolist()

def process_sample(sample, tokenizer):
    """tokenize each tweet sample, replace hashes and mentions
    """
    tokens = tokenizer.tokenize(sample.lower())

    for i, token in enumerate(tokens):
        if token.startswith("#"):
            tokens[i] = hash_replace
        elif token.startswith("@"):
            tokens[i] = mention_replace
    return tokens

def preprocessing(data, tokenizer):
    """apply preprocessing to all data
    """

    new_data = data.apply(lambda sample: process_sample(sample, tokenizer))
    return new_data

def transform_labels(data, class_mapping):
    """map class to data, not used for the task
    """

    new_data = data.apply(lambda label: class_mapping[label])

    return new_data

def create_vocabulary_embeddings(tweets):
    """read embeddings, create vocabulary with padding, unknown tokens, hash and mention replacement,
        create mapping with words and indexes
    """

    import os
    import pickle

    if os.path.exists('emb.pkl'):
        with open('emb.pkl', 'rb') as fp:
            vocabulary, embedding_list, word2index, index2word = pickle.load(fp)
        return vocabulary, embedding_list, word2index, index2word

    wv_from_text = KeyedVectors.load_word2vec_format('embed_tweets_de_100D_fasttext', binary=False)

    vocabulary = set(wv_from_text.vocab.keys())

    tweet_vocab = set()

    for tweet in tweets:
        for token in tweet:
            tweet_vocab.add(token)

    vocabulary = vocabulary.intersection(tweet_vocab)

    vocabulary = list(sorted(vocabulary))

    embedding_list = []

    for item in vocabulary:
        embedding_list.append(wv_from_text[item])

    vocabulary = ['PAD', 'UNK', hash_replace, mention_replace] + vocabulary
    embedding_list = [np.zeros(100), np.random.randn(100), np.random.randn(100), np.random.randn(100)] + embedding_list

    word2index = {}
    index2word = {}
    for i, word in enumerate(vocabulary):
        word2index[word] = i
        index2word[i] = word

    with open('emb.pkl', 'wb') as fp:
        pickle.dump([vocabulary, embedding_list, word2index, index2word], fp)

    return vocabulary, embedding_list, word2index, index2word



def transform_tokens_to_indices(tokens, word2index, unk_token='UNK', return_as_list=True):
    """
    transform tokens to indices 
    """
    indices = []
    for token in tokens:
        if token in word2index:
            indices.append(word2index[token])
        else:
            indices.append(word2index[unk_token])
    return indices if not return_as_list else indices

def create_model(vocabulary, embedding_list, maxlen, batch_size, optimizer):
    """not used, saved for Grid Search implementation later on"""
    model = Sequential()
    model.add(Embedding(len(vocabulary), 100, weights=[np.array(embedding_list)], input_length=maxlen-1, mask_zero=True, trainable=False))
    model.add(LSTM(128, input_shape=(batch_size, maxlen-1), return_sequences=True))
    model.add(TimeDistributed(Dense(len(vocabulary), activation='relu')))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # summarize the model
    print(model.summary())

    return model

def main():
    """main pipeline
    """

    # For reproducabilty 
    np.random.seed(28111993)
    random.seed(28111993)


    # Read data & labels
    print('Reading tweets')
    tweets, classification_labels = read_tweets()

    #find maximum tweet length
    tweets_lengths = [len(tweet) for tweet in tweets]
    maxlen = max(tweets_lengths)

    #make all tweets same length
    tweets = [tweet + [0]*(maxlen - len(tweet)) for tweet in tweets]

    # Reduce word embedding set, prepare embedding matrix with the mapping token -> index and index -> token
    print('Preparing vocab + embeddings')
    vocabulary, embedding_list, word2index, index2word = create_vocabulary_embeddings(tweets)

    #print (len(vocabulary), len(embedding_list), (len(word2index), len(index2word))

    # Classification tasks (not used currently)
    #print('Preparing labels')
    # Transform into one-hot encoding 
    #classification_labels_one_hot_encoding = [to_categorical(label, num_classes=3) for label in classification_labels]


    # Language model tasks
    tweets_shifted_by_one = [tweet[1:] for tweet in tweets] 
    tweets_shifted_by_one_index = [transform_tokens_to_indices(tweet, word2index) for tweet in tweets_shifted_by_one]
    tweets_shifted_by_one_index_one_hot = [to_categorical(tweet_shifted, num_classes=len(vocabulary)) for tweet_shifted in tweets_shifted_by_one_index]

    # Encode the input
    tweets_index = [transform_tokens_to_indices(tweet, word2index, return_as_list=True)[:-1] for tweet in tweets]

    indices = list(range(0, len(tweets)))
    random.shuffle(indices)

    # Split train/test
    # for later
    # idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=28111993, shuffle=True)


    # Sanity Check
    # print('Sanity check')
    # i = indices[28]
    # print(tweets[i])
    # print(tweets_index[i])
    # print([index2word[j] for j in tweets_index[i]])
    # print(tweets_shifted_by_one_index[i])
    # print(classification_labels[i], classification_labels_one_hot_encoding[i])

    
    #Hyperparameter Optimization
    optimizer_list = [optimizers.SGD(lr=0.001), optimizers.RMSprop(lr=0.001), optimizers.Adagrad(lr=0.001), optimizers.Adadelta(lr=0.001), optimizers.Adam(lr=0.001), optimizers.Adamax(lr=0.001)]
    learning_rate_list = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    batch_size_list = [8, 12, 16, 24]


    optimizer = optimizers.RMSprop(lr=0.001)
    batch_size = 12

    print('Building model with batch size {}, learning_rate 0.001 and {} optimizer'.format(12,optimizer))
    model = Sequential()

    assert word2index['PAD'] == 0 # To be just that the index 0 is the padding
    model.add(Embedding(len(vocabulary), 100, weights=[np.array(embedding_list)], input_length=maxlen-1, mask_zero=True, trainable=True))
    model.add(LSTM(128, input_shape=(batch_size, maxlen-1), return_sequences=True))
    model.add(TimeDistributed(Dense(len(vocabulary), activation='relu')))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # summarize the model
    print(model.summary())

    ''' batching, saved for later
    for epoch in range(10):
        batch_indices = []
        for iter, i in enumerate(indices):
            batch_indices.append(i)
            if len(batch_indices) % batch_size == 0 or iter == len(indices) - 1: # Handle also the last batch
                X_batch = np.array([tweets_index[i] for i in batch_indices])
                y_batch = np.array([tweets_shifted_by_one_index_one_hot[i] for i in batch_indices])
                model.fit(X_batch, y_batch, epochs=1)
                batch_indices = []
    #'''


    X_batch = np.array([tweets_index[i] for i in indices])
    y_batch = np.array([tweets_shifted_by_one_index_one_hot[i] for i in indices])
    

    model.fit(X_batch, y_batch, epochs=10, batch_size=batch_size)

if __name__ == '__main__':
    main()