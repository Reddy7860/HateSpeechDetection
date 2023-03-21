import os

from numpy.random import seed
import numpy as np

seed(1)

os.environ['PYTHONHASHSEED'] = '0'
os.environ['THEANO_FLAGS'] = "floatX=float64,device=cpu,openmp=True"
# os.environ['THEANO_FLAGS']="openmp=True"
os.environ['OMP_NUM_THREADS'] = '16'
import theano

theano.config.openmp = True

# import tensorflow as tf
# tf.set_random_seed(2)
# single thread
# session_conf = tf.ConfigProto(
#  intra_op_parallelism_threads=1,
#  inter_op_parallelism_threads=1)

# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# sess = tf.Session(config=session_conf)
# with sess.as_default():
#  print(tf.constant(42).eval())

import datetime
import logging
import sys
import functools
import gensim
import numpy
import random as rn

import codecs
import pdb
import json

import pandas as pd
import pickle
from keras.layers import Embedding
from scikeras.wrappers import KerasClassifier #from tensorflow.keras.wrappers.scikit_learn import KerasClassifier # from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score,StratifiedKFold #from sklearn.cross_validation import cross_val_predict, train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from keras_preprocessing import sequence # from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tabulate import tabulate
from tqdm import trange
import random


from ml import util
from ml import nlp
from ml import text_preprocess as tp
from ml import dnn_model_creator as dmc
from ml import five_words_selector as fws

from keras.utils import to_categorical 
	
import math
from numpy import nan

from nltk.corpus import stopwords
from collections import Counter
from itertools import chain
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


MAX_SEQUENCE_LENGTH = 100  # maximum # of words allowed in a tweet
WORD_EMBEDDING_DIM_OUTPUT = 300
WORD_EMBEDDING_DIM_OUTPUT = 300
CPUS = 1


def get_word_vocab(tweets, out_folder, normalize_option):
    print("Hi from get_word_vocab")
    print(len(tweets))
    # random_words = fws.generate_five_random_words()
    # print("Random 5 words are : ")
    # print(random_words)
    print(nlp.stopwords)

    from nltk.corpus import stopwords
    # set of stop words 
    eng_stop_words = stopwords.words("english")

    tweet_without_stopwords = tweets.apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stop_words)]))

    # tweet_without_stopwords=  tweets.apply(lambda x: [item for item in x if item not in eng_stop_words])


    word_vectorizer = CountVectorizer(
    # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    # tokenizer=functools.partial(nlp.tokenize, stem_or_lemma=normalize_option),
    tokenizer = functools.partial(nlp.glove_tokenize),
    preprocessor=tp.strip_hashtags,
    ngram_range=(1, 1),
    # stop_words=nlp.stopwords,  # We do better when we keep stopwords
    # stop_words = None,
    decode_error='replace',
    max_features=50000,
    min_df=1,
    max_df=0.99
    )
    

    # word_vectorizer = CountVectorizer()

    # logger.info("\tgenerating word vectors, {}".format(datetime.datetime.now()))
    # counts = word_vectorizer.fit_transform(tweets).toarray()
    counts = word_vectorizer.fit_transform(tweet_without_stopwords).toarray()
    # logger.info("\t\t complete, dim={}, {}".format(counts.shape, datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(word_vectorizer.get_feature_names())}

    print("Length if the vocabulary ")
    print(len(vocab))
    pickle.dump(vocab, open(out_folder + "/" + "DNN_WORD_EMBEDDING" + ".pk", "wb"))

    word_embedding_input = []
    for tweet in counts:
        tweet_vocab = []
        for i in range(0, len(tweet)):
            if tweet[i] != 0:
                tweet_vocab.append(i)
        word_embedding_input.append(tweet_vocab)
        
    # print("Love Index: ")
    # print(vocab['love'])
    # random_words = fws.generate_five_random_words()
    # print("Random 5 words are : ")
    # print(random_words)
    # random_five_words_data = pd.DataFrame(columns=['word','index'])
    # temp_cnt = 0
    # for wrd in range(0,15):
    #     try:
    #         random_five_words_data.loc[wrd,"word"] = str(random_words[wrd])
    #         random_five_words_data.loc[wrd,"index"] = vocab[str(random_words[wrd])]
    #     except Exception as e:
    #         print(e)

    return word_embedding_input, vocab


def create_model(model_descriptor: str, max_index=100, wemb_matrix=None, wdist_matrix=None):
    '''A model that uses word embeddings'''
    if wemb_matrix is None:
        print("wemb_matrix is None")
        if wdist_matrix is not None:
            print("wdist_matrix is None")
            embedding_layers = [Embedding(input_dim=max_index, output_dim=WORD_EMBEDDING_DIM_OUTPUT,
                                          input_length=MAX_SEQUENCE_LENGTH),
                                Embedding(input_dim=max_index, output_dim=len(wdist_matrix[0]),
                                          weights=[wdist_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False)]
        else:
            embedding_layers = [Embedding(input_dim=max_index, output_dim=WORD_EMBEDDING_DIM_OUTPUT,
                                          input_length=MAX_SEQUENCE_LENGTH)]

    else:
        print("wemb_matrix is not None")
        if wdist_matrix is not None:
            print("wdist_matrix is not None")
            concat_matrices = util.concat_matrices(wemb_matrix, wdist_matrix)
            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layers = [Embedding(input_dim=max_index, output_dim=len(concat_matrices[0]),
                                          weights=[concat_matrices],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False)]
        else:
            embedding_layers = [Embedding(input_dim=max_index, output_dim=len(wemb_matrix[0]),
                                          weights=[wemb_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False)]
    print("Embedding layers : ")
    print(embedding_layers)
    if model_descriptor.startswith("b_"):
        print("model starts with b_")
        model_descriptor = model_descriptor[2:].strip()
        model = dmc.create_model_with_branch(embedding_layers, model_descriptor)
    elif model_descriptor.startswith("f_"):
        print("model starts with f_")
        model = dmc.create_final_model_with_concat_cnn(embedding_layers, model_descriptor)
    else:
        print("model starts doesnot start with b_ and f_")
        model = dmc.create_model_without_branch(embedding_layers, model_descriptor)
    # create_model_conv_lstm_multi_filter(embedding_layer)

    print("New run started at {}\n{}".format(datetime.datetime.now(), model.summary()))

    # logger.info("New run started at {}\n{}".format(datetime.datetime.now(), model.summary()))
    return model


class MyKerasClassifier(KerasClassifier):
    def predict(self, x, **kwargs):
        kwargs = self.filter_sk_params(self.model.predict, kwargs)
        proba = self.model.predict(x, **kwargs)
        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > 0.5).astype('int32')
        return self.classes_[classes]


# def pretrained_embedding_with_wdist(word_vocab: dict, models: list, expected_emb_dim, randomize_strategy,
#                                     word_dist_scores_file=None):
#     # logger.info("\tloading pre-trained embedding model... {}".format(datetime.datetime.now()))
#     # logger.info("\tloading complete. {}".format(datetime.datetime.now()))
#     word_dist_scores = None
#     if word_dist_scores_file is not None:
#         print("using word dist features...")
#         word_dist_scores = util.read_word_dist_features(word_dist_scores_file)
#         expected_emb_dim += 2
#
#     randomized_vectors = {}
#     matrix = numpy.zeros((len(word_vocab), expected_emb_dim))
#     count = 0
#     random = 0
#     for word, i in word_vocab.items():
#         is_in_model = False
#         for model in models:
#             if word in model.wv.vocab.keys():
#                 is_in_model = True
#                 vec = model.wv[word]
#                 if word_dist_scores is not None:
#                     vec = util.append_word_dist_features(vec, word, word_dist_scores)
#                 matrix[i] = vec
#                 break
#
#         if not is_in_model:
#             random += 1
#             model = models[0]
#             if randomize_strategy == 1:  # randomly set values following a continuous uniform distribution
#                 vec = numpy.random.random_sample(expected_emb_dim)
#                 if word_dist_scores is not None:
#                     vec = util.append_word_dist_features(vec, word, word_dist_scores)
#                 matrix[i] = vec
#             elif randomize_strategy == 2:  # randomly take a vector from the model
#                 if word in randomized_vectors.keys():
#                     vec = randomized_vectors[word]
#                 else:
#                     max = len(model.wv.vocab.keys()) - 1
#                     index = rn.randint(0, max)
#                     word = model.index2word[index]
#                     vec = model.wv[word]
#                     randomized_vectors[word] = vec
#                 if word_dist_scores is not None:
#                     vec = util.append_word_dist_features(vec, word, word_dist_scores)
#                 matrix[i] = vec
#         count += 1
#         if count % 100 == 0:
#             print(count)
#     models.clear()
#     if randomize_strategy != 0:
#         print("randomized={}".format(random))
#     else:
#         print("oov={}".format(random))
#     return matrix


def build_pretrained_embedding_matrix(word_vocab: dict, models: list, expected_emb_dim, randomize_strategy
                                      ):
    # logger.info("\tloading pre-trained embedding model... {}".format(datetime.datetime.now()))
    # logger.info("\tloading complete. {}".format(datetime.datetime.now()))
    print("Length of list")
    # print(len(models))

    randomized_vectors = {}
    matrix = numpy.zeros((len(word_vocab), expected_emb_dim))
    count = 0
    random = 0
    for word, i in word_vocab.items():
        
        # print(word)
        # print(i)
        is_in_model = False
        for model in models:
            # print(type(model))
            # if word in model.wv.vocab.keys():
            if word in model:
                is_in_model = True
                vec = model[word]
                matrix[i] = vec
                break

        if not is_in_model:
            random += 1
            model = models[0]
            if randomize_strategy == '1' or randomize_strategy == 1:  # randomly set values following a continuous uniform distribution
                vec = numpy.random.random_sample(expected_emb_dim)
                matrix[i] = vec
            elif randomize_strategy == '2' or randomize_strategy == 2:  # randomly take a vector from the model
                if word in randomized_vectors:   #if word in randomized_vectors.keys():
                    vec = randomized_vectors[word]
                else:
                    # max = len(model.wv.vocab.keys()) - 1
                    max = len(model) - 1
                    index = rn.randint(0, max)
                    # print("In the index part")
                    # print(index)
                    word = model.index_to_key[index]
                    vec = model[word]  #vec = model.wv[word]
                    randomized_vectors[word] = vec
                matrix[i] = vec
        count += 1
        # if count % 100 == 0:
        #     print(count)
    if randomize_strategy != '0':
        print("randomized={}".format(random))
    else:
        print("oov={}".format(random))

    models.clear()

    return matrix


def build_word_dist_matrix(word_vocab: dict,
                           word_dist_scores_file):
    word_dist_scores = util.read_word_dist_features(word_dist_scores_file)
    expected_emb_dim = 2

    matrix = numpy.zeros((len(word_vocab), expected_emb_dim))
    count = 0
    for word, i in word_vocab.items():
        vec = util.build_word_dist_features(word, word_dist_scores)
        matrix[i] = vec

        count += 1
        if count % 100 == 0:
            print(count)

    return matrix


def grid_search_dnn(dataset_name, outfolder, model_descriptor: str,
                    cpus, nfold, X_train, y_train, X_test, y_test,X_test_non_hate, X_test_data_common,Adversial_non_hate_X_train_data,Adversial_select_X_train_data,adversial_target,vocab, X_train_index, X_test_index,
                    embedding_layer_max_index, pretrained_embedding_matrix=None,
                    word_dist_matrix=None,
                    instance_tags_train=None, instance_tags_test=None,
                    accepted_ds_tags: list = None):
    print("Dataset name : ",str(dataset_name))
    print("outfolder name : ",str(outfolder))
    print("model_descriptor name : ",str(model_descriptor))
    print(cpus)
    print(nfold)
    # print(X_train)
    # print(y_train.tail())
    # print(X_test)
    # print(y_test.tail())
    # print(X_train_index)
    # print(X_test_index)
    print(embedding_layer_max_index)
    print("\t== Perform ANN ...")
    subfolder = outfolder + "/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    create_model_with_args = \
        functools.partial(create_model, max_index=embedding_layer_max_index,
                          wemb_matrix=pretrained_embedding_matrix,
                          wdist_matrix=word_dist_matrix,
                          model_descriptor=model_descriptor)

    print("Hello KerasClassifier")
    print(create_model_with_args)
    # model = MyKerasClassifier(build_fn=create_model_with_args, verbose=0)
    # model = KerasClassifier(build_fn=create_model_with_args, verbose=0)
    # model = KerasClassifier(model=create_model_with_args, verbose=0)

    # print(model)

    

    model = KerasClassifier(model=create_model_with_args, verbose=0, batch_size=100,
                            epochs=1,random_state = 143)
    adversial_non_hate_model = KerasClassifier(model=create_model_with_args, verbose=0, batch_size=100,
                            epochs=1,random_state = 143)
    adversial_select_model = KerasClassifier(model=create_model_with_args, verbose=0, batch_size=100,
                            epochs=1,random_state = 143)
    #
    # nfold_predictions = cross_val_predict(model, X_train, y_train, cv=nfold)

    # define the grid search parameters
    batch_size = [100]
    epochs = [10]
    # param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    # param_grid = dict(batch_size=batch_size, epochs=epochs)

    # print(param_grid)

    #it seems that the default gridsearchcv can have problem with stratifiedkfold sometimes, on w and ws dataset when we add "mixed_data"
    # fold=StratifiedKFold(n_splits=nfold) # fold=StratifiedKFold(n_folds=nfold, y=y_train)
    # _classifier = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=cpus,
    #                            cv=fold)

    #this is the original grid search cv object to replace the above
    # _classifier = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=cpus,
                            #   cv=nfold)
    # print(help(_classifier))
    print("\tfitting model...{}".format(datetime.datetime.now()))
    print("Data shape : ")
    print(X_train.shape)
    print(y_train.shape)

    print("Adversial Non Hate Data shape : ")
    print(Adversial_non_hate_X_train_data.shape)

    print("Adversial Select Data shape : ")
    print(Adversial_select_X_train_data.shape)

    print(adversial_target.shape)

    # grid_result = _classifier.fit(X_train, y_train)
    y_train = to_categorical(y_train)
    adversial_target = to_categorical(adversial_target)
    y_test = to_categorical(y_test)
    print(y_train.shape)
    print(adversial_target.shape)
    print(y_train)
    
    for i in range(1,11):
        model.partial_fit(X_train, y_train)
    for i in range(1,11):
        adversial_non_hate_model.partial_fit(Adversial_non_hate_X_train_data, adversial_target)
    for i in range(1,11):
        adversial_select_model.partial_fit(Adversial_select_X_train_data, adversial_target)
    # model.fit(X_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # print("_classifier.best_estimator_ : ",_classifier.best_estimator_)
    # _classifier.fit(X_train, y_train)
    print("\tcrossfold running...{}".format(datetime.datetime.now()))
    # nfold_predictions = cross_val_predict(_classifier.best_estimator_, X_train, y_train, cv=nfold)
    # best_param_ann = _classifier.best_params_
    # print("\tdone {}".format(datetime.datetime.now()))
    # print("\tbest params for {} model are:{}".format(model_descriptor, best_param_ann))
    # best_estimator = _classifier.best_estimator_
    nfold_predictions = model.predict(X_train)
    nfold_predictions_test = model.predict(X_test)
    # nfold_predictions_test_love = model.predict(X_test_love_data)
    nfold_predictions_test_non_hate = model.predict(X_test_non_hate)
    nfold_predictions_test_random_5_words = model.predict(X_test_data_common)


    adversial_non_hate_nfold_predictions = adversial_non_hate_model.predict(X_test_non_hate)
    adversial_select_nfold_predictions = adversial_select_model.predict(X_test_data_common)
    # adversial_nfold_predictions_test = adversial_model.predict(X_test)
    # # nfold_predictions_test_love = model.predict(X_test_love_data)
    # adversial_nfold_predictions_test_non_hate = adversial_model.predict(X_test_non_hate)
    # adversial_nfold_predictions_test_random_5_words = adversial_model.predict(X_test_data_common)

    print("Below is training score")
    print(model.score(X_train,y_train))
    print("Below is test score")
    print(model.score(X_test,y_test))
    # print("below is test love score")
    # print(model.score(X_test_love_data,y_test))
    print("below is random non hate words score")
    print(model.score(X_test_non_hate,y_test))
    # print(X_test_non_hate[0])
    print("below is random common words score")
    print(model.score(X_test_data_common,y_test))

    print("Below is adversial non hate score")
    print(adversial_non_hate_model.score(X_test_non_hate,y_test))

    print("Below is adversial select score")
    print(adversial_select_model.score(X_test_data_common,y_test))

    # print("Below is adversial training score")
    # print(adversial_model.score(Adversial_X_train_data,adversial_target))
    # print("Below adversial is test score")
    # print(adversial_model.score(X_test,y_test))
    # print("below is adversial random non hate words score")
    # print(adversial_model.score(X_test_non_hate,y_test))
    # # print(X_test_non_hate[0])
    # print("below is adversial random common words score")
    # print(adversial_model.score(X_test_data_common,y_test))

    print("Training classification report")
    print(classification_report(y_train, nfold_predictions))
    print("Test Classification report")
    print(classification_report(y_test, nfold_predictions_test))
    # print("Test Love Classification report")
    # print(classification_report(y_test, nfold_predictions_test_love))
    print("Random non hate words Classification report")
    print(classification_report(y_test, nfold_predictions_test_non_hate))
    print("Random common words Classification report")
    print(classification_report(y_test, nfold_predictions_test_random_5_words))

    print("Adversial Non Hate classification report")
    print(classification_report(y_test, adversial_non_hate_nfold_predictions))

    print("Adversial Select classification report")
    print(classification_report(y_test, adversial_select_nfold_predictions))

    # print("Adversial Training classification report")
    # print(classification_report(adversial_target, adversial_nfold_predictions))
    # print("Adversial Test Classification report")
    # print(classification_report(y_test, adversial_nfold_predictions_test))
    # print("Adversial Random non hate words Classification report")
    # print(classification_report(y_test, adversial_nfold_predictions_test_non_hate))
    # print("Adversial Random common words adversial_Classification report")
    # print(classification_report(y_test, adversial_nfold_predictions_test_random_5_words))


    # logloss = log_loss(y_test,model.predict_proba(X_test))
    logloss = log_loss(y_train,model.predict_proba(X_train))
    print(f"Log Loss for training : {logloss}")
    adversial_non_hate_logloss = log_loss(adversial_target,adversial_non_hate_model.predict_proba(Adversial_non_hate_X_train_data))
    print(f"Adversial Non Hate Log Loss for training : {adversial_non_hate_logloss}")
    adversial_select_logloss = log_loss(adversial_target,adversial_select_model.predict_proba(Adversial_select_X_train_data))
    print(f"Adversial Select Log Loss for training : {adversial_select_logloss}")

    # keys_list = [x for x in range(100)]

    # # initialize dictionary
    # sent_count_dict = {}
    
    # # iterating through the elements of list
    # for i in keys_list:
    #     sent_count_dict[i] = 0

    # for idx in range(0,len(X_train)):
    #     counts = Counter(X_train[idx])
    #     ## we are excluding the 0 
    #     current_sentence_length = len(counts) - 1
    #     sent_count_dict[current_sentence_length] = sent_count_dict[current_sentence_length] + 1

    # print(sent_count_dict)

    # sent_df = pd.DataFrame(sent_count_dict, index=[1]) 

    # sent_df = sent_df.transpose()

    # sent_df.reset_index(inplace=True)
    # sent_df =sent_df.rename(columns = {'index':'Sentence_Length'})
    # sent_df.columns = ['Sentence_Length','Count']

    # x = sent_df['Sentence_Length']
    # y = sent_df['Count']

    # plt.bar(x,y,align='center') # A bar chart
    # plt.xlabel('Words Count')
    # plt.ylabel('Frequency')

    # plot_filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.png'

    # plt.savefig(outfolder + '/' + plot_filename)


    # def most_frequent(List):
    #     occurence_count = Counter(List)
    #     most_frequent_words = occurence_count.most_common(11)
    #     print("most_frequent_words")
    #     print(most_frequent_words)
    #     most_frequent_list = [most_frequent_words[1][0],most_frequent_words[2][0],most_frequent_words[3][0],most_frequent_words[4][0],most_frequent_words[5][0],most_frequent_words[6][0],most_frequent_words[7][0],most_frequent_words[8][0],most_frequent_words[9][0],most_frequent_words[10][0]]
    #     return most_frequent_list

    # for epo in range(0,5):
    #     print(f"Running the {epo} :")
    #     nfold_predictions = np.argmax(nfold_predictions, axis=-1)
    #     nfold_actual = np.argmax(y_train, axis=-1)
    #     # print("Before ")
    #     print(nfold_predictions)
    #     print(len(nfold_predictions))

        
    #     adversial_nfold_predictions = np.argmax(adversial_nfold_predictions, axis=-1)
    #     adversial_nfold_actual = np.argmax(adversial_target, axis=-1)
    #     # print(np.argmax(adversial_nfold_predictions, axis=-1))
    #     print(adversial_nfold_predictions)
    #     print(len(adversial_nfold_predictions))


    #     predicted_non_hate_dataset = []
    #     actual_non_hate_dataset = []

    #     for train_idx in range(0,len(nfold_predictions)):
    #         # capturing the predicted class as non-hate
    #         if nfold_predictions[train_idx] == 1:
    #             predicted_non_hate_dataset.append(X_train[train_idx])
    #         # capturing the actual class as non - hate
    #         if nfold_actual[train_idx] == 1:
    #             actual_non_hate_dataset.append(X_train[train_idx])

    #     adversial_predicted_non_hate_dataset = []
    #     adversial_actual_non_hate_dataset = []

    #     for train_idx in range(0, len(adversial_nfold_predictions)):
    #         # capturing the predicted class as non-hate
    #         if adversial_nfold_predictions[train_idx] == 1:
    #             adversial_predicted_non_hate_dataset.append(Adversial_X_train_data[train_idx])
    #         # capturing the actual class as non - hate
    #         if adversial_nfold_actual[train_idx] == 1:
    #             adversial_actual_non_hate_dataset.append(Adversial_X_train_data[train_idx])

    #     # converting to the normal list
    #     predicted_non_hate_dataset = list(chain(*predicted_non_hate_dataset))
    #     actual_non_hate_dataset = list(chain(*actual_non_hate_dataset))
    #     adversial_predicted_non_hate_dataset = list(chain(*adversial_predicted_non_hate_dataset))
    #     adversial_actual_non_hate_dataset = list(chain(*adversial_actual_non_hate_dataset))

    #     most_frequent_list = most_frequent(predicted_non_hate_dataset)
    #     actual_most_frequent_list = most_frequent(actual_non_hate_dataset)
    #     adversial_most_frequent_list = most_frequent(adversial_predicted_non_hate_dataset)
    #     adversial_actual_most_frequent_list = most_frequent(adversial_actual_non_hate_dataset)

    #     print("Most Frequent Predicted List :")
    #     print(most_frequent_list)
    #     print("Most Frequent Actual List :")
    #     print(actual_most_frequent_list)

    #     print("Adversial Most Frequent Predicted List :")
    #     print(adversial_most_frequent_list)
    #     print("Adversial Most Frequent Actual List :")
    #     print(adversial_actual_most_frequent_list)

    #     ## Checking the words from the index
    #     most_frequent_words_list = []
    #     for idx in range(0,len(most_frequent_list)):
    #         value = {i for i in vocab if vocab[i]==most_frequent_list[idx]}
    #         most_frequent_words_list.append(value)

    #     adversial_most_frequent_words_list = []
    #     for idx in range(0,len(adversial_most_frequent_list)):
    #         value = {i for i in vocab if vocab[i]==adversial_most_frequent_list[idx]}
    #         adversial_most_frequent_words_list.append(value)

    #     actual_most_frequent_words_list = []
    #     for idx in range(0,len(actual_most_frequent_list)):
    #         value = {i for i in vocab if vocab[i]==actual_most_frequent_list[idx]}
    #         actual_most_frequent_words_list.append(value)
        
    #     adversial_actual_most_frequent_words_list = []
    #     for idx in range(0,len(adversial_actual_most_frequent_list)):
    #         value = {i for i in vocab if vocab[i]==adversial_actual_most_frequent_list[idx]}
    #         adversial_actual_most_frequent_words_list.append(value)

    #     print("Most frequent words list :")
    #     print(most_frequent_words_list)

    #     print("Most frequent actual words list :")
    #     print(actual_most_frequent_words_list)

    #     print("Most frequent Adversial words list :")
    #     print(adversial_most_frequent_words_list)

    #     print("Most frequent Adversial actual words list :")
    #     print(adversial_actual_most_frequent_words_list)

    #     for idx in range(0,5):
    #         print(f"Replacing {most_frequent_words_list[idx]} with 0")
    #         for sent in range(0,len(X_train)):
    #             X_train[sent][X_train[sent] == most_frequent_list[idx]] = 0
        
    #     for idx in range(0,5):
    #         print(f"Replacing {adversial_most_frequent_words_list[idx]} with 0")
    #         for sent in range(0,len(Adversial_X_train_data)):
    #             Adversial_X_train_data[sent][Adversial_X_train_data[sent] == adversial_most_frequent_list[idx]] = 0

    #     keys_list = [x for x in range(100)]
    #     # initialize dictionary
    #     sent_count_dict = {}
    #     # iterating through the elements of list
    #     for i in keys_list:
    #         sent_count_dict[i] = 0

    #     for idx in range(0,len(X_train)):
    #         counts = Counter(X_train[idx])
    #         ## we are excluding the 0 
    #         current_sentence_length = len(counts) - 1
    #         sent_count_dict[current_sentence_length] = sent_count_dict[current_sentence_length] + 1

    #     print(sent_count_dict)
    #     sent_df = pd.DataFrame(sent_count_dict, index=[1]) 
    #     sent_df = sent_df.transpose()
    #     sent_df.reset_index(inplace=True)
    #     sent_df =sent_df.rename(columns = {'index':'Sentence_Length'})
    #     sent_df.columns = ['Sentence_Length','Count']
    #     x = sent_df['Sentence_Length']
    #     y = sent_df['Count']
    #     plt.bar(x,y,align='center') # A bar chart
    #     plt.xlabel('Words Count')
    #     plt.ylabel('Frequency')
    #     plot_filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '_epoch_' + str(epo) + '.png'

    #     plt.savefig(outfolder + '/' + plot_filename)
        
    #     print("Below is the model after removing the top 5 frequent words and partial fit the model")
    #     model.partial_fit(X_train, y_train)

    #     print("Below is the Adversial model after removing the top 5 frequent words and partial fit the model")
    #     adversial_model.partial_fit(Adversial_X_train_data, adversial_target)

    #     logloss = log_loss(y_train,model.predict_proba(X_train))
    #     print(f"Log Loss for training : {logloss}")

    #     adversial_logloss = log_loss(adversial_target,adversial_model.predict_proba(Adversial_X_train_data))
    #     print(f"Adversial Log Loss for training : {adversial_logloss}")

    #     print(f"below is the training score after removing the word {most_frequent_list[0]} , {most_frequent_list[1]} , {most_frequent_list[2]} , {most_frequent_list[3]} , {most_frequent_list[4]}")
    #     print(model.score(X_train,y_train))

    #     print(f"below is the Adversial training score after removing the word {adversial_most_frequent_list[0]} , {adversial_most_frequent_list[1]} , {adversial_most_frequent_list[2]} , {adversial_most_frequent_list[3]} , {adversial_most_frequent_list[4]}")
    #     print(adversial_model.score(Adversial_X_train_data,adversial_target))

    #     logloss = log_loss(y_test,model.predict_proba(X_test))
    #     print(f"Log Loss : {logloss}")

    #     adversial_logloss = log_loss(y_test,adversial_model.predict_proba(X_test))
    #     print(f"Adversial Log Loss : {logloss}")

    #     nfold_predictions = model.predict(X_train)
    #     adversial_nfold_predictions = adversial_model.predict(Adversial_X_train_data)

    #     print("Below is the training score")
    #     print(model.score(X_train,y_train))

    #     print("Below is the Adversial training score")
    #     print(adversial_model.score(Adversial_X_train_data,adversial_target))

    #     print("Below is the training classification report")
    #     print(classification_report(y_train, nfold_predictions))

    #     print("Below is the Adversial training classification report")
    #     print(classification_report(adversial_target, adversial_nfold_predictions))

    #     print("Classification report for non hate appending")
    #     nfold_predictions_test_non_hate = model.predict(X_test_non_hate)
    #     print(classification_report(y_test, nfold_predictions_test_non_hate))

    #     print("Adversial Classification report for non hate appending")
    #     adversial_nfold_predictions_test_non_hate = adversial_model.predict(X_test_non_hate)
    #     print(classification_report(y_test, adversial_nfold_predictions_test_non_hate))

    #     filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '_epoch_' + str(epo) + '_cnn_gru_model.sav'

    #     pickle.dump(model, open(outfolder + "/" + filename, "wb"))




    # print(X_train)
    # print(len(X_train))
    # nfold_predictions = np.argmax(nfold_predictions, axis=-1)
    # nfold_actual = np.argmax(y_train, axis=-1)
    # print(len(nfold_predictions))
    # print(len(nfold_actual))

    # predicted_non_hate_dataset = []
    # actual_non_hate_dataset = []

    # for train_idx in range(0,len(nfold_predictions)):
    #     # print(nfold_predictions[train_idx])
    #     if nfold_predictions[train_idx] == 1:
    #         # print("inside loop")
    #         # print(X_train[train_idx])
    #         predicted_non_hate_dataset.append(X_train[train_idx])

    #     if nfold_actual[train_idx] == 1:
    #         # print("inside loop")
    #         # print(X_train[train_idx])
    #         actual_non_hate_dataset.append(X_train[train_idx])
    
    # print(predicted_non_hate_dataset[0])
    # predicted_non_hate_dataset = list(chain(*predicted_non_hate_dataset))
    # print(len(predicted_non_hate_dataset))

    # print(actual_non_hate_dataset[0])
    # actual_non_hate_dataset = list(chain(*actual_non_hate_dataset))
    # print(len(actual_non_hate_dataset))

    # def most_frequent(List):
    #     occurence_count = Counter(List)
    #     most_frequent_words = occurence_count.most_common(11)
    #     print("most_frequent_words")
    #     print(most_frequent_words)
    #     most_frequent_list = [most_frequent_words[1][0],most_frequent_words[2][0],most_frequent_words[3][0],most_frequent_words[4][0],most_frequent_words[5][0],most_frequent_words[6][0],most_frequent_words[7][0],most_frequent_words[8][0],most_frequent_words[9][0],most_frequent_words[10][0]]
    #     return most_frequent_list
    
    # most_frequent_list = most_frequent(predicted_non_hate_dataset)
    # actual_most_frequent_list = most_frequent(actual_non_hate_dataset)
    # print("Most Frequent List :")
    # print(most_frequent_list)

    # print("Most Frequent Actual List :")
    # print(actual_most_frequent_list)

    # most_frequent_words_list = []
    # for idx in range(0,len(most_frequent_list)):
    #     value = {i for i in vocab if vocab[i]==most_frequent_list[idx]}
    #     most_frequent_words_list.append(value)
    # print("Most frequent words list :")
    # print(most_frequent_words_list)

    # actual_most_frequent_words_list = []
    # for idx in range(0,len(actual_most_frequent_list)):
    #     value = {i for i in vocab if vocab[i]==actual_most_frequent_list[idx]}
    #     actual_most_frequent_words_list.append(value)
    # print("Most frequent actual words list :")
    # print(actual_most_frequent_words_list)

    # for idx in range(0,5):
    #     print(f"Replacing {most_frequent_words_list[idx]} with 0")
    #     for sent in range(0,len(X_train)):
    #         X_train[sent][X_train[sent] == most_frequent_list[idx]] = 0
    
    # print("Below is the model after removing the top 5 frequent words and partial fit the model")
    # model.partial_fit(X_train, y_train)
    # print(f"below is the training score after removing the word {most_frequent_list[0]} , {most_frequent_list[1]} , {most_frequent_list[2]} , {most_frequent_list[3]} , {most_frequent_list[4]}")
    # print(model.score(X_train,y_train))

    # nfold_predictions = model.predict(X_train)

    # print("Below is the training score")
    # print(model.score(X_train,y_train))

    # print("Below is the training classification report")
    # print(classification_report(y_train, nfold_predictions))

    # print("Classification report for non hate appending")
    # nfold_predictions_test_non_hate = model.predict(X_test_non_hate)
    # print(classification_report(y_test, nfold_predictions_test_non_hate))

    # filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '_epoch_' + str(idx) + '_cnn_gru_model.sav'

    # pickle.dump(model, open(outfolder + "/" + filename, "wb"))

    

    

    # for idx in range(0,10):
    #     print(f"below is random non hate words score after replacing word {most_frequent_list[idx]} with 0")
    #     for sent in range(0,len(X_test_non_hate)):
    #         X_test_non_hate[sent][X_test_non_hate[sent] == most_frequent_list[idx]] = 0
    #     nfold_predictions_test_non_hate = model.predict(X_test_non_hate)
    #     print(model.score(X_test_non_hate,y_test))
    #     print(f"Random non hate words Classification report after replacing word {most_frequent_list[idx]} with 0")
    #     print(classification_report(y_test, nfold_predictions_test_non_hate))


    # #### Code for saving the model ######
    # filename = 'cnn_gru_model.sav'
    # pickle.dump(model, open(outfolder + "/" + filename, "wb"))


    # util.save_classifier_model(best_estimator, ann_model_file)

    # logger.info("testing on development set ....")
    # if (X_test is not None):
    #     print("\tpredicting...{}".format(datetime.datetime.now()))
    #     heldout_predictions_final = best_estimator.predict(X_test)
    #     print("\tsaving...{}".format(datetime.datetime.now()))
    #     util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test,
    #                      X_train_index, X_test_index,
    #                      model_descriptor, dataset_name,
    #                      3, outfolder, instance_tags_train, instance_tags_test, accepted_ds_tags)

    # else:
    #     print("\tsaving...{}".format(datetime.datetime.now()))
    #     util.save_scores(nfold_predictions, y_train, None, y_test,X_train_index, X_test_index,
    #                      model_descriptor, dataset_name, 3,
    #                      outfolder, instance_tags_train, instance_tags_test, accepted_ds_tags)

        # util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
        #                       time_ann_predict_dev,
        #                       time_ann_train, y_test)


def output_data_stats(X_train_data, y_train):
    labels={}
    for y in y_train:
        if y in labels.keys():
            labels[y]+=1
        else:
            labels[y]=1
    print("training instances={}, training labels={}, training label distribution={}".
          format(len(X_train_data), len(y_train),labels))

def get_t2_data():
    tweets = []
    files = ['racism.json', 'neither.json', 'sexism.json']
    for file in files:
        with codecs.open('/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/tweet_data/' + file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        for line in data:
            tweet_full = json.loads(line)
            tweets.append({
                'id': tweet_full['id'],
                'text': tweet_full['text'].lower(),
                'label': tweet_full['Annotation'],
                'name': tweet_full['user']['name'].split()[0]
                })

    #pdb.set_trace()
    return tweets


def gridsearch(input_data_file, dataset_name, sys_out, model_descriptor: str,
               print_scores_per_class,
               word_norm_option,
               randomize_strategy,
               pretrained_embedding_models=None, expected_embedding_dim=None,
               word_dist_features_file=None, use_mixed_data=False):

    # data_set_name = "hate_speech"
    data_set_name = "racism_and_sexism"
    
    raw_data = pd.DataFrame()
    if data_set_name == "racism_and_sexism":
        tweets = get_t2_data()
        raw_data = pd.DataFrame(tweets)
        raw_data['label'] = raw_data['label'].replace(["sexism","racism"],0)
        raw_data['label'] = raw_data['label'].replace(["none"],1)

        raw_data.columns = ['id','tweet','class','name']

    else:
        raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")


    # print(raw_data['class'].isna().sum())
    print("Raw Data Input Tail : ")
    raw_data['class'] = raw_data['class'].replace([2],1)
    print(raw_data['class'].value_counts())
    print(raw_data.tail())
    print(sys_out)
    print(word_norm_option)
    non_hate_data = raw_data.loc[raw_data['class']==1,]
    hate_data = raw_data.loc[raw_data['class']==0,]

    non_hate_data.reset_index(drop=True, inplace=True)
    hate_data.reset_index(drop=True, inplace=True)


    # stop = set(stopwords.words('english'))
    # hate_dataset = hate_data
    # hate_dataset['tweet'] = hate_dataset['tweet'].str.lower()
    # hate_dataset['tweet_without_stopwords'] = hate_dataset['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    # top_words = Counter(" ".join(hate_dataset.tweet_without_stopwords).split()).most_common(10)

    # print(top_words)

    print("Non hate top words")
    stop = set(stopwords.words('english'))
    non_hate_data['tweet'] = non_hate_data['tweet'].str.lower()
    non_hate_data['tweet_without_stopwords'] = non_hate_data['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    top_words = Counter(" ".join(non_hate_data.tweet_without_stopwords).split()).most_common(10)

    print(top_words)

    M = get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
    non_hate_M = get_word_vocab(non_hate_data.tweet, sys_out, word_norm_option)
    hate_M = get_word_vocab(hate_data.tweet, sys_out, word_norm_option)
    
    # print(M.shape)
    # M=self.feature_scale(M)
    M0 = M[0]
    vocab = M[1]

    hate_vocab = list(hate_M[1].keys())

    non_hate_words_set = []
    non_hate_set = []
    for key in non_hate_M[1].keys() :
        # print(key)
        if key not in hate_vocab:
            non_hate_set.append(non_hate_M[1][key])
            non_hate_words_set.append(key)

    del non_hate_M
    del hate_M
    del hate_vocab

    print("Length of non hate words : ")
    print(len(non_hate_set))
    print(len(non_hate_words_set))
    # print(non_hate_words_set)
    # print(non_hate_set)
    # print(non_hate_set)
#     print(vocab)
    # print(non_hate_set[0:5])
    # print(non_hate_vocab)
    # print(hate_vocab)
    # print("Love Index: ")
    # print(M[1]['love'])
    # love_index = M[1]['love']

    filtered_sentence = fws.get_filtered_sentence()
    print("Filtered sentence length : ")
    print(len(filtered_sentence))
    select_set = []
    for wrd in filtered_sentence:
        if wrd in vocab:
            select_set.append(vocab[wrd])
    print(len(select_set))
    # print(select_set)

    # selected_model = "SGDClassifier"
    # selected_model = "cnn_gru"
    selected_model = "BertModel"

    if selected_model == "SGDClassifier":
        print("Performing SGD Classifier")

        stop = set(stopwords.words('english'))
        raw_data['tweet'] = raw_data['tweet'].str.lower()
        raw_data['tweet_without_stopwords'] = raw_data['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        
        select_set_words_list = []
        for idx in range(0,len(select_set)):
            value = {i for i in vocab if vocab[i]==select_set[idx]}
            select_set_words_list.append(value)

        print("Select set 5 sample words")
        select_set_words_list = [i for sub_list in select_set_words_list for i in sub_list]
        print(select_set_words_list[0:5])

        non_hate_set_words_list = []
        for idx in range(0,len(non_hate_set)):
            value = {i for i in vocab if vocab[i]==non_hate_set[idx]}
            non_hate_set_words_list.append(value)

        print("Non Hate set 5 sample words")
        non_hate_set_words_list = [i for sub_list in non_hate_set_words_list for i in sub_list]
        print(non_hate_set_words_list[0:5])

        text_trainval, y_trainval = raw_data['tweet_without_stopwords'], raw_data['class']

        print(y_trainval.value_counts())

        text_train, text_val, y_train, y_val = train_test_split(
        text_trainval, y_trainval, stratify=y_trainval, random_state=0)

        print(y_train.value_counts())

        print("type of text_trainval: {}".format(type(text_trainval)))
        print("length of text_trainval: {}".format(len(text_trainval)))
        print("text_trainval[1]:\n{}".format(text_trainval[1]))
        
        adversial_actual_text_trainval = text_train
        # adversial_y_trainval = y_trainval
            
        adversial_non_hate_text_trainval = []
        # Iterate over the sentences
        for i, sentence in enumerate(adversial_actual_text_trainval):
            # Select 5 random words from the list
            random_words = rn.sample(non_hate_set_words_list, 50)
            # Append the random words to the sentence
            adversial_non_hate_text_trainval.append(sentence + ' ' + ' '.join(random_words))
            
        adversial_select_text_trainval = []
        # Iterate over the sentences
        for i, sentence in enumerate(adversial_actual_text_trainval):
            # Select 5 random words from the list
            random_words = rn.sample(select_set_words_list, 50)
            # Append the random words to the sentence
            adversial_select_text_trainval.append(sentence + ' ' + ' '.join(random_words))
        
        print("Adversarial train data prepared")
        print(len(adversial_actual_text_trainval))
        print(len(adversial_non_hate_text_trainval))
        print(len(adversial_select_text_trainval))
        
        print(adversial_actual_text_trainval[1])
        print(adversial_non_hate_text_trainval[1])
        print(adversial_select_text_trainval[1])
        
        # adversial_non_hate_text_trainval = adversial_actual_text_trainval + adversial_non_hate_text_trainval
        # adversial_select_text_trainval = adversial_actual_text_trainval  + adversial_select_text_trainval

        adversial_non_hate_text_trainval = [*adversial_actual_text_trainval, *adversial_non_hate_text_trainval]
        adversial_select_text_trainval = [*adversial_actual_text_trainval, *adversial_select_text_trainval]

        # adversial_non_hate_y_trainval = adversial_y_trainval + adversial_y_trainval
        # adversial_select_y_trainval = adversial_y_trainval + adversial_y_trainval

        # adversial_y_train = y_train + y_train
        adversial_y_train = [*y_train, *y_train]

        print(adversial_y_train[:5])
        
        adversial_non_hate_text_trainval = pd.Series(adversial_non_hate_text_trainval)
        adversial_select_text_trainval = pd.Series(adversial_select_text_trainval)

        # adversial_non_hate_y_trainval = pd.Series(adversial_non_hate_y_trainval)
        # adversial_select_y_trainval = pd.Series(adversial_select_y_trainval)

        adversial_y_train = pd.Series(adversial_y_train)

        print(adversial_y_train.value_counts())

        malory = list(text_trainval)
        adversial_non_hate_malory = list(adversial_non_hate_text_trainval)
        adversial_select_malory = list(adversial_select_text_trainval)
        
        print(malory[0])
        print(adversial_non_hate_malory[0])
        print(adversial_select_malory[0])

        vect = CountVectorizer()
        vect.fit(malory)
        print(f"Length of features in original data: {len(vect.get_feature_names())}")
        
        adversial_non_hate_vect = CountVectorizer()
        adversial_non_hate_vect.fit(adversial_non_hate_malory)
        print(f"Length of features in Non Hate Adversial data: {len(adversial_non_hate_vect.get_feature_names())}")

        adversial_select_vect = CountVectorizer()
        adversial_select_vect.fit(adversial_select_malory)
        print(f"Length of features in Select Adversial data: {len(adversial_select_vect.get_feature_names())}")

        # X = vect.transform(malory)
        # adversial_non_hate_X = adversial_non_hate_vect.transform(adversial_non_hate_malory)
        # adversial_select_X = adversial_select_vect.transform(adversial_select_malory)

        # text_train, text_val, y_train, y_val = train_test_split(
        # text_trainval, y_trainval, stratify=y_trainval, random_state=0)
        
        # adversial_non_hate_text_train, adversial_non_hate_text_val, adversial_non_hate_y_train, adversial_non_hate_y_val = train_test_split(
        # adversial_non_hate_text_trainval, adversial_non_hate_y_trainval, stratify=adversial_y_trainval, random_state=0)

        # adversial_select_text_train, adversial_select_text_val, adversial_select_y_train, adversial_select_y_val = train_test_split(
        # adversial_select_text_trainval, adversial_select_y_trainval, stratify=adversial_y_trainval, random_state=0)

        

        text_val_non_hate = []
        # Iterate over the sentences
        for i, sentence in enumerate(text_val):
            # Select 5 random words from the list
            random_words = rn.sample(non_hate_set_words_list, 50)
            # Append the random words to the sentence
            text_val_non_hate.append(sentence + ' ' + ' '.join(random_words))

        text_val_select_set = []
        # Iterate over the sentences
        for i, sentence in enumerate(text_val):
            # Select 5 random words from the list
            random_words = rn.sample(select_set_words_list, 50)
            # Append the random words to the sentence
            text_val_select_set.append(sentence + ' ' + ' '.join(random_words))
            
            
        # adversial_non_hate_text_val_non_hate = []
        # # Iterate over the sentences
        # for i, sentence in enumerate(adversial_non_hate_text_val):
        #     # Select 5 random words from the list
        #     random_words = rn.sample(non_hate_set_words_list, 10)
        #     # Append the random words to the sentence
        #     adversial_non_hate_text_val_non_hate.append(sentence + ' ' + ' '.join(random_words))

        # adversial_non_hate_text_val_select_set = []
        # # Iterate over the sentences
        # for i, sentence in enumerate(adversial_non_hate_text_val):
        #     # Select 5 random words from the list
        #     random_words = rn.sample(select_set_words_list, 10)
        #     # Append the random words to the sentence
        #     adversial_non_hate_text_val_select_set.append(sentence + ' ' + ' '.join(random_words))

        # adversial_select_text_val_non_hate = []
        # # Iterate over the sentences
        # for i, sentence in enumerate(adversial_select_text_val):
        #     # Select 5 random words from the list
        #     random_words = rn.sample(non_hate_set_words_list, 10)
        #     # Append the random words to the sentence
        #     adversial_select_text_val_non_hate.append(sentence + ' ' + ' '.join(random_words))

        # adversial_select_text_val_select_set = []
        # # Iterate over the sentences
        # for i, sentence in enumerate(adversial_select_text_val):
        #     # Select 5 random words from the list
        #     random_words = rn.sample(select_set_words_list, 10)
        #     # Append the random words to the sentence
        #     adversial_select_text_val_select_set.append(sentence + ' ' + ' '.join(random_words))

        print("After appending the sentences : ")
        print(text_val)
        print(type(text_val))
        print(text_val_non_hate[0])
        print(text_val_select_set[0])

        vect = CountVectorizer()
        X_train = vect.fit_transform(text_train)
        X_val = vect.transform(text_val)
        X_val_non_hate = vect.transform(text_val_non_hate)
        X_val_select_set = vect.transform(text_val_select_set)
        

        feature_names = vect.get_feature_names()

        adversial_non_hate_vect = CountVectorizer()
        adversial_non_hate_X_train = adversial_non_hate_vect.fit_transform(adversial_non_hate_text_trainval)
        adversial_non_hate_X_val_non_hate = adversial_non_hate_vect.transform(text_val_non_hate)

        # adversial_non_hate_X_train = adversial_non_hate_vect.fit_transform(text_val_non_hate)
        # adversial_non_hate_X_val_non_hate = adversial_non_hate_vect.fit_transform(text_val_non_hate)
        # adversial_non_hate_X_val = adversial_non_hate_vect.transform(adversial_non_hate_text_val)
        # adversial_non_hate_X_val_non_hate = adversial_non_hate_vect.transform(adversial_non_hate_text_val_non_hate)
        # adversial_non_hate_X_val_select_set = adversial_non_hate_vect.transform(adversial_non_hate_text_val_select_set)

        adversial_select_vect = CountVectorizer()
        adversial_select_X_train = adversial_select_vect.fit_transform(adversial_select_text_trainval)
        adversial_select_X_val_select_set = adversial_select_vect.transform(text_val_select_set)
        # adversial_select_X_val = adversial_select_vect.transform(adversial_select_text_val)
        # adversial_select_X_val_non_hate = adversial_select_vect.transform(adversial_select_text_val_non_hate)
        # adversial_select_X_val_select_set = adversial_select_vect.transform(adversial_select_text_val_select_set)


        feature_names = vect.get_feature_names()
        adversial_non_hate_feature_names = adversial_non_hate_vect.get_feature_names()
        adversial_select_feature_names = adversial_select_vect.get_feature_names()

        print(f"Length of features in training data: {len(vect.get_feature_names())}")
        print(f"Length of features in adversial Non Hate training data: {len(adversial_non_hate_vect.get_feature_names())}")
        print(f"Length of features in adversial Select training data: {len(adversial_select_vect.get_feature_names())}")

        # create the SGDClassifier object
        clf = SGDClassifier(loss='log')
        adversial_non_hate_clf = SGDClassifier(loss='log')
        adversial_select_clf = SGDClassifier(loss='log')

        # fit the model to the training data
        sgdc = clf.fit(X_train, y_train)
        adversial_non_hate_sgdc = adversial_non_hate_clf.fit(adversial_non_hate_X_train, adversial_y_train)
        adversial_select_sgdc = adversial_select_clf.fit(adversial_select_X_train, adversial_y_train)

        print(f"Train accuracy score : {sgdc.score(X_train, y_train)}")
        print(f"Test accuracy score : {sgdc.score(X_val, y_val)}")
        print(f"Non Hate dataset accuracy score : {sgdc.score(X_val_non_hate, y_val)}")
        print(f"Select dataset accuracy score : {sgdc.score(X_val_select_set, y_val)}")
        
        # print(f"Adversial Non Hate Train accuracy score : {adversial_non_hate_sgdc.score(adversial_non_hate_X_train, adversial_non_hate_y_train)}")
        # print(f"Adversial Non Hate  Test accuracy score : {adversial_non_hate_sgdc.score(adversial_non_hate_X_val, adversial_non_hate_y_val)}")
        # print(f"Adversial Non Hate Non Hate dataset accuracy score : {adversial_non_hate_sgdc.score(adversial_non_hate_X_val_non_hate, adversial_non_hate_y_val)}")
        # print(f"Adversial Non Hate Select dataset accuracy score : {adversial_non_hate_sgdc.score(adversial_non_hate_X_val_select_set, adversial_non_hate_y_val)}")

        # print(f"Adversial Select Train accuracy score : {adversial_select_sgdc.score(adversial_select_X_train, adversial_select_y_train)}")
        # print(f"Adversial Select Test accuracy score : {adversial_select_sgdc.score(adversial_select_X_val, adversial_select_y_val)}")
        # print(f"Adversial Select Non Hate dataset accuracy score : {adversial_select_sgdc.score(adversial_select_X_val_non_hate, adversial_select_y_val)}")
        # print(f"Adversial Select Select dataset accuracy score : {adversial_select_sgdc.score(adversial_select_X_val_select_set, adversial_select_y_val)}")

        nfold_predictions = sgdc.predict(X_train)
        nfold_predictions_test = sgdc.predict(X_val)
        nfold_predictions_test_non_hate = sgdc.predict(X_val_non_hate)
        nfold_predictions_test_select_set = sgdc.predict(X_val_select_set)
        
        # adversial_non_hate_nfold_predictions = adversial_non_hate_sgdc.predict(adversial_non_hate_X_train)
        # adversial_non_hate_nfold_predictions_test = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val)
        adversial_non_hate_nfold_predictions_test_non_hate = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val_non_hate)
        # adversial_non_hate_nfold_predictions_test_select_set = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val_select_set)

        # adversial_select_nfold_predictions = adversial_select_sgdc.predict(adversial_select_X_train)
        # adversial_select_nfold_predictions_test = adversial_select_sgdc.predict(adversial_select_X_val)
        # adversial_select_nfold_predictions_test_non_hate = adversial_select_sgdc.predict(adversial_select_X_val_non_hate)
        adversial_select_nfold_predictions_test_select_set = adversial_select_sgdc.predict(adversial_select_X_val_select_set)


        print("Training classification report :")
        print(classification_report(y_train, nfold_predictions))
        print(classification_report(y_val, nfold_predictions_test))
        print(classification_report(y_val, nfold_predictions_test_non_hate))
        print(classification_report(y_val, nfold_predictions_test_select_set))

        print("Advesarial Non Hate classification with Attack 1")
        
        # print("Adversial Non Hate Training classification report :")
        # print(classification_report(adversial_non_hate_y_train, adversial_non_hate_nfold_predictions))
        # print(classification_report(adversial_non_hate_y_val, adversial_non_hate_nfold_predictions_test))
        # print(y_val)
        print(classification_report(y_val, adversial_non_hate_nfold_predictions_test_non_hate))
        # print(classification_report(adversial_non_hate_y_val, adversial_non_hate_nfold_predictions_test_select_set))

        print("Advesarial Select classification with Attack 2")
        # print("Adversial Select Training classification report :")
        # print(classification_report(adversial_select_y_train, adversial_select_nfold_predictions))
        # print(classification_report(adversial_select_y_val, adversial_select_nfold_predictions_test))
        # print(classification_report(adversial_select_y_val, adversial_select_nfold_predictions_test_non_hate))
        print(classification_report(y_val, adversial_select_nfold_predictions_test_select_set))

#         for epo in range(0,5):
#             print(f"Running the {epo} :")
#         #     print(text_train)
#             predicted_non_hate_dataset = []
#             actual_non_hate_dataset = []
            
#             for train_idx in range(0,len(nfold_predictions)):
#                 # capturing the predicted class as non-hate
#                 if nfold_predictions[train_idx] == 1:
#                     predicted_non_hate_dataset.append(X_train[train_idx])
#                 # capturing the actual class as non - hate
#                 # if nfold_actual[train_idx] == 1:
#                 #     actual_non_hate_dataset.append(X_train[train_idx])
                    
#             coef = sgdc.coef_.ravel()
#             feature_names = np.array(feature_names)

#             inds = np.argsort(coef)
#             # low = inds[:10]
#             high = inds[-10:]

#             important = np.hstack([ high])
#             print(feature_names[important])
            
#             words_to_remove = feature_names[important]
            
#             words_to_remove_regex = '|'.join(words_to_remove)
            
#             text_train = text_train.replace(to_replace=words_to_remove_regex, value='', regex=True)
            
# #             print(text_train)
            
#             vect = CountVectorizer()
#             X_train = vect.fit_transform(text_train)
#             X_val = vect.transform(text_val)
#             X_val_non_hate = vect.transform(text_val_non_hate)
#             X_val_select_set = vect.transform(text_val_select_set)
            
#             clf = SGDClassifier(loss='log')
            
#             feature_names = vect.get_feature_names()
            
#             sgdc = clf.partial_fit(X_train, y_train,classes=[0,1])
            
#             nfold_predictions = sgdc.predict(X_train)
#             nfold_predictions_test = sgdc.predict(X_val)
#             nfold_predictions_test_non_hate = sgdc.predict(X_val_non_hate)
#             nfold_predictions_test_select_set = sgdc.predict(X_val_select_set)

#             print("Training classification report ")
#             print(classification_report(y_train, nfold_predictions))
#             print("Testing classification report ")
#             print(classification_report(y_val, nfold_predictions_test))
#             print("Non Hate Dataset classification report ")
#             print(classification_report(y_val, nfold_predictions_test_non_hate))
#             print("Select Dataset classification report ")
#             print(classification_report(y_val, nfold_predictions_test_select_set))
            
            
#         for epo in range(0,5):
#             print(f"Running the {epo} :")
#         #     print(text_train)
#             adversial_non_hate_predicted_non_hate_dataset = []
#             adversial_actual_non_hate_dataset = []
#             adversial_select_predicted_non_hate_dataset = []
            
#             for train_idx in range(0,len(adversial_non_hate_nfold_predictions)):
#                 # capturing the predicted class as non-hate
#                 if adversial_non_hate_nfold_predictions[train_idx] == 1:
#                     adversial_non_hate_predicted_non_hate_dataset.append(adversial_non_hate_X_train[train_idx])
#                 # capturing the actual class as non - hate
#                 # if nfold_actual[train_idx] == 1:
#                 #     actual_non_hate_dataset.append(X_train[train_idx])

#             for train_idx in range(0,len(adversial_select_nfold_predictions)):
#                 # capturing the predicted class as non-hate
#                 if adversial_select_nfold_predictions[train_idx] == 1:
#                     adversial_select_predicted_non_hate_dataset.append(adversial_select_X_train[train_idx])
                    
#             adversial_non_hate_coef = adversial_non_hate_sgdc.coef_.ravel()
#             adversial_non_hate_feature_names = np.array(adversial_non_hate_feature_names)

#             adversial_select_coef = adversial_select_sgdc.coef_.ravel()
#             adversial_select_feature_names = np.array(adversial_select_feature_names)

#             inds = np.argsort(adversial_non_hate_coef)
#             # low = inds[:10]
#             high = inds[-10:]
#             important = np.hstack([ high])
#             print(adversial_non_hate_feature_names[important])
            
#             words_to_remove = adversial_non_hate_feature_names[important]
            
#             words_to_remove_regex = '|'.join(words_to_remove)
            
#             adversial_non_hate_text_train = adversial_non_hate_text_train.replace(to_replace=words_to_remove_regex, value='', regex=True)

#             inds = np.argsort(adversial_select_coef)
#             # low = inds[:10]
#             high = inds[-10:]
#             important = np.hstack([ high])
#             print(adversial_select_feature_names[important])
            
#             words_to_remove = adversial_select_feature_names[important]
            
#             words_to_remove_regex = '|'.join(words_to_remove)
            
#             adversial_select_text_train = adversial_select_text_train.replace(to_replace=words_to_remove_regex, value='', regex=True)
            
# #             print(adversial_text_train)
            
#             adversial_non_hate_vect = CountVectorizer()
#             adversial_non_hate_X_train = adversial_non_hate_vect.fit_transform(adversial_non_hate_text_train)
#             adversial_non_hate_X_val = adversial_non_hate_vect.transform(adversial_non_hate_text_val)
#             adversial_non_hate_X_val_non_hate = adversial_non_hate_vect.transform(adversial_non_hate_text_val_non_hate)
#             adversial_non_hate_X_val_select_set = adversial_non_hate_vect.transform(adversial_non_hate_text_val_select_set)

#             adversial_select_vect = CountVectorizer()
#             adversial_select_X_train = adversial_select_vect.fit_transform(adversial_select_text_train)
#             adversial_select_X_val = adversial_select_vect.transform(adversial_select_text_val)
#             adversial_select_X_val_non_hate = adversial_select_vect.transform(adversial_select_text_val_non_hate)
#             adversial_select_X_val_select_set = adversial_select_vect.transform(adversial_select_text_val_select_set)
            
#             adversial_non_hate_clf = SGDClassifier(loss='log')
#             adversial_select_clf = SGDClassifier(loss='log')
            
#             adversial_non_hate_feature_names = adversial_non_hate_vect.get_feature_names()
#             adversial_select_feature_names = adversial_select_vect.get_feature_names()
            
#             adversial_non_hate_sgdc = adversial_non_hate_clf.partial_fit(adversial_non_hate_X_train, adversial_non_hate_y_train,classes=[0,1])
            
#             adversial_non_hate_nfold_predictions = adversial_non_hate_sgdc.predict(adversial_non_hate_X_train)
#             adversial_non_hate_nfold_predictions_test = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val)
#             adversial_non_hate_nfold_predictions_test_non_hate = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val_non_hate)
#             adversial_non_hate_nfold_predictions_test_select_set = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val_select_set)

#             adversial_non_hate_nfold_sgdc = adversial_non_hate_clf.partial_fit(adversial_non_hate_X_train, adversial_non_hate_y_train,classes=[0,1])
#             adversial_select_sgdc = adversial_select_clf.partial_fit(adversial_select_X_train, adversial_select_y_train,classes=[0,1])
            
#             adversial_non_hate_nfold_predictions = adversial_non_hate_sgdc.predict(adversial_non_hate_X_train)
#             adversial_non_hate_nfold_predictions_test = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val)
#             adversial_non_hate_nfold_predictions_test_non_hate = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val_non_hate)
#             adversial_non_hate_nfold_predictions_test_select_set = adversial_non_hate_sgdc.predict(adversial_non_hate_X_val_select_set)

#             adversial_select_nfold_predictions = adversial_select_sgdc.predict(adversial_select_X_train)
#             adversial_select_nfold_predictions_test = adversial_select_sgdc.predict(adversial_select_X_val)
#             adversial_select_nfold_predictions_test_non_hate = adversial_select_sgdc.predict(adversial_select_X_val_non_hate)
#             adversial_select_nfold_predictions_test_select_set = adversial_select_sgdc.predict(adversial_select_X_val_select_set)

#             print("Adversial Non Hate Training classification report ")
#             print(classification_report(adversial_non_hate_y_train, adversial_non_hate_nfold_predictions))
#             print("Adversial Non Hate Testing classification report ")
#             print(classification_report(adversial_non_hate_y_val, adversial_non_hate_nfold_predictions_test))
#             print("Adversial Non Hate Non Hate Dataset classification report ")
#             print(classification_report(adversial_non_hate_y_val, adversial_non_hate_nfold_predictions_test_non_hate))
#             print("Adversial Non Hate Select Dataset classification report ")
#             print(classification_report(adversial_non_hate_y_val, adversial_non_hate_nfold_predictions_test_select_set))

#             print("Adversial Select Training classification report ")
#             print(classification_report(adversial_select_y_train, adversial_select_nfold_predictions))
#             print("Adversial Select Testing classification report ")
#             print(classification_report(adversial_select_y_val, adversial_select_nfold_predictions_test))
#             print("Adversial Select Non Hate Dataset classification report ")
#             print(classification_report(adversial_select_y_val, adversial_select_nfold_predictions_test_non_hate))
#             print("Adversial Select Select Dataset classification report ")
#             print(classification_report(adversial_select_y_val, adversial_select_nfold_predictions_test_select_set))
        


    elif selected_model == "cnn_gru":

        adversial_actual_tweets = raw_data.tweet

        select_set_words_list = []
        for idx in range(0,len(select_set)):
            value = {i for i in vocab if vocab[i]==select_set[idx]}
            select_set_words_list.append(value)

        print("Select set 5 sample words")
        select_set_words_list = [i for sub_list in select_set_words_list for i in sub_list]
        print(select_set_words_list[0:5])

        non_hate_set_words_list = []
        for idx in range(0,len(non_hate_set)):
            value = {i for i in vocab if vocab[i]==non_hate_set[idx]}
            non_hate_set_words_list.append(value)

        print("Non Hate set 5 sample words")
        non_hate_set_words_list = [i for sub_list in non_hate_set_words_list for i in sub_list]
        print(non_hate_set_words_list[0:5])

        adversial_non_hate_text = []
        # Iterate over the sentences
        for i, sentence in enumerate(adversial_actual_tweets):
            # Select 5 random words from the list
            random_words = rn.sample(non_hate_set_words_list, 50)
            # Append the random words to the sentence
            adversial_non_hate_text.append(sentence + ' ' + ' '.join(random_words))
            
        adversial_select_text = []
        # Iterate over the sentences
        for i, sentence in enumerate(adversial_actual_tweets):
            # Select 5 random words from the list
            random_words = rn.sample(select_set_words_list, 50)
            # Append the random words to the sentence
            adversial_select_text.append(sentence + ' ' + ' '.join(random_words))

        print(adversial_actual_tweets[1])
        print(adversial_non_hate_text[1])
        print(adversial_select_text[1])

        print(len(adversial_actual_tweets))
        print(len(adversial_non_hate_text))
        print(len(adversial_select_text))

        # Assuming that all three variables have the same length
        adversial_non_hate_text = pd.concat([pd.Series(adversial_actual_tweets), pd.Series(adversial_non_hate_text)], ignore_index=True)
        adversial_select_text = pd.concat([pd.Series(adversial_actual_tweets), pd.Series(adversial_select_text)], ignore_index=True)
                                    
        # adversial_text = pd.concat([pd.Series(adversial_actual_tweets), pd.Series(adversial_non_hate_text), pd.Series(adversial_select_text)], ignore_index=True)
        adversial_target = pd.concat([pd.Series(raw_data['class']), pd.Series(raw_data['class'])], ignore_index=True)

        print(adversial_non_hate_text.head())
        print(adversial_select_text.head())

        print(len(adversial_non_hate_text))
        print(len(adversial_select_text))

        adversial_non_hate_M = get_word_vocab(adversial_non_hate_text, sys_out, word_norm_option)
        adversial_non_hate_M0 = adversial_non_hate_M[0]

        adversial_select_M = get_word_vocab(adversial_select_text, sys_out, word_norm_option)
        adversial_select_M0 = adversial_select_M[0]
        

        # temp_cnt = 0
        # for wrd in range(0,15):
        #     try:
        #         random_five_words_data.loc[wrd,"word"] = str(random_words[wrd])
        #         random_five_words_data.loc[wrd,"index"] = M[1][str(random_words[wrd])]
        #     except Exception as e:
        #         print(e)
        # # random_five_words_data = M[2]
        # print(random_five_words_data)
        
        print(len(M0))
        print("Converted the data to matrix")
        print(M[0][0])

        print(len(adversial_non_hate_M0))
        print("Converted the Non Hate Adversial data to matrix")
        print(adversial_non_hate_M[0][0])

        print(len(adversial_select_M0))
        print("Converted the Select Adversial data to matrix")
        print(adversial_select_M[0][0])

        length_less_than_50 = 0
        length_between_50_and_100 = 0
        length_grater_100 = 0
        length_emb_word = []
        for idx in range(0,len(M0)):
            length_emb_word.append(len(M0[idx]))
            # if length_emb_word < 50:
            #     length_less_than_50 = length_less_than_50 + 1
            # elif length_emb_word >= 50 and length_emb_word < 100:
            #     length_between_50_and_100 = length_between_50_and_100 + 1
            # else:
            #     length_grater_100 += 1
        
        adversial_non_hate_length_emb_word = []
        for idx in range(0,len(adversial_non_hate_M0)):
            adversial_non_hate_length_emb_word.append(len(adversial_non_hate_M0[idx]))

        adversial_select_length_emb_word = []
        for idx in range(0,len(adversial_select_M0)):
            adversial_select_length_emb_word.append(len(adversial_select_M0[idx]))

        print("Embedding Summary : ")
        print(max(length_emb_word))
        print("Adversial Non Hate Embedding Summary : ")
        print(max(adversial_non_hate_length_emb_word))
        print("Adversial Select Embedding Summary : ")
        print(max(adversial_select_length_emb_word))
        # print(length_less_than_50)
        # print(length_between_50_and_100)
        # print(length_grater_100)
        print(M[0][0])
        print(adversial_non_hate_M0[0][0])
        print(adversial_select_M0[0][0])


        # print(M[1])
        # print("model matrix")
        # print(M0)

        pretrained_word_matrix = None
        if pretrained_embedding_models is not None:
            pretrained_word_matrix = build_pretrained_embedding_matrix(M[1],
                                                                    pretrained_embedding_models,
                                                                    expected_embedding_dim,
                                                                    randomize_strategy)
        word_dist_matrix = None
        if word_dist_features_file is not None:
            word_dist_matrix = build_word_dist_matrix(M[1],
                                                    word_dist_features_file)

        # split the dataset into two parts, 0.75 for train and 0.25 for testing
        if 'ds' in raw_data.columns:
            print("ds is in the columns")
            col_datasource=raw_data['ds']
        else:
            col_datasource=raw_data[raw_data.columns[0]]
        X_train_data, X_test_data, y_train, y_test, ds_train, ds_test, index_train, index_test=\
            train_test_split(M0, raw_data['class'], col_datasource,
                            list(raw_data.index.values),
                            test_size=0.25,
                            random_state=42)
        

        # print("Checking the test data set ")
        # print(X_test_data)

        # X_train_data, X_test_data, y_train, y_test, ds_train, ds_test, index_train, index_test=\
        #     train_test_split(M0, raw_data['class'], col_datasource,
        #                      list(raw_data.index.values),
        #                      test_size=0.25,
        #                      random_state=42, shuffle=True)

        print("Split the data into train and test done")

        accepted_ds_tags = None
        if print_scores_per_class:
            accepted_ds_tags = ["w"]

        # using mixed data?
        if use_mixed_data:
            mixed_data_folder=input_data_file[0:input_data_file.rfind("/")]
            mixed_data_file=mixed_data_folder+"/labeled_data_all_mixed.csv"
            mixed_data = pd.read_csv(mixed_data_file, sep=',', encoding="utf-8")
            print("mixed_data : ")
            # print(mixed_data.tail())
            MX = get_word_vocab(mixed_data.tweet, sys_out, word_norm_option)
            # M=self.feature_scale(M)
            MX0 = MX[0]

            # split the dataset into two parts, 0.75 for train and 0.25 for testing
            MX_X_train_data, MX_X_test_data, MX_y_train, MX_y_test, MX_ds_train, MX_ds_test = \
                train_test_split(MX0, mixed_data['class'],
                                mixed_data['ds'],
                                test_size=0.25,
                                random_state=42)
            X_train_data=numpy.concatenate((X_train_data, MX_X_train_data))
            X_test_data = numpy.concatenate((X_test_data, MX_X_test_data))
            y_train = y_train.append(MX_y_train, ignore_index=True) #numpy.concatenate((y_train, MX_y_train))
            y_test = y_test.append(MX_y_test, ignore_index=True)
            ds_train = ds_train.append(MX_ds_train, ignore_index=True)
            ds_test = ds_test.append(MX_ds_test, ignore_index=True)

        y_train = y_train.astype(int)
        
        y_test = y_test.astype(int)

        X_test_non_hate = []
        X_test_data_common = []
        

    #     print(X_test_data[0])
        for i in range(len(X_test_data)):
            X_test_non_hate.append(X_test_data[i]+rn.sample(non_hate_set, 50))
        
        for i in range(len(X_test_data)):
            X_test_data_common.append(X_test_data[i]+rn.sample(select_set, 50))

        print(X_test_non_hate[0])
        print(X_test_data_common[0])
        

    #     def most_frequent(List):
    #         occurence_count = Counter(List)
    #         most_frequent_words = occurence_count.most_common(10)
    #         print("most_frequent_words")
    #         print(most_frequent_words)
    #         most_frequent_list = [most_frequent_words[0][0],most_frequent_words[1][0],most_frequent_words[2][0],most_frequent_words[3][0],most_frequent_words[4][0],most_frequent_words[5][0],most_frequent_words[6][0],most_frequent_words[7][0],most_frequent_words[8][0],most_frequent_words[9][0]]
    #         return most_frequent_list
        
    #     list_of_words = []
    #     for idx in range(0,len(X_test_non_hate)):
    # #         print(X_test_non_hate[idx])
    #         list_of_words.append(X_test_non_hate[idx])
        
    #     print("Total length of words :")
    #     list_of_words = list(chain(*list_of_words))
    #     print(len(list_of_words))
    #     most_frequent_list = most_frequent(list_of_words)
    #     print("Most frequent index list :")
    #     print(most_frequent_list)
        
    #     most_frequent_words_list = []
    #     for idx in range(0,len(most_frequent_list)):
    #         value = {i for i in vocab if vocab[i]==most_frequent_list[idx]}
    #         most_frequent_words_list.append(value)
    #     print("Most frequent words list :")
    #     print(most_frequent_words_list)

        X_train_data = sequence.pad_sequences(X_train_data, maxlen=MAX_SEQUENCE_LENGTH)
        X_test_data = sequence.pad_sequences(X_test_data, maxlen=MAX_SEQUENCE_LENGTH)
        X_test_non_hate = sequence.pad_sequences(X_test_non_hate, maxlen=MAX_SEQUENCE_LENGTH)
        X_test_data_common = sequence.pad_sequences(X_test_data_common, maxlen=MAX_SEQUENCE_LENGTH)

        Adversial_non_hate_X_train_data = sequence.pad_sequences(adversial_non_hate_M0, maxlen=MAX_SEQUENCE_LENGTH)
        Adversial_select_X_train_data = sequence.pad_sequences(adversial_select_M0, maxlen=MAX_SEQUENCE_LENGTH)
        
        # X_test_love_data = X_test_data.copy()
        # X_test_five_words_data = X_test_data.copy()

        print(X_test_data[0])
        # print(X_test_love_data[0])
        print(X_test_data_common[0])

        ## Adding the love to end of the embedding
        # print("Adding Love")

        # def rotate(l, n):
        #     return l[n:] + l[:n]

        # for idx in range(0,len(X_test_love_data)):
            
        #     # X_test_love_data[idx] = np.append(X_test_love_data[idx], 10176)
        #     current_array = X_test_love_data[idx] 
        #     # print(current_array)
        #     current_array = rotate(current_array,1)
        #     current_array = np.append(current_array,love_index)
        #     # print(current_array)
        #     X_test_love_data[idx] = current_array
        # print("appending 5 words to vector")

        
        # # random_five_words_data = [item for item in random_five_words_data if not(math.isnan(item)) == True]
        # # print(random_five_words_data)
        # append_indices = []
        # for lst_idx in range(0,len(random_five_words_data)):
        #     print(random_five_words_data.loc[lst_idx,'index'])
        #     if math.isnan(random_five_words_data.loc[lst_idx,'index']):
        #         print("True")
        #     else:
        #         if len(append_indices) < 5:
        #             print(random_five_words_data.loc[lst_idx,'word'])
        #             append_indices.append(random_five_words_data.loc[lst_idx,'index'])

        # for idx in range(0,len(X_test_five_words_data)):
            
        #     # X_test_love_data[idx] = np.append(X_test_love_data[idx], 10176)
        #     current_array = X_test_five_words_data[idx] 
        #     # print(current_array)
        #     # print(current_array.shape)
        #     ite = 0
        #     while ite < 5:
        #         current_array = rotate(current_array,1)
        #         current_array = np.append(current_array,append_indices[ite])
        #         ite = ite+1
        #     # print(current_array)
        #     # print(current_array.shape)
        #     # print(len(random_five_words_data))
        #     # print(random_five_words_data)
        #     # print(random_five_words_data[0]["index"])
        #     # print(random_five_words_data[1]["index"])
        #     # print(random_five_words_data[2]["index"])
        #     # print(random_five_words_data[3]["index"])
        #     # print(random_five_words_data[4])
        #     # current_array = np.append(current_array,append_indices[0],append_indices[1],append_indices[2],append_indices[3],append_indices[4])
        #     # print(current_array)
        #     # print(current_array.shape)
        #     X_test_five_words_data[idx] = current_array

        # print(X_test_love_data[0])
        # print(X_test_five_words_data[0])
        # # print(X_test_data.shape)
        # # print(len(X_test_data))
        # print(X_test_data[0])

        output_data_stats(X_train_data, y_train)
        print("Adversial Non Hate Output stats")
        output_data_stats(Adversial_non_hate_X_train_data, adversial_target)
        print("Adversial Select Output stats")
        output_data_stats(Adversial_select_X_train_data, adversial_target)
        # exit(0)

        grid_search_dnn(dataset_name, sys_out, model_descriptor,
                        CPUS, 5,
                        X_train_data,
                        y_train, X_test_data, y_test,X_test_non_hate,X_test_data_common,Adversial_non_hate_X_train_data,Adversial_select_X_train_data,adversial_target,vocab, index_train, index_test,
                        len(M[1]), pretrained_word_matrix, word_dist_matrix,
                        ds_train, ds_test, accepted_ds_tags)
        print("complete {}".format(datetime.datetime.now()))

    else:
        # set of stop words 
        eng_stop_words = stopwords.words("english")
        tweet_without_stopwords = raw_data["tweet_without_stopwords"] = raw_data["tweet"].apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stop_words)]))


        text = tweet_without_stopwords.values
        labels = raw_data['class'].values

        val_ratio = 0.25
        # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
        batch_size = 16

        # Indices of the train and validation splits stratified by labels
        # train_idx, val_idx = train_test_split(
        #     np.arange(len(labels)),
        #     test_size = val_ratio,
        #     shuffle = True,
        #     stratify = labels,
        #     random_state = 42)
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size = val_ratio,
            shuffle = True,
            stratify = labels)

        training_data = raw_data.loc[train_idx,]
        testing_data = raw_data.loc[val_idx,]

        training_data.reset_index(inplace=True,drop=True)
        testing_data.reset_index(inplace=True,drop=True)

        training_data["tweet_select_attack"] = ""
        training_data["tweet_non_hate_attack"] = ""
        testing_data["tweet_select_attack"] = ""
        testing_data["tweet_non_hate_attack"] = ""


        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case = True
            )
        
        adversial_non_hate_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case = True
            )
        
        adversial_select_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case = True
            )

        for i in range(0,len(training_data)):
            training_data.loc[i,"tweet_select_attack"] = training_data.loc[i,"tweet_without_stopwords"] +" "+ " ".join(random.choices(filtered_sentence, k=50))
            training_data.loc[i,"tweet_non_hate_attack"] = training_data.loc[i,"tweet_without_stopwords"] +" "+ " ".join(random.choices(non_hate_words_set, k=50))

        for i in range(0,len(testing_data)):
        #     testing_data.loc[i,"tweet_select_attack"] = testing_data.loc[i,"tweet"] +" "+ " ".join(rn.sample(filtered_sentence, 10))
            testing_data.loc[i,"tweet_select_attack"] = testing_data.loc[i,"tweet_without_stopwords"] +" "+ " ".join(random.choices(filtered_sentence, k=50))
            testing_data.loc[i,"tweet_non_hate_attack"] = testing_data.loc[i,"tweet_without_stopwords"] +" "+ " ".join(random.choices(non_hate_words_set, k=50))
        
        print("Training Appending : ")
        print(training_data.loc[0,"tweet_without_stopwords"])
        print(training_data.loc[0,"tweet_select_attack"])
        print(training_data.loc[0,"tweet_non_hate_attack"])

        print("Testing Appending : ")
        print(testing_data.loc[0,"tweet_without_stopwords"])
        print(testing_data.loc[0,"tweet_select_attack"])
        print(testing_data.loc[0,"tweet_non_hate_attack"])

        # Concatenate the two columns and reset the index
        combined_non_hate_text = pd.concat([training_data['tweet_without_stopwords'],training_data["tweet_non_hate_attack"]], ignore_index=True)
        combined_select_text = pd.concat([training_data['tweet_without_stopwords'], training_data["tweet_select_attack"]], ignore_index=True)
        combined_class = pd.concat([training_data['class'], training_data["class"]], ignore_index=True)

        # Convert the resulting Series to a list
        combined_non_hate_text = combined_non_hate_text.tolist()
        combined_select_text = combined_select_text.tolist()
        combined_class = combined_class.tolist()

        print(len(training_data))
        print(len(combined_non_hate_text))
        print(len(combined_select_text))

        training_data.set_index(train_idx, inplace=True)
        testing_data.set_index(val_idx, inplace=True)

        def print_rand_sentence():
            '''Displays the tokens and respective IDs of a random text sample'''
            index = 1
            table = np.array([tokenizer.tokenize(text[index]), 
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text[index]))]).T
            print(tabulate(table,
                        headers = ['Tokens', 'Token IDs'],
                        tablefmt = 'fancy_grid'))

        print_rand_sentence()

        token_id = []
        attention_masks = []

        select_token_id = []
        select_attention_masks = []

        non_hate_token_id = []
        non_hate_attention_masks = []

        adversial_non_hate_token_id = []
        adversial_non_hate_attention_masks = []

        adversial_select_token_id = []
        adversial_select_attention_masks = []

        def preprocessing(input_text, tokenizer):
            '''
            Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
            - input_ids: list of token ids
            - token_type_ids: list of token type ids
            - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
            '''
            return tokenizer.encode_plus(
                                input_text,
                                add_special_tokens = True,
                                max_length = 100,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'pt'
                        )


        for sample in text:
            encoding_dict = preprocessing(sample, tokenizer)
            token_id.append(encoding_dict['input_ids']) 
            attention_masks.append(encoding_dict['attention_mask'])


        token_id = torch.cat(token_id, dim = 0)
        attention_masks = torch.cat(attention_masks, dim = 0)
        labels = torch.tensor(labels)

        select_text = testing_data['tweet_select_attack'].values
        select_labels = testing_data['class'].values

        for sample in select_text:
            select_encoding_dict = preprocessing(sample, tokenizer)
            select_token_id.append(select_encoding_dict['input_ids']) 
            select_attention_masks.append(select_encoding_dict['attention_mask'])


        select_token_id = torch.cat(select_token_id, dim = 0)
        select_attention_masks = torch.cat(select_attention_masks, dim = 0)
        select_labels = torch.tensor(select_labels)

        non_hate_text = testing_data['tweet_non_hate_attack'].values
        non_hate_labels = testing_data['class'].values

        for sample in non_hate_text:
            non_hate_encoding_dict = preprocessing(sample, tokenizer)
            non_hate_token_id.append(non_hate_encoding_dict['input_ids']) 
            non_hate_attention_masks.append(non_hate_encoding_dict['attention_mask'])


        non_hate_token_id = torch.cat(non_hate_token_id, dim = 0)
        non_hate_attention_masks = torch.cat(non_hate_attention_masks, dim = 0)
        non_hate_labels = torch.tensor(non_hate_labels)

        # non_hate_text = testing_data['tweet_non_hate_attack'].values
        # non_hate_labels = testing_data['class'].values

        for sample in combined_non_hate_text:
            adversial_encoding_dict = preprocessing(sample, adversial_non_hate_tokenizer)
            adversial_non_hate_token_id.append(adversial_encoding_dict['input_ids']) 
            adversial_non_hate_attention_masks.append(adversial_encoding_dict['attention_mask'])


        adversial_non_hate_token_id = torch.cat(adversial_non_hate_token_id, dim = 0)
        adversial_non_hate_attention_masks = torch.cat(adversial_non_hate_attention_masks, dim = 0)
        adversial_non_hate_labels = torch.tensor(combined_class)

        for sample in combined_select_text:
            adversial_encoding_dict = preprocessing(sample, adversial_select_tokenizer)
            adversial_select_token_id.append(adversial_encoding_dict['input_ids']) 
            adversial_select_attention_masks.append(adversial_encoding_dict['attention_mask'])


        adversial_select_token_id = torch.cat(adversial_select_token_id, dim = 0)
        adversial_select_attention_masks = torch.cat(adversial_select_attention_masks, dim = 0)
        adversial_select_labels = torch.tensor(combined_class)

        print(token_id[1])
        print(select_token_id[1])
        print(non_hate_token_id[1])

        # print(token_id[1])

        def print_rand_sentence_encoding():
            '''Displays tokens, token IDs and attention mask of a random text sample'''
        #     index = random.randint(0, len(text) - 1)
            index = 1
            tokens = tokenizer.tokenize(tokenizer.decode(token_id[index]))
            token_ids = [i.numpy() for i in token_id[index]]
            attention = [i.numpy() for i in attention_masks[index]]

            table = np.array([tokens, token_ids, attention]).T
            print(tabulate(table, 
                        headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                        tablefmt = 'fancy_grid'))

        print_rand_sentence_encoding()

        # val_ratio = 0.2
        # # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
        # batch_size = 16

        # # Indices of the train and validation splits stratified by labels
        # train_idx, val_idx = train_test_split(
        #     np.arange(len(labels)),
        #     test_size = val_ratio,
        #     shuffle = True,
        #     stratify = labels)

        # Train and validation sets
        train_set = TensorDataset(token_id[train_idx], 
                                attention_masks[train_idx], 
                                labels[train_idx])

        val_set = TensorDataset(token_id[val_idx], 
                                attention_masks[val_idx], 
                                labels[val_idx])

        select_set = TensorDataset(select_token_id, 
                        select_attention_masks, 
                        select_labels)

        non_hate_set = TensorDataset(non_hate_token_id, 
                        non_hate_attention_masks, 
                        non_hate_labels)

        adversial_non_hate_set = TensorDataset(adversial_non_hate_token_id, 
                        adversial_non_hate_attention_masks, 
                        adversial_non_hate_labels)
        
        adversial_select_set = TensorDataset(adversial_select_token_id, 
                        adversial_select_attention_masks, 
                        adversial_select_labels)

        # Prepare DataLoader
        train_dataloader = DataLoader(
                    train_set,
                    sampler = RandomSampler(train_set),
                    batch_size = batch_size
                )

        validation_dataloader = DataLoader(
                    val_set,
                    sampler = SequentialSampler(val_set),
                    batch_size = batch_size
                )

        select_validation_dataloader = DataLoader(
                    select_set,
                    sampler = SequentialSampler(select_set),
                    batch_size = batch_size
                )

        non_hate_validation_dataloader = DataLoader(
                    non_hate_set,
                    sampler = SequentialSampler(non_hate_set),
                    batch_size = batch_size
                )

        adversial_non_hate_validation_dataloader = DataLoader(
                    adversial_non_hate_set,
                    sampler = SequentialSampler(adversial_non_hate_set),
                    batch_size = batch_size
                )
        
        adversial_select_validation_dataloader = DataLoader(
                    adversial_select_set,
                    sampler = SequentialSampler(adversial_select_set),
                    batch_size = batch_size
                )

        def b_tp(preds, labels):
            '''Returns True Positives (TP): count of correct predictions of actual class 1'''
            return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

        def b_fp(preds, labels):
            '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
            return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

        def b_tn(preds, labels):
            '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
            return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

        def b_fn(preds, labels):
            '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
            return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

        def b_metrics(preds, labels):
            '''
            Returns the following metrics:
            - accuracy    = (TP + TN) / N
            - precision   = TP / (TP + FP)
            - recall      = TP / (TP + FN)
            - specificity = TN / (TN + FP)
            '''
            preds = np.argmax(preds, axis = 1).flatten()
            labels = labels.flatten()
            tp = b_tp(preds, labels)
            tn = b_tn(preds, labels)
            fp = b_fp(preds, labels)
            fn = b_fn(preds, labels)
            b_accuracy = (tp + tn) / len(labels)
            b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
            b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
            b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
            return b_accuracy, b_precision, b_recall, b_specificity

        
        
        # Load the BertForSequenceClassification model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
        )

        adversial_non_hate_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
        )

        adversial_select_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
        )

        # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr = 5e-5,
                                    eps = 1e-08
                                    )

        adversial_non_hate_optimizer = torch.optim.AdamW(adversial_non_hate_model.parameters(), 
                                    lr = 5e-5,
                                    eps = 1e-08
                                    )


        adversial_select_optimizer = torch.optim.AdamW(adversial_select_model.parameters(), 
                                            lr = 5e-5,
                                            eps = 1e-08
                                            )

        # Run on GPU
        model.cuda()
        adversial_non_hate_model.cuda()
        adversial_select_model.cuda()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
        epochs = 10

        for _ in trange(epochs, desc = 'Epoch'):
            
            # ========== Training ==========
            
            # Set model to training mode
            model.train()
            
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()
                # Forward pass
                train_output = model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask, 
                                    labels = b_labels)
                # Backward pass
                train_output.loss.backward()
                optimizer.step()
                # Update tracking variables
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            # # ==========Adversial Non Hate Training ==========
            
            # Set model to training mode
            adversial_non_hate_model.train()
            
            # Tracking variables
            adversial_non_hate_tr_loss = 0
            adversial_non_hate_nb_tr_examples, adversial_non_hate_nb_tr_steps = 0, 0

            for step, batch in enumerate(adversial_non_hate_validation_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                adversial_non_hate_optimizer.zero_grad()
                # Forward pass
                adversial_train_output = adversial_non_hate_model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask, 
                                    labels = b_labels)
                # Backward pass
                adversial_train_output.loss.backward()
                adversial_non_hate_optimizer.step()
                # Update tracking variables
                adversial_non_hate_tr_loss += adversial_train_output.loss.item()
                adversial_non_hate_nb_tr_examples += b_input_ids.size(0)
                adversial_non_hate_nb_tr_steps += 1

            # # ==========Adversial Select Training ==========
            
            # Set model to training mode
            adversial_select_model.train()
            
            # Tracking variables
            adversial_select_tr_loss = 0
            adversial_select_nb_tr_examples, adversial_select_nb_tr_steps = 0, 0

            for step, batch in enumerate(adversial_select_validation_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                adversial_select_optimizer.zero_grad()
                # Forward pass
                adversial_train_output = adversial_select_model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask, 
                                    labels = b_labels)
                # Backward pass
                adversial_train_output.loss.backward()
                adversial_select_optimizer.step()
                # Update tracking variables
                adversial_select_tr_loss += adversial_train_output.loss.item()
                adversial_select_nb_tr_examples += b_input_ids.size(0)
                adversial_select_nb_tr_steps += 1

            # ========== Validation (Original Test) ==========

            # Set model to evaluation mode
            model.eval()

            # Tracking variables 
            val_accuracy = []
            val_precision = []
            val_recall = []
            val_specificity = []
            all_labels = []
            all_preds = []

            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    # Forward pass
                    eval_output = model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                all_labels.extend(label_ids.flatten().tolist())
                all_preds.extend(np.round(np.argmax(logits, axis=1)).flatten().tolist())

                # Calculate validation metrics
                b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
                val_accuracy.append(b_accuracy)
                # Update precision only when (tp + fp) !=0; ignore nan
                if b_precision != 'nan': val_precision.append(b_precision)
                # Update recall only when (tp + fn) !=0; ignore nan
                if b_recall != 'nan': val_recall.append(b_recall)
                # Update specificity only when (tn + fp) !=0; ignore nan
                if b_specificity != 'nan': val_specificity.append(b_specificity)
                    
            # # Set model to evaluation mode
            # adversial_non_hate_model.eval()

            # # Tracking variables 
            # adversial_non_hate_val_accuracy = []
            # adversial_non_hate_val_precision = []
            # adversial_non_hate_val_recall = []
            # adversial_non_hate_val_specificity = []
            # adversial_non_hate_all_labels = []
            # adversial_non_hate_all_preds = []

            # for batch in validation_dataloader:
            #     batch = tuple(t.to(device) for t in batch)
            #     b_input_ids, b_input_mask, b_labels = batch
            #     with torch.no_grad():
            #         # Forward pass
            #         eval_output = adversial_non_hate_model(b_input_ids, 
            #                             token_type_ids = None, 
            #                             attention_mask = b_input_mask)
            #     logits = eval_output.logits.detach().cpu().numpy()
            #     label_ids = b_labels.to('cpu').numpy()

            #     adversial_non_hate_all_labels.extend(label_ids.flatten().tolist())
            #     adversial_non_hate_all_preds.extend(np.round(np.argmax(logits, axis=1)).flatten().tolist())

            #     # Calculate validation metrics
            #     b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            #     adversial_non_hate_val_accuracy.append(b_accuracy)
            #     # Update precision only when (tp + fp) !=0; ignore nan
            #     if b_precision != 'nan': adversial_non_hate_val_precision.append(b_precision)
            #     # Update recall only when (tp + fn) !=0; ignore nan
            #     if b_recall != 'nan': adversial_non_hate_val_recall.append(b_recall)
            #     # Update specificity only when (tn + fp) !=0; ignore nan
            #     if b_specificity != 'nan': adversial_non_hate_val_specificity.append(b_specificity)

            # ========== Select Validation ==========

            # Set model to evaluation mode
            model.eval()

            # Tracking variables 
            select_val_accuracy = []
            select_val_precision = []
            select_val_recall = []
            select_val_specificity = []
            select_all_labels = []
            select_all_preds = []

            for batch in select_validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    # Forward pass
                    eval_output = model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                select_all_labels.extend(label_ids.flatten().tolist())
                select_all_preds.extend(np.round(np.argmax(logits, axis=1)).flatten().tolist())

                # Calculate validation metrics
                b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
                select_val_accuracy.append(b_accuracy)
                # Update precision only when (tp + fp) !=0; ignore nan
                if b_precision != 'nan': select_val_precision.append(b_precision)
                # Update recall only when (tp + fn) !=0; ignore nan
                if b_recall != 'nan': select_val_recall.append(b_recall)
                # Update specificity only when (tn + fp) !=0; ignore nan
                if b_specificity != 'nan': select_val_specificity.append(b_specificity)
                    
            # Set model to evaluation mode
            adversial_select_model.eval()

            # Tracking variables 
            adversial_select_val_accuracy = []
            adversial_select_val_precision = []
            adversial_select_val_recall = []
            adversial_select_val_specificity = []
            adversial_select_all_labels = []
            adversial_select_all_preds = []

            for batch in select_validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    # Forward pass
                    eval_output = adversial_select_model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                adversial_select_all_labels.extend(label_ids.flatten().tolist())
                adversial_select_all_preds.extend(np.round(np.argmax(logits, axis=1)).flatten().tolist())

                # Calculate validation metrics
                b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
                adversial_select_val_accuracy.append(b_accuracy)
                # Update precision only when (tp + fp) !=0; ignore nan
                if b_precision != 'nan': adversial_select_val_precision.append(b_precision)
                # Update recall only when (tp + fn) !=0; ignore nan
                if b_recall != 'nan': adversial_select_val_recall.append(b_recall)
                # Update specificity only when (tn + fp) !=0; ignore nan
                if b_specificity != 'nan': adversial_select_val_specificity.append(b_specificity)

            # ========== Non Hate Validation ==========

            # Set model to evaluation mode
            model.eval()

            # Tracking variables 
            non_hate_val_accuracy = []
            non_hate_val_precision = []
            non_hate_val_recall = []
            non_hate_val_specificity = []
            non_hate_all_labels = []
            non_hate_all_preds = []

            for batch in non_hate_validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    # Forward pass
                    eval_output = model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                non_hate_all_labels.extend(label_ids.flatten().tolist())
                non_hate_all_preds.extend(np.round(np.argmax(logits, axis=1)).flatten().tolist())

                # Calculate validation metrics
                b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
                non_hate_val_accuracy.append(b_accuracy)
                # Update precision only when (tp + fp) !=0; ignore nan
                if b_precision != 'nan': non_hate_val_precision.append(b_precision)
                # Update recall only when (tp + fn) !=0; ignore nan
                if b_recall != 'nan': non_hate_val_recall.append(b_recall)
                # Update specificity only when (tn + fp) !=0; ignore nan
                if b_specificity != 'nan': non_hate_val_specificity.append(b_specificity)
                    
            # Set model to evaluation mode
            adversial_non_hate_model.eval()

            # Tracking variables 
            adversial_non_hate_val_accuracy = []
            adversial_non_hate_val_precision = []
            adversial_non_hate_val_recall = []
            adversial_non_hate_val_specificity = []
            adversial_non_hate_all_labels = []
            adversial_non_hate_all_preds = []

            for batch in non_hate_validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    # Forward pass
                    eval_output = adversial_non_hate_model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                adversial_non_hate_all_labels.extend(label_ids.flatten().tolist())
                adversial_non_hate_all_preds.extend(np.round(np.argmax(logits, axis=1)).flatten().tolist())

                # Calculate validation metrics
                b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
                adversial_non_hate_val_accuracy.append(b_accuracy)
                # Update precision only when (tp + fp) !=0; ignore nan
                if b_precision != 'nan': adversial_non_hate_val_precision.append(b_precision)
                # Update recall only when (tp + fn) !=0; ignore nan
                if b_recall != 'nan': adversial_non_hate_val_recall.append(b_recall)
                # Update specificity only when (tn + fp) !=0; ignore nan
                if b_specificity != 'nan': adversial_non_hate_val_specificity.append(b_specificity)

            print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
            print('\n\t - Adversial Non Hate Train loss: {:.4f}'.format(adversial_non_hate_tr_loss / adversial_non_hate_nb_tr_steps))
            print('\n\t - Adversial Select Train loss: {:.4f}'.format(adversial_select_tr_loss / adversial_select_nb_tr_steps))
            
            
            print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
            print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
            print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
            print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')
            
            # print('\t - Adversial Validation Accuracy: {:.4f}'.format(sum(adversial_val_accuracy)/len(adversial_val_accuracy)))
            # print('\t - Adversial Validation Precision: {:.4f}'.format(sum(adversial_val_precision)/len(adversial_val_precision)) if len(adversial_val_precision)>0 else '\t - Adversial Validation Precision: NaN')
            # print('\t - Adversial Validation Recall: {:.4f}'.format(sum(adversial_val_recall)/len(adversial_val_recall)) if len(adversial_val_recall)>0 else '\t - Adversial Validation Recall: NaN')
            # print('\t - Adversial Validation Specificity: {:.4f}\n'.format(sum(adversial_val_specificity)/len(adversial_val_specificity)) if len(adversial_val_specificity)>0 else '\t - Adversial Validation Specificity: NaN')

            print('\t - Non Hate Validation Accuracy: {:.4f}'.format(sum(non_hate_val_accuracy)/len(non_hate_val_accuracy)))
            print('\t - Non Hate Validation Precision: {:.4f}'.format(sum(non_hate_val_precision)/len(non_hate_val_precision)) if len(non_hate_val_precision)>0 else '\t - Non Hate Validation Precision: NaN')
            print('\t - Non Hate Validation Recall: {:.4f}'.format(sum(non_hate_val_recall)/len(non_hate_val_recall)) if len(non_hate_val_recall)>0 else '\t - Non Hate Validation Recall: NaN')
            print('\t - Non Hate Validation Specificity: {:.4f}\n'.format(sum(non_hate_val_specificity)/len(non_hate_val_specificity)) if len(non_hate_val_specificity)>0 else '\t - Non Hate Validation Specificity: NaN')

            print('\t - Select Validation Accuracy: {:.4f}'.format(sum(select_val_accuracy)/len(select_val_accuracy)))
            print('\t - Select Validation Precision: {:.4f}'.format(sum(select_val_precision)/len(select_val_precision)) if len(select_val_precision)>0 else '\t - Select Validation Precision: NaN')
            print('\t - Select Validation Recall: {:.4f}'.format(sum(select_val_recall)/len(select_val_recall)) if len(select_val_recall)>0 else '\t - Select Validation Recall: NaN')
            print('\t - Select Validation Specificity: {:.4f}\n'.format(sum(select_val_specificity)/len(select_val_specificity)) if len(select_val_specificity)>0 else '\t - Select Validation Specificity: NaN')
            
            print('\t - Adversial Non Hate Validation Accuracy: {:.4f}'.format(sum(adversial_non_hate_val_accuracy)/len(adversial_non_hate_val_accuracy)))
            print('\t - Adversial Non Hate Validation Precision: {:.4f}'.format(sum(adversial_non_hate_val_precision)/len(adversial_non_hate_val_precision)) if len(adversial_non_hate_val_precision)>0 else '\t - Adversial Non Hate Validation Precision: NaN')
            print('\t - Adversial Non Hate Validation Recall: {:.4f}'.format(sum(adversial_non_hate_val_recall)/len(adversial_non_hate_val_recall)) if len(adversial_non_hate_val_recall)>0 else '\t - Adversial Non Hate Validation Recall: NaN')
            print('\t - Adversial Non Hate Validation Specificity: {:.4f}\n'.format(sum(adversial_non_hate_val_specificity)/len(adversial_non_hate_val_specificity)) if len(adversial_non_hate_val_specificity)>0 else '\t - Adversial Non Hate Validation Specificity: NaN')

            print('\t - Adversial Select Validation Accuracy: {:.4f}'.format(sum(adversial_select_val_accuracy)/len(adversial_select_val_accuracy)))
            print('\t - Adversial Select Validation Precision: {:.4f}'.format(sum(adversial_select_val_precision)/len(adversial_select_val_precision)) if len(adversial_select_val_precision)>0 else '\t - Adversial Select Validation Precision: NaN')
            print('\t - Adversial Select Validation Recall: {:.4f}'.format(sum(adversial_select_val_recall)/len(adversial_select_val_recall)) if len(adversial_select_val_recall)>0 else '\t - Adversial Select Validation Recall: NaN')
            print('\t - Adversial Select Validation Specificity: {:.4f}\n'.format(sum(adversial_select_val_specificity)/len(adversial_select_val_specificity)) if len(adversial_select_val_specificity)>0 else '\t - Adversial Select Validation Specificity: NaN')

            

            print("Validation Classification Report :")
            print(classification_report(all_labels, all_preds))

            print("Select Validation Classification Report :")
            print(classification_report(select_all_labels, select_all_preds))

            print("Non Hate Validation Classification Report :")
            print(classification_report(non_hate_all_labels, non_hate_all_preds))
            
            # print("Adversial Validation Classification Report :")
            # print(classification_report(adversial_all_labels, adversial_all_preds))

            print(" Adversial Select Validation Classification Report :")
            print(classification_report(adversial_select_all_labels, adversial_select_all_preds))

            print(" Adversial Non Hate Validation Classification Report :")
            print(classification_report(adversial_non_hate_all_labels, adversial_non_hate_all_preds))








def cross_eval_dnn(dataset_name, outfolder, model_descriptor: str,
                   cpus, nfold, X_data, y_data,
                   embedding_layer_max_index, pretrained_embedding_matrix=None,
                   instance_data_source_tags=None, accepted_ds_tags: list = None):
    print("== Perform ANN ...")
    subfolder = outfolder + "/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    create_model_with_args = \
        functools.partial(create_model, max_index=embedding_layer_max_index,
                          wemb_matrix=pretrained_embedding_matrix,
                          model_descriptor=model_descriptor)
    # model = MyKerasClassifier(build_fn=create_model_with_args, verbose=0)
    model = KerasClassifier(build_fn=create_model_with_args, verbose=0, batch_size=100)
    model.fit(X_data, y_data)

    nfold_predictions = cross_val_predict(model, X_data, y_data, cv=nfold)

    util.save_scores(nfold_predictions, y_data, None, None,
                     model_descriptor, dataset_name, 3,
                     outfolder, instance_data_source_tags, accepted_ds_tags)

    # util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
    #                       time_ann_predict_dev,
    #


# def cross_fold_eval(input_data_file, dataset_name, sys_out, model_descriptor: str,
#                     print_scores_per_class,
#                     word_norm_option,
#                     randomize_strategy,
#                     pretrained_embedding_model=None, expected_embedding_dim=None):
#     raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
#     M = get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
#     # M=self.feature_scale(M)
#     M0 = M[0]
#

#     pretrained_word_matrix = None
#     if pretrained_embedding_model is not None:
#         pretrained_word_matrix = pretrained_embedding(M[1], pretrained_embedding_model, expected_embedding_dim,
#                                                       randomize_strategy)
#
#     # split the dataset into two parts, 0.75 for train and 0.25 for testing
#     X_data = M0
#     y_data = raw_data['class']
#     y_data = y_data.astype(int)
#
#     X_data = sequence.pad_sequences(X_data, maxlen=MAX_SEQUENCE_LENGTH)
#
#     instance_data_source_column = None
#     accepted_ds_tags = None
#     if print_scores_per_class:
#         instance_data_source_column = pd.Series(raw_data.ds)
#         accepted_ds_tags = ["c", "td"]
#
#     cross_eval_dnn(dataset_name, sys_out, model_descriptor,
#                    -1, 5,
#                    X_data,
#                    y_data,
#                    len(M[1]), pretrained_word_matrix,
#                    instance_data_source_column, accepted_ds_tags)
#     print("complete {}".format(datetime.datetime.now()))


##############################################
##############################################

# /home/zqz/Work/data/GoogleNews-vectors-negative300.bin.gz
# 300

if __name__ == "__main__":
    print("start {}".format(datetime.datetime.now()))
    emb_model = None
    emb_models = None
    emb_dim = None
    params = {}

    sys_argv = sys.argv
    if len(sys.argv) == 2:
        print("argument length is 2")
        sys_argv = sys.argv[1].split(" ")

    for arg in sys_argv:
        pv = arg.split("=", 1)
        print("PV : ",str(pv))
        if (len(pv) == 1):
            continue
        params[pv[0]] = pv[1]
    if "scoreperclass" not in params.keys():
        print("scoreperclass in parameters")
        params["scoreperclass"] = False
    else:
        params["scoreperclass"] = True
    if "word_norm" not in params.keys():
        print("word_norm not in parameters")
        params["word_norm"] = 1
    if "oov_random" not in params.keys():
        print("oov_random not in parameters")
        params["oov_random"] = 0
    if "emb_model" in params.keys():
        print("emb_model not in parameters")
        emb_models = []
        print("===> use pre-trained embeddings...")
        model_str = params["emb_model"].split(',')
        for m_s in model_str:
            gensimFormat = ".gensim" in m_s
            if gensimFormat:
                emb_models.append(gensim.models.KeyedVectors.load(m_s, mmap='r'))
            else:
                # print("ms")
                # print(m_s)
                emb_models.append(gensim.models.KeyedVectors. \
                                  load_word2vec_format(m_s, binary=True))
        print("<===loaded {} models".format(len(emb_models)))
    if "emb_dim" in params.keys():
        print("emb_dim not in parameters")
        emb_dim = int(params["emb_dim"])
    if "gpu" in params.keys():
        print("gpu not in parameters")
        if params["gpu"] == "1":
            print("using gpu...")
        else:
            print("using cpu...")
    if "wdist" in params.keys():
        print("wdist not in parameters")
        wdist_file = params["wdist"]
    else:
        wdist_file = None


    use_mixed_data=False

    print("<<<<<< Using Mixed Data={} >>>>>>>".format(use_mixed_data))
    gridsearch(params["input"],
               params["dataset"],  # dataset name
               params["output"],  # output
               params["model_desc"],  # model descriptor
               params["scoreperclass"],  # print scores per class
               params["word_norm"],  # 0-stemming, 1-lemma, other-do nothing
               params["oov_random"],  # 0-ignore oov; 1-random init by uniform dist; 2-random from embedding
               emb_models,
               emb_dim,
               wdist_file,
               use_mixed_data)
    # K.clear_session()
    # ... code
    sys.exit(0)

    # input=/home/zqz/Work/chase/data/ml/ml/rm/labeled_data_all.csv
    # output=/home/zqz/Work/chase/output
    # dataset=rm
    # model_desc="dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax"




# emb_model=/home/zz/Work/data/glove.840B.300d.bin.gensim
# emb_dim=300