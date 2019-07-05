import argparse
import os
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SpatialDropout1D, Conv1D, MaxPooling1D, Activation, Embedding, Flatten, GlobalMaxPooling1D, LSTM
from keras.preprocessing.text import Tokenizer
from keras import regularizers, callbacks, optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import pickle
from gensim.models import KeyedVectors

pan_dir = ''
model_name = 'dl_model_a0_w0'

def evaluate_model(model, X_val, y_val):
    y_predict = (np.asarray(model.predict(X_val))).round()

    acc = metrics.accuracy_score(y_val, y_predict)
    logging.info('Accuracy: {}'.format(acc))

    conf_matrix = metrics.confusion_matrix(y_val, y_predict)
    logging.info('Confusion matrix: {}'.format(conf_matrix))

    precision = metrics.precision_score(y_val, y_predict)
    logging.info('Precision score: {}'.format(precision))

    recall = metrics.recall_score(y_val, y_predict)
    logging.info('Recall score: {}'.format(recall))

    val_f1 = metrics.f1_score(y_val, y_predict)
    logging.info('F1 score: {}'.format(val_f1))

    val_auc = metrics.roc_auc_score(y_val, y_predict)
    logging.info('Auc score: {}'.format(val_auc))

    # model_plot_file = os.path.join(pan_dir, 'models', '{}.png'.format(final_model_name))
    # plot_model(model, to_file=model_plot_file, show_shapes=True, show_layer_names=True)

def load_data():
  train_df_file = os.path.join(pan_dir, 'data', 'dataframes', 'train.pkl') 
  train_df = pd.read_pickle(train_df_file)
  train_df['bot'] = train_df['bot'].apply(lambda x: 1 if x == 'bot' else 0)

  test_df_file = os.path.join(pan_dir, 'data', 'dataframes', 'test.pkl') 
  test_df = pd.read_pickle(test_df_file)
  test_df['bot'] = test_df['bot'].apply(lambda x: 1 if x == 'bot' else 0)

  return train_df['text'], train_df['bot'], test_df['text'], test_df['bot']

def load_tokenizer(X_train, num_words=None):
  file_path = os.path.join(pan_dir, 'data', 'tokenizers', 'tokenizer_{}.pickle'.format(num_words))
  if os.path.isfile(file_path):
    with open(file_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    logging.info('Tokenizer loaded from disk')
    return tokenizer
  else:
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train)

    with open(file_path, 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file,protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('Tokenizer fit on texts and stored on disk')

    return tokenizer

def create_embedding_weights_matrix(word_vectors, word_index, embedding_dims=300):
  weights_matrix = np.zeros((len(word_index) + 1, embedding_dims))

  count = 0
  for word, idx in word_index.items():
      if word in word_vectors:
          weights_matrix[idx] = word_vectors[word]
          count += 1
  logging.info('Words found on word2vec: {}'.format(count))

  return weights_matrix 

def load_embedding_layer(tokenizer, seq_len):
  vocab_size = len(tokenizer.word_index) + 1
  logging.info('Vocab size: {}'.format(vocab_size))

  # Load word vectors
  logging.info("Loading Google's word2vec vectors")
  filename = os.path.join('/home/agon/SemEvalData', 'GoogleNews-vectors-negative300.bin')
  model = KeyedVectors.load_word2vec_format(filename, binary=True)
  weights_matrix = create_embedding_weights_matrix(model.wv, tokenizer.word_index)
  
  return Embedding(input_dim=vocab_size, 
                              output_dim=weights_matrix.shape[1], 
                              input_length=seq_len,
                              weights=[weights_matrix],
                              trainable=False
                              )

def define_conv_model(tokenizer, seq_len, filters=64, kernel_size=4, hidden_dims=256):
    model = Sequential()

    # embedding_layer = load_embedding_layer(tokenizer, seq_len=seq_len)
    vocab_size = len(tokenizer.word_index) + 1
    print('seq_len: {}'.format(seq_len))
    print('vocab_size: {}'.format(vocab_size))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, input_length=seq_len)
    
    model.add(embedding_layer)
    model.add(Dropout(0.5))

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu'))
    model.add(Dropout(0.5))
    # model.add(SpatialDropout1D(0.5))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=4))

    model.add(GlobalMaxPooling1D())
    # model.add(Flatten())

    model.add(Dense(hidden_dims, 
                    activation='relu'
                    ))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

def define_deep_conv_model(tokenizer, seq_len, filters=64, kernel_size=4, hidden_dims=256):
    model = Sequential()

    vocab_size = len(tokenizer.word_index) + 1
    print('seq_len: {}'.format(seq_len))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, input_length=seq_len)
    model.add(embedding_layer)
    model.add(Dropout(0.5))

    for i in range(0, 5):
      model.add(Conv1D(filters,
                      kernel_size,
                      activation='relu'))
      model.add(Dropout(0.5))
      model.add(MaxPooling1D(pool_size=2))

    # model.add(GlobalMaxPooling1D())
    model.add(Flatten())

    for i in range(0, 2):
      model.add(Dense(hidden_dims, 
                      activation='relu'
                      ))
      model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size, learning_rate):
  model_dir = os.path.join(pan_dir, 'data', 'models')
  model_location = os.path.join(model_dir, '{}.h5'.format(model_name))
  model_weights_location = os.path.join(model_dir, '{}_weights.h5'.format(model_name))

  # Implement Early Stopping
  early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=5,
                          verbose=1)
                          #   restore_best_weights=True)
  save_best_model = callbacks.ModelCheckpoint(model_weights_location, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
  
  adam = optimizers.Adam(lr=learning_rate, decay=0.01)
  model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
  history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=1000,
              verbose=2,
              validation_data=(X_val, y_val),
              callbacks=[early_stopping_callback, save_best_model])
  
  #reload best weights
  model.load_weights(model_weights_location)

  logging.info('Model trained. Storing model on disk.')
  model.save(model_location)
  logging.info('Model stored on disk.')

def load_pretrained_model(model_name):
    model_file = os.path.join(pan_dir, 'data', 'models', "{}.h5".format(model_name))
    model = load_model(model_file)
    return model

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path",'-p', default="/home/agon/Competitions/PAN - Author Profiling/",
                      help="Use this argument to change the PAN directory path (the default path is: '/home/agon/Competitions/PAN - Author Profiling/')")
  parser.add_argument("--model", '-m', default="",
                      help="Use this argument to continue training a stored model")
  parser.add_argument("--learning_rate", '-l', default="0.001",
                      help="Use this argument to set the learning rate to use. Default: 0.001")
  parser.add_argument("--evaluate", '-e', action='store_true', default="False",
                      help="Use this argument to set run on evaluation mode")
  args = parser.parse_args()
  
  global pan_dir
  pan_dir = args.path

  logs_path = os.path.join(pan_dir, 'logs', 'dl_model_log.log')
  logging.basicConfig(filename=logs_path, filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

  evaluate_mode = args.evaluate
  learning_rate = float(args.learning_rate)
  batch_size = 32
  model_name = args.model

  X_train, y_train, X_val, y_val = load_data()
  lengths = np.array([len(text) for text in X_train])
  print('Count: {}'.format(len(lengths)))
  print('Min length: {}'.format(lengths.min()))
  print('Avg length: {}'.format(lengths.mean()))
  print('Std length: {}'.format(lengths.std()))
  print('Max length: {}'.format(lengths.max()))
  print('Count of sequences > 11000: {}'.format(len([length for length in lengths if length > 11000])))
  seq_length = int(lengths.mean() + lengths.std())
  
  tokenizer = load_tokenizer(X_train)

  train_sequences = tokenizer.texts_to_sequences(X_train)
  X_train = pad_sequences(train_sequences, maxlen=seq_length, padding='post')

  val_sequences = tokenizer.texts_to_sequences(X_val)
  X_val = pad_sequences(val_sequences, maxlen=seq_length, padding='post')

  if model_name:
    model = load_pretrained_model(model_name)
  else:
    model = define_conv_model(tokenizer, seq_length)

  logging.info(model.summary())

  if evaluate_mode is True:
    evaluate_model(model, X_val, y_val)
  else:
    train_model(model, X_train, y_train, X_val, y_val, batch_size, learning_rate)
    evaluate_model(model, X_val, y_val)


if __name__ == "__main__":
    main()