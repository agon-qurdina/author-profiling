import pandas as pd
import os
import getopt
import sys
from time import sleep
import logging
from keras.models import load_model
from datetime import datetime
from clean_parse_tsv import generate_df
from parse_xml_files import generate_df_on_the_fly
import numpy as np
from sklearn.utils import shuffle
import xml.etree.cElementTree as ET
from keras.preprocessing.sequence import pad_sequences
import pickle

pan_dir_path = '/home/faerber19/AuthorProfiling/' # '/vol3/AuthorProfiling/'
logging.basicConfig(filename='{}/logs/preidctions_log.log'.format(pan_dir_path), filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
seq_length = 11251
model_name = 'dl_model_a0_w0'
output_path = ''
input_dir = ''

def _convert_tsv_to_dataframe(tsv_file):
    df = generate_df(tsv_file)

    # df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', names=['author', 'text', 'bot', 'gender'])
    
    # Remove rows with empty texts
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)

    # Shuffle it
    df = shuffle(df, random_state=13)

    return df

def _predict(model, X_val):
    predicted_values = model.predict_classes(X_val)

    logging.info('predicted_values: ')
    logging.info(predicted_values)

    formatted_pred = predicted_values.reshape((-1,))
    return pd.Series(formatted_pred)

def _load_tokenizer():
  file_path = os.path.join(pan_dir_path, 'data', 'tokenizers', 'tokenizer_None.pickle')
  with open(file_path, 'rb') as tokenizer_file:
      tokenizer = pickle.load(tokenizer_file)
  return tokenizer

def _create_output_dataframe(input_df, y_pred):
  # outputs_dir = os.path.join(pan_dir_path, 'outputs')

  truefalsedict = {0: 'human', 1: 'bot'}
  y_pred_df = pd.DataFrame(y_pred, columns=['predicted_type'])
  y_pred_df['predicted_type'] = y_pred_df['predicted_type'].map(truefalsedict, na_action=None)
  y_pred_df['author'] = input_df['author'].values
  # Reorder the columns
  y_pred_df = y_pred_df[['author', 'predicted_type']]

  output_en_path = os.path.join(output_path, 'en')
  if not os.path.exists(output_en_path):
    os.mkdir(output_en_path)
  for index, row in y_pred_df.iterrows():
    root = ET.Element("author", id=row['author'], lang='en', type=row['predicted_type'])
    tree = ET.ElementTree(root)
    tree.write(os.path.join(output_en_path, '{}.xml'.format(row['author'])))
  
  return y_pred_df

def _write_output_dataframe_to_file(y_pred_df, outfile):
    logging.info('Writing output dataframe to file')
    y_pred_df.to_csv(outfile, sep=' ', index=False, header=False)


def main():
  global output_path, input_dir
  output_path = sys.argv[1]
  input_dir = sys.argv[2]
  logging.info('output_path: {}'.format(output_path))
  logging.info('input_dir: {}'.format(input_dir))
  input_dir = os.path.join(input_dir, 'en')

  logging.info("Starting...")

  # Load model
  model_file = os.path.join(pan_dir_path, 'data', 'models', "{}.h5".format(model_name))
  model = load_model(model_file)

  # Generate dataframe from xml files (df columns: 'text', 'author')
  df = generate_df_on_the_fly(input_dir)
  X = df['text']
  logging.info("Dataframe generated")

  # Generate and pad sequences
  tokenizer = _load_tokenizer()
  sequences = tokenizer.texts_to_sequences(X)
  X = pad_sequences(sequences, maxlen=seq_length, padding='post')
  logging.info("Sequences generated and padded")

  # Make the prediction
  y_pred = _predict(model, X)
  logging.info("Predictions made")

  # Create output dataframe to write on disk
  y_pred_df = _create_output_dataframe(df, y_pred)
  logging.info("Output dataframe and XML files created: {}".format(len(y_pred_df)))
  
  # Write preditions df to file
  outfile = os.path.join(pan_dir_path, 'temp_output', 'prediction_output_df.pkl')
  _write_output_dataframe_to_file(y_pred_df, outfile)
  logging.info("The predictions have been written to the output folder.")


if __name__ == "__main__":
    main()
