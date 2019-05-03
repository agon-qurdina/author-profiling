import pandas as pd
import re
import csv
from gensim.parsing import preprocessing
import contractions
from sklearn.utils import shuffle
import pickle
import os
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import math
import numpy as np

pan_dir_path = '/home/agon/Competitions/PAN - Author Profiling/'

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def clean_text(text):
    """ Cleans the text in the only argument in various steps 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    if isfloat(text):
        try:
            if math.isnan(text):
                return ''
        except TypeError:
            print('text: {}'.format(text))
            return ''
    
    # Replace newlines by space. We want only one doc vector.
    text = text.replace('\n', ' ').lower()
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove stop words
    text = preprocessing.remove_stopwords(text)
    # Remove html tags and numbers: can numbers possible be useful?
    text = preprocessing.strip_tags(preprocessing.strip_numeric(text))
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
    #text = re.sub(r'[^\w\s]', '', text.lower())   
    # STEMMING (Porter) automatically lower-cases as well
    # To stem or not to stem, that is the question
    #text = preprocessing.stem_text(text)
    return text

def process_text(text):
  if isfloat(text):
        try:
            if math.isnan(text):
                return ''
        except TypeError:
            print('text: {}'.format(text))
            return ''

  # remove links from the text
  text = re.sub(r'https?:\/\/.*[\r\n]*', 'https ', text, flags=re.MULTILINE)
  # split into words
  tokens = word_tokenize(text)
  # convert to lower case
  tokens = [w.lower() for w in tokens]
  # remove punctuation from each word
  table = str.maketrans('', '', string.punctuation)
  stripped = [w.translate(table) for w in tokens]
  # remove remaining tokens that are not alphabetic
  words = [word for word in stripped if word.isalpha()]
  # filter out stop words
  stop_words = set(stopwords.words('english'))
  words = [w for w in words if not w in stop_words]
  return ' '.join(words).strip()

def generate_df(tsv_file, preprocess_texts=True):
  df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', names=['author', 'text', 'bot', 'gender'])

  if preprocess_texts:
      # print('Cleaning texts...')
      # df.text = df['text'].apply(clean_text)
      print('Processing texts...')
      df.text = df['text'].apply(process_text)
  
  # Remove rows with empty texts
  df['text'].replace('', np.nan, inplace=True)
  df.dropna(subset=['text'], inplace=True)

  # Shuffle it
  df = shuffle(df, random_state=13)

  return df

def main(preprocess_texts=True):
  datasets = ['train', 'test']
  for dataset in datasets:
    df_file = os.path.join(pan_dir_path, 'data', 'dataframes', '{}.pkl'.format(dataset)) 
    if os.path.isfile(df_file):
      df = pd.read_pickle(df_file)
      print("{} dataframe shape = {}".format(dataset, df.shape))
      print(list(df.loc[df['author'] == '304c12afb0a7e20f49bf84e054993e98'].head(1).text))
    else:
      tsv_file = os.path.join(pan_dir_path, 'data', 'tsv', '{}.tsv'.format(dataset))
      df = generate_df(tsv_file, preprocess_texts=preprocess_texts)

      print("{} dataframe shape after cleaning = {}".format(dataset, df.shape))
      print('Writing dataframe to disk...')
      df.to_pickle(df_file)

if __name__ == "__main__":
  main()