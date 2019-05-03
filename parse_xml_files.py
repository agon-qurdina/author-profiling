from lxml import etree
from bs4 import BeautifulSoup
import csv
import os
import regex as re
import base64
import pandas as pd

pan_dir = '/vol3/AuthorProfiling/'
TWEETS_DELIMITER = "\n"

def _parse_truths(truth_file_path):
  truths = {}
  regexp_1 = re.compile(r'(\w+):::(\w+):::(\w+)')
  for line in open(truth_file_path,'r'):
    re_match = regexp_1.match(line)
    author = re_match.group(1)
    bot = re_match.group(2)
    gender = re_match.group(3)
    truths[author] = { 'bot': bot, 'gender': gender }
  return truths

def _get_text(full_file_path):
  la_belle_soupe = BeautifulSoup(open(full_file_path).read(),"lxml-xml")
  documents = la_belle_soupe.find_all("document")
  doc_list = []
  for document in documents:
    tweet = document.get_text()
    doc_list.append(tweet)
  return TWEETS_DELIMITER.join(doc_list)

def _parse_data():
  xml_files_dir = os.path.join(pan_dir, 'data', 'en')
  truth_files = ['truth.txt', 'truth-dev.txt', 'truth-train.txt']
  train_truths = _parse_truths(os.path.join(xml_files_dir, truth_files[2]))
  dev_truths = _parse_truths(os.path.join(xml_files_dir, truth_files[1]))

  train_data = []
  test_data = []
  for filename in os.listdir(xml_files_dir):
    if not filename in truth_files:
      full_file_path = os.path.join(xml_files_dir, filename)
      text = _get_text(full_file_path)
      author = filename.replace('.xml', '')
      if author in train_truths.keys():
        truth = train_truths[author]
        train_data.append({ 'author': author, 'text': text, **truth })
      else:
        truth = dev_truths[author]
        test_data.append({ 'author': author, 'text': text, **truth })
  return train_data, test_data

def _write_to_tsv(data, name):
    fieldnames = ['author', 'text', 'bot', 'gender']
    output_tsv_path = os.path.join(pan_dir, 'data', 'tsv', '{}.tsv'.format(name))
    tsv = open(output_tsv_path, 'w', encoding='utf-8')
    writer = csv.DictWriter(tsv, delimiter='\t', fieldnames=fieldnames)
    for record in data:
      writer.writerow(record)

def parse_xml_files(xml_files_dir, skip_truths=False):
  data = []
  truth_files = ['truth.txt', 'truth-dev.txt', 'truth-train.txt']

  if not skip_truths:
    truths = _parse_truths(os.path.join(xml_files_dir, truth_files[0]))

  for filename in os.listdir(xml_files_dir):
    if not filename in truth_files:
      full_file_path = os.path.join(xml_files_dir, filename)
      text = _get_text(full_file_path)
      author = filename.replace('.xml', '')
      if skip_truths:
        data.append({ 'author': author, 'text': text })
      else:
        if author in truths.keys():
          truth = truths[author]
          data.append({ 'author': author, 'text': text, **truth })
  return data

def generate_tsv(xml_files_dir, filename = 'data', pan_dir_path=''):
  if pan_dir_path:
    global pan_dir
    pan_dir = pan_dir_path
  tsv_file_path = os.path.join(pan_dir, 'data', 'tsv', '{}.tsv'.format(filename))
  if not os.path.isfile(tsv_file_path):
    data = parse_xml_files(xml_files_dir)
    _write_to_tsv(data, filename)
  return tsv_file_path

def generate_df_on_the_fly(xml_files_dir):
  data = parse_xml_files(xml_files_dir, skip_truths=True)
  # fieldnames = ['author', 'text', 'bot', 'gender']
  df = pd.DataFrame(data)
  return df
  # f = StringIO()
  # writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
  # writer.writerows(data)
  # return base64.b64encode(f.getvalue())

def main():
  train_data, test_data = _parse_data()
  _write_to_tsv(train_data, 'train')
  _write_to_tsv(test_data, 'test')

  # authors = []

  # xml_files_dir = os.path.join(pan_dir_path, 'data', 'en')
  # limit = 1
  # i = 0
  # for filename in os.listdir(xml_files_dir):
  #   if filename != 'truth.txt':
  #     full_file_path = os.path.join(xml_files_dir, filename)
  #   write_to_tsv()
    # i += 1
    # doc = etree.parse(full_file_path)
    # root = doc.getroot()
    # authors.append(root)
    # if i == limit:
    #   break

  # print(full_file_path)
  # i = 0
  # for child in root.iter():
  #   i += 1
  #   print(child.text)
  # print('Count: {}'.format(i))

if __name__ == "__main__":
    main()