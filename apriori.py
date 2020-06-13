
"""**Importing necessary packages and inputting global parameters**"""

import os
import sys
import gzip
import shutil
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
import time


data_path = os.getcwd()

dataset = sys.argv[1]   #dataset name
K = int(sys.argv[2])    #number of items in frequent itemsets
F = int(sys.argv[3])    #minimum frequency for itemset to be said as frequent itemset


"""**Check if dataset is extracted or extract it**"""

if not os.path.exists(os.path.join(data_path,'docword_{}.txt'.format(dataset))):
  with gzip.open(os.path.join(data_path,'docword.{}.txt.gz'.format(dataset)), 'rb') as f_in:
      with open(os.path.join(data_path,'docword_{}.txt'.format(dataset)), 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)

"""**Read Vocabulary File**"""

vocab_file = open(os.path.join(data_path,'vocab.{}.txt'.format(dataset)),'r')
vocab_dataset = vocab_file.read()
vocab_file.close()

vocab_dataset = vocab_dataset.split('\n')[:-1]

#vocab_dataset[:10],len(vocab_dataset)

"""**Create maps for word to index and index to word in vocabulary**"""

vocab_word2idx = {}
vocab_idx2word = {}

for i,word in enumerate(vocab_dataset):
  vocab_word2idx[word] = i+1
  vocab_idx2word[i+1] = word

"""**Read the dataset**"""

docword_dataset = pd.read_csv(os.path.join(data_path,'docword_{}.txt'.format(dataset)),sep=" ", skiprows=3)
docword_dataset.columns = ['Doc_Id','Word_Id','Count']
#docword_dataset.head()

docword_file = open(os.path.join(data_path,'docword_{}.txt'.format(dataset)),'r')
dataset_info = ''.join([next(docword_file) for i in range(3)])
docword_file.close()
dataset_info = dataset_info.split('\n')

D = int(dataset_info[0])
W = int(dataset_info[1])
NNZ = int(dataset_info[2])
print (D,W,NNZ)

"""**Transforming the dataset to a one hot encoded version of transactions**"""

from tqdm import tqdm
transactions = np.zeros(shape=(D,W),dtype = np.int8)
for document_id in tqdm(range(1,D+1)):
  items = tuple(docword_dataset[docword_dataset['Doc_Id']==document_id]['Word_Id'].values)
  for item in items:
    transactions[document_id-1][item-1] = 1

data = pd.DataFrame(transactions )
data.columns = vocab_dataset
print (data.head())

print (data.shape)

"""**Finding the K Frequent itemsets using Apriori**"""

start_time = time.time()
res = apriori(data, min_support = F/D,max_len = K+1)
stop_time = time.time()
time_taken = stop_time - start_time
#print (res)

def words(itemset):
  t = []
  for item in itemset:
    t.append(vocab_idx2word[item])
  return tuple(t)

res['Frequency'] = res['support']*D
res['Items'] = res['itemsets'].apply(lambda x: words(x))
res

s = str (res[res['Items'].apply(lambda x:len(x))==K])
t = str('\n\n\nTime taken to find {} frequent items with minimum frequency {} for {} dataset is {} seconds\n\n\n\n'.format(K,F,dataset,time_taken))

print (s)
print (t)

"""**Storing the output in text file**"""

with open(os.path.join(data_path,'test_output.txt'),'w') as f:

  f.write ('\n\n\nOutput for {} dataset for K={} and F = {}\n\n\n'.format(dataset,K,F))

  f.write(s)
  f.write(t)
