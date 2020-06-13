
"""**Importing necessary packages and inputting global parameters**"""

import os
import sys
import gzip
import shutil
import pandas as pd
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt

data_path = os.getcwd()
#output_file = open('test.txt','w')

dataset = sys.argv[1]   #dataset name
K = int(sys.argv[2])    #number of items in frequent itemsets
F = int(sys.argv[3])    #minimum frequency required for itemset to be frequent

lambda_factor = 0.9    # lambda factor in setting minimum item support of each item
const_factor = 1e-1     #support difference constraint factor
"""**Check if dataset is extracted or extract it**"""
print ('\nMS Apriori for dataset %s with K=%d,F=%d,lambda_factor=%f,constraint_factor=%f\n' %(dataset,K,F,lambda_factor,const_factor))
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

"""**Create maps for word to index, index to word in vocabulary, word counts,
multiple itemset supports, and support counts
**"""

vocab_word2idx = {}
vocab_idx2word = {}
vocab_wordCount = {}
mis = {}
supp_count = {}

for i,word in enumerate(vocab_dataset):
  vocab_word2idx[word] = i+1
  vocab_idx2word[i+1] = word

"""**Read the dataset**"""
docword_dataset = pd.read_csv(os.path.join(data_path,'docword_{}.txt'.format(dataset)),sep=" ", skiprows=3)
docword_dataset.columns = ['Doc_Id','Word','Count']
docword_dataset['Word'] = docword_dataset['Word'].apply(lambda x:vocab_idx2word[x])
#print (docword_dataset.head())


docword_file = open(os.path.join(data_path,'docword_{}.txt'.format(dataset)),'r')
dataset_info = ''.join([next(docword_file) for i in range(3)])
docword_file.close()
dataset_info = dataset_info.split('\n')

D = int(dataset_info[0])
W = int(dataset_info[1])
NNZ = int(dataset_info[2])
print ('Number of documents,words and non zero frequency words')
print (D,W,NNZ)

#Converting the dataset to transaction form
from tqdm import tqdm
transactions = []



print ('#####Reading input file as a tuple of words for each document#####')
item_list = []
max_occ_item = 0
for document_id in tqdm(range(1,D+1)):
  items = tuple(docword_dataset[docword_dataset['Doc_Id']==document_id]['Word'].values)
  word_counts = tuple(docword_dataset[docword_dataset['Doc_Id']==document_id]['Count'].values)
  for i,word in enumerate(items):
      supp_count[word] = supp_count.get(word,0) + 1
      vocab_wordCount[word] = vocab_wordCount.get(word,0) + word_counts[i]

      if word not in item_list:
          item_list.append(word)


  transactions.append(items)

  for word in item_list:
      if supp_count[word] < F:
          mis[word] = 10
      else:
          mis[word] =  lambda_factor * supp_count[word]/D


'''for word in item_list:
    #mis[word] = lambda_factor * vocab_wordCount[word]/sum(vocab_wordCount.values())
    mis[word] = lambda_factor * (1 - word_minOcc[word]/word_maxOcc[word])
'''
'''plt.hist(mis.values())
plt.show()'''

N = D

def init_pass(M):       #function to find the initial candidate itemsets of Length 1


    L = []
    j = -1
    for i in range(len(M)):

        item = M[i]
        if j==-1:
            if supp_count[item]/N >= mis[item]:    #ITem M[j] contains the first item whose support is greater than or equal to that that item's mis value
                L.append(item)
                j = i

        else:
            if supp_count[item]/N >=mis[M[j]]:     #for all other items check if its support value is greater than MIS of Item j
                L.append(M[i])

    return L


def level2_candidate_gen(L,const_factor):   #function to generate 2-itemset candidates
    C = []

    for i in range(len(L)):
        l = L[i]
        if supp_count.get(l)/N >=mis.get(l):
            for j in range(i+1,len(L)):
                h = L[j]
                if supp_count.get(h)/N >=mis[l] and abs(supp_count.get(h)/N -supp_count[l]/N)<=const_factor:
                    C.append((l,h))

    return C

def MS_candidate_gen(F,const_factor):      #function to generate Candidates for F(i+1) given F(i)
    C = []
    for i in range(len(F)):
        f1 = F[i][0:-1]
        for j in range(i+1,len(F)):
            f2 = F[j][0:-1]
            if (f1==f2) and supp_count[f1[-1]] - supp_count[f2[-1]]<=const_factor:

                t = (F[j][-1],)
                temp = F[i] + t
                C.append(temp)




    for c in C:
        subsets = list(itertools.combinations(c,len(c)-1))
        #print (subsets)
        for s in subsets:
            if (c[0] in s) or (mis[c[0]]==mis[c[1]]):
                if s not in F:
                    C.remove(c)

    return C



def msapriori(K,const_factor=0.001):    #function to generate K itemset
    M = sorted(item_list,key = lambda x:mis[x])
    L = init_pass(M)
    print ('Length of L = %d' %(len(L)))
    F = []
    C_counts = {}

    for item in L:
        if supp_count.get(item)/N >= mis[item]:
            F.append((item))

    for i in range(2,K+1):
        if F==[]:
            break
        next_f = []

        print ('Number of items in %d frequent itemset = %d' %((i-1),len(F)))

        print ('######Computing level %d candidate itemset######' %i)
        if i==2:

            CK = level2_candidate_gen(L,const_factor)
        else:
            CK = MS_candidate_gen(F,const_factor)

        print ('Level %d candidate itemset computed of length %d' %(i,len(CK)))

        #print ('\n\nFrequent Itemset for K=%d\n' %(i-1))
        #print (F)

        print ('######Pruning level %d candidate set to get our %d itemset######' %(i,i))

        print ('######Generating counts of each candidate######')
        for t in tqdm(transactions):
            for c in CK:
                if (set(c).issubset(set(t))):
                    C_counts[str(c)] = C_counts.get(str(c),0) + 1





        for candidate in CK:
            if C_counts.get(str(candidate),0)/N >= mis[candidate[0]]:
                next_f.append(candidate)


        F = next_f


    return (F,C_counts)



start = time.time()

itemset,counts =  msapriori(K,const_factor)
print ('\n%d Frequent itemsets:\n' %K)
for item in itemset:
    if K==1:

        print ((item,supp_count[item]), end=' , ')
    else:

        print ((item,counts[str(item)]), end=' , ')
print ('\nNumber of items in %d frequent itemset = %d' %(K,len(itemset)))

stop = time.time()
t = stop - start
print ('\n\nTime taken = %d hours,%d minutes,%f seconds\n\n' %(t//3600,t//60,t%60))
#output_file.close()
