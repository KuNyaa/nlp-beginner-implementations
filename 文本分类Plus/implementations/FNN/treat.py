import sys
import csv
import nltk
import string
import datetime
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from string import ascii_lowercase as letters
stopwords = stopwords.words('english')

            
def load_data(path):
    '''Read data from a tsv file'''
    tsvfile=open(path, 'r')
    tsvreader = csv.reader(tsvfile, dialect='excel')
    data = []
    for row in tsvreader:
        data.append(row)
    tsvfile.close()
    return data

def save_data(path, data):
    '''Save data to a csv file'''
    csvfile=open(path, 'w', newline='')
    csvwriter = csv.writer(csvfile, dialect='excel')#delimiter=',')
    for result in data:
        csvwriter.writerow(result)
    csvfile.close()

name = sys.argv[1]

data = load_data(name)[1:]

output = [['PhraseId', 'Sentiment']] + [[156061, 3]]
for sample in data:
    output.append([int(sample[0]) + 1, int(sample[1])])
save_data(name + '_', output)

'''
print(len(train_input))
print(len(test_input))
save_data('train.in', train_input)
save_data('test.in', test_input)
'''
