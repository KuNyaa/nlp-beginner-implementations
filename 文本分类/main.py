import csv
import numpy as np
import time
import random


classes_num = 5

def load_data(path):
    '''Read data from a tsv file'''
    tsvfile=open(path, 'r')
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    data = []
    for row in tsvreader:
        data.append(row)
    tsvfile.close()
    return np.array(data)

def save_data(path, data):
    csvfile=open(path, 'w', newline='')
    csvwriter = csv.writer(csvfile, dialect='excel')#delimiter=',')
    for result in data:
        csvwriter.writerow(result)
    csvfile.close()

    
def get_dicts(text, n):
    '''Generate the dictionary of n_gram model

    >>>text = [['How', 'are', 'you'], ['I', 'am', 'fine']]
    >>>dicts = get_dicts(text, 2)
    >>>print(dicts)
    ['', 'am', 'am fine', 'are', 'are you', 'fine', 'how', 'how are', 'i', 'i am', 'you']
    '''
    assert(n >= 1)
    dicts = set()
    dicts.add('')  # In python, '' is in any string, so we use it as the bias position
    for sentence in text:
        for i in range(len(sentence)):
            for length in range(n):
                if i - length >= 0:
                    dicts.add(' '.join([word.lower() for word in sentence[i-length: i+1]]))
    dicts = sorted(list(dicts))
    
    return np.array(dicts)


def label_to_one_hot(label):
    '''Convert a label to a one hot vector'''
    one_hot = np.zeros(classes_num)
    one_hot[label] = 1
    return one_hot


def get_features(data_orig, dicts, tag):
    '''Convert sentences to feature vectors, and return the feature matrix, each column is a sample.
       data_orig : the raw data
           dicts : the dictionary of n_gram model
             tag : 'train' or 'test'
    '''
    features = []
    labels = []
    n = len(dicts)
    for i in range(1, len(data_orig)):   # ignore the headers
        sentence = data_orig[i][2].lower()
        feature = np.array([word in sentence for word in dicts], np.int32)
        # the 1st element is the bias term
        if tag == 'train':
            label = int(data_orig[i][3]) # 0 1 2 3 4
            labels.append(label_to_one_hot(label))
        features.append(feature)

    return np.array(features).T, np.array(labels).T


def max_label(x):
    '''Return the label with maximun probability in vector x'''
    return list(x).index(np.max(x))


def softmax(Y):
    '''Compute the softmax for each column of a matrix Y'''
    #logC = -np.max(Y, axis=0, keepdims=True)
    exp_x = np.exp(Y) #* np.exp(logC)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def predict(X, W):
    '''Generate the prediction of given input matrix X with the weight matrix
       X  : a matrix of input data, each column is a smaple
       W  : a weight matrix
    '''
    Y = W.dot(X)
    return softmax(Y)


def loss(label_Y, pred_Y):
    '''Calculate the loss for the prediction pred_Y with label_Y'''
    m = label_Y.shape[1] # batch size
    Y = (np.log(pred_Y) / m) * label_Y
    '''
    if np.sum(pred_Y == 0) > 0:
        print(np.sum(pred_Y == 0))
    '''
    return - np.sum(Y)

def vectors_to_labels(Y):
    '''Convert prediction matrix Y to a vector of label.(each column of Y is a sample)'''
    labels = []
    Y = list(Y.T)
    for vec in Y:
        vec = list(vec)
        labels.append(vec.index(max(vec)))
    return np.array(labels)

def calc_the_acc(label_Y, pred_Y):
    '''Calculate the accuracy of prediction'''
    return np.sum(label_Y == pred_Y) / label_Y.shape[0]

def training(X, Y, epochs, method='mini batch', W=None, learning_rate=0.003, batch_size=64):
    '''Training the weight vector W on the dataset (X, Y)
                   X : input data matrix
                   Y : labels matrix
                   W : initial value of the weight vector
              epochs : times go through the whole dataset
              method : 'SGD', 'BGD' or 'mini batch'
          batch_size : the batch size for mini-batch method
       learning_rate : learning rate for gradient descent
    '''
    n = X.shape[0]
    m = Y.shape[1]
    if method == 'SGD':
        batch_size = 1
    elif method == 'BGD':
        batch_size = m
    assert(method in ['SGD', 'BGD', 'mini batch'])
        
    if not W:
        W = 0.1*np.random.randn(classes_num, n)
        W[:,0] = np.zeros(classes_num)
    loss_cache = []
    for k in range(epochs+1):
        blocks = m // batch_size
        for i in range(blocks + 1): # 1 for the last not full block
            begin, end = i * batch_size, (i+1) * batch_size
            if end > m:
                end = m
            if begin == end: # last block is empty
                break
            train_X, train_Y = X[:, begin:end], Y[:, begin:end]
            pred_Y = predict(train_X, W)
            L = loss(train_Y, pred_Y)
            dW = (pred_Y - train_Y).dot(train_X.T)
            #gradient descent
            W = W - learning_rate * dW
        if k % 100 == 0:
            pred_Y = predict(X, W)
            L = loss(Y, pred_Y)
            acc = calc_the_acc(vectors_to_labels(Y), vectors_to_labels(pred_Y))
            #cache the data for ploting
            loss_cache.append(L)
            print('epochs ' + str(k) + ':  loss:' + str(L) + '  acc:' + str(acc))
    return W, loss_cache

# load data
train_orig = load_data('./data/train.tsv')
test_orig  = load_data('./data/test.tsv')

train_size = 10000
test_size = len(test_orig)
# get dictionary
text = [train_orig[i][2].split(' ') for i in range(1,train_size)]
dicts = get_dicts(text, 3)

# convert raw data to training data
train_X, train_Y = get_features(train_orig[:train_size], dicts, tag='train')
test_X, _ = get_features(test_orig[:test_size], dicts, tag='test')

print('loaded')
del dicts
del train_orig
del test_orig
# softmax regression
W, loss_cache = training(train_X, train_Y, 3000)

# make prediction on testset

test_Y = vectors_to_labels(predict(test_X, W))

output = [['PhraseId', 'Sentiment']]
for i in range(1, test_size):
    output.append([test_orig[i][0], test_Y[i-1]])
save_data('result.csv', output)

'''
st = time.time()
text = [train_orig[i][2].split(' ') for i in range(1,3)]
text = [['I', 'am', 'fine'], ['How', 'are', 'you']]
train_ = [[],[1,1,'I am fine','2'], [1,1,'How are you', '1']]
test_ =[[],[1,1,'you are fine']]
dicts = get_dicts(text, 2)
dicts_size = len(dicts)
train_x, train_y = get_features(train_, 3, dicts, 'train')
test_x , _ = get_features(test_, 3, dicts, 'test')
W = np.random.randn((dicts_size + 1) * classes_num)
print(predict(test_x, W))
'''
'''
print(train_orig[0])
print(dicts)
print(train_x.shape, train_y.shape)
print(train_x)
print(train_y)
print(time.time() - st)
'''
