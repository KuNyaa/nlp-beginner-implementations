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

class FNN(object):
    '''a simple FNN model for sentiment claasification'''
    def __init__(self, layer_sizes, num_calsses, regularization_rate):

        layers = len(layer_sizes)
        
        self.input_x = tf.placeholder(tf.float32, [None, layer_sizes[0]], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_calsses], name = 'input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')
                                                
        self.cache = {}
        
        self.cache['A0'] = self.input_x
        
        #full connection layers
        for i in range(layers - 1):
            with tf.name_scope('layer-' + str(i)):
                W = tf.Variable(tf.truncated_normal([layer_sizes[i], layer_sizes[i + 1]], stddev = 0.1),name = 'W')
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W) # L2 
                b = tf.Variable(tf.constant(1.0, shape = [layer_sizes[i + 1]]), name = 'b')
                self.cache['Z' + str(i + 1)] = tf.nn.xw_plus_b(self.cache['A' + str(i)], W, b, name='Z')
                #activation 
                A = tf.nn.relu(self.cache['Z' + str(i + 1)], name = 'A')
                #dropout
                self.cache['A' + str(i + 1)] = tf.nn.dropout(A, self.dropout_keep_prob)
            
        #softmax regression
        with tf.name_scope('softmax-regression'):
            W = tf.Variable(tf.truncated_normal([layer_sizes[-1], num_calsses], stddev = 0.1), name = 'W')
            
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W) # L2 
            b = tf.Variable(tf.constant(1.0, shape = [num_calsses]), name = 'b')
            self.scores = tf.nn.xw_plus_b(self.cache['A' + str(layers - 1)], W, b, name = 'scores')
            self.predictions = tf.argmax(self.scores, 1, name = 'predictions')

        #L2 regularization
        with tf.name_scope('L2-regularization'):
            l2_regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
            self.regular_loss = tf.contrib.layers.apply_regularization(l2_regularizer)
            
        #loss
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y, name = 'loss')) + self.regular_loss

        #accuracy
        with tf.name_scope('accuracy'):
            correct_cases = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_cases, 'float'), name = 'accuracy')
            
                                    
            
def load_data(path):
    '''Read data from a tsv file'''
    tsvfile=open(path, 'r')
    tsvreader = csv.reader(tsvfile, delimiter='\t')
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


train_orig = load_data('./data/train.tsv')[1:]
test_orig = load_data('./data/test.tsv')[1:]

argv_names = ['n', 'layer_sizes', 'learning_rate', 'dropout_keep_prob', \
              'regularization_rate', 'batch_size', 'epochs']
print(sys.argv)
argv = sys.argv[1:]

FLAGS = {}

for i in [0, 5, 6]:
    FLAGS[argv_names[i]] = int(argv[i])
for i in [1]:
    FLAGS[argv_names[i]] = list(map(int, argv[i].split(',')))
for i in [2, 3, 4]:
    FLAGS[argv_names[i]] = float(argv[i])

def get_sentences(data):
    sentences = []
    for sample in data:
        sentences.append(sample[2].split(' '))
    return sentences

def checker(word):
    if word in stopwords:
        return False
    if word.isalpha():
        return True
    word = word.strip("-'")
    return "'" in word or '-' in word

def valid(pair):
    for word in pair:
        if checker(word) == False:
            return False
    return True

def n_gram(sentence, n):
    sentence = [word.lower() for word in sentence]
    n_gram = set()
    length = len(sentence)
    for i in range(length):
        for k in range(n):
            if i + k < length and valid(sentence[i:i+k+1]):
                n_gram.add(tuple(sentence[i:i+k+1]))
    return n_gram

def n_gram_dict(sentences, n):
    n_gram_dict = set()
    for sentence in sentences:
        n_gram_dict |= n_gram(sentence, n)
    return n_gram_dict

sentences = get_sentences(train_orig)
words = sorted(n_gram_dict(sentences, FLAGS['n']))

FLAGS['layer_sizes'] = [len(words)] + FLAGS['layer_sizes']

def get_vector(n_gram_dict, sentence, n):
    grams = n_gram(sentence, n)
    vector = list(map(lambda x: x in grams, n_gram_dict.copy()))
    return np.array(vector, np.int32)

def one_hot(lable):
    return np.array([0] * lable + [1] + [0] * (4 - lable))

train_x, train_y = [], []

for sample in train_orig:
    train_x.append(get_vector(words, sample[2].split(' '), FLAGS['n']))
    train_y.append(one_hot(int(sample[3])))
    
print("{}: training data loaded".format(datetime.datetime.now().isoformat()))

test_x = []
for sample in test_orig:
    test_x.append(get_vector(words, sample[2].split(' '), FLAGS['n']))

print("{}: test data loaded".format(datetime.datetime.now().isoformat()))
train_x , train_y = np.array(train_x), np.array(train_y)
test_x = np.array(test_x)

#training

fnn = FNN(layer_sizes = FLAGS['layer_sizes'], num_calsses = 5, \
          regularization_rate = FLAGS['regularization_rate'])

global_step = tf.Variable(0 , name = 'global_step', trainable = False)
optimizer = tf.train.AdamOptimizer(FLAGS['learning_rate'])
gradidents = optimizer.compute_gradients(fnn.loss)
train_op = optimizer.apply_gradients(gradidents, global_step = global_step)

#limit the memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
#config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

def train_step(x_batch, y_batch):
    feed_dict = {fnn.input_x : x_batch, fnn.input_y : y_batch, \
                 fnn.dropout_keep_prob : FLAGS['dropout_keep_prob']}
    _, step, loss, acc = sess.run([train_op, global_step, fnn.loss, fnn.accuracy], feed_dict = feed_dict)

    time_str = datetime.datetime.now().isoformat()

    if(step % 1000 == 0):
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))

def make_predictions(x_batch, y_batch):
    feed_dict = {fnn.input_x : x_batch, fnn.input_y : y_batch, \
                 fnn.dropout_keep_prob : 1.0}
    predictions = sess.run(fnn.predictions, feed_dict = feed_dict)

    return predictions

def get_batches(input_x, input_y, batch_size):

    m = input_x.shape[0]
    k = m // batch_size
    batches = []
    for i in range(k):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append((input_x[start:end], input_y[start:end]))
    if m % batch_size != 0:
        batches.append((input_x[k * batch_size:], input_y[k * batch_size:]))

    return batches

batches = get_batches(train_x, train_y, FLAGS['batch_size'])

del train_x
del train_y

#epochs iter
for k in range(FLAGS['epochs']):
    for batch in batches:
        x_batch, y_batch = batch
        train_step(x_batch, y_batch)

#test_y = make_predictions(test_x, [[0,0,0,0,0]] * len(test_x))
test_y = []
for test_case in test_x:
    test_y.append(make_predictions([test_case], [[0,0,0,0,0]])[0])
sess.close()

#output_result

output_name = 'result' + \
              '__gram_' + str(FLAGS['n']) + \
              '__layers_' + str(','.join(map(str,FLAGS['layer_sizes']))) + \
              '__keep_prob_' + str(FLAGS['dropout_keep_prob']) + \
              '__learning_rate_' + str(FLAGS['learning_rate']) + \
              '__regular_' + str(FLAGS['regularization_rate']) + \
              '__batch_size_' + str(FLAGS['batch_size']) + \
              '__epochs_' + str(FLAGS['epochs']) + \
              '.csv'

print('saved: ' + output_name)
output = [['PhraseId', 'Sentiment']]
id = 156060
for i in range(len(test_y)):
    output.append([id + i + 1, test_y[i]])
save_data(output_name, output)

'''
print(len(train_input))
print(len(test_input))
save_data('train.in', train_input)
save_data('test.in', test_input)
'''
