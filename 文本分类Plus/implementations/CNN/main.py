import tensorflow as tf
import numpy as np
import sys
import csv
import random
import datetime


class CNN(object):
    '''a simple model for CNN for sentiment clasification'''
    def __init__ (self, seq_length, num_classes, dict_size, \
                  embedding_size, filter_sizes, num_filters):

        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = 'input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')

        #embedding layer 
        with  tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([dict_size, embedding_size], -1.0, 1.0), name = 'W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #convolution and max-pooling layer
        pooled_outputs = []
        for filter_size in filter_sizes :
            with tf.name_scope('conv-maxpool-' + str(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = 'W')
                b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = 'b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides = [1, 1, 1, 1], \
                                    padding = 'VALID', name = 'conv')
                #activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name = 'relu')

                #max-pooling
                pooled = tf.nn.max_pool(h, ksize = [1, seq_length - filter_size + 1, 1, 1], \
                                        strides = [1, 1, 1, 1], padding = 'VALID', name = 'pool')

                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        #dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        #softmax_regression
        with tf.name_scope('scores'):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev = 0.1), name = 'W')
            b = tf.Variable(tf.constant(1.0, shape = [num_classes]), name = 'b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = 'scores')
            self.predictions = tf.argmax(self.scores, 1, name = 'predictions')
            
        #loss
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y))

        #accuracy
        with tf.name_scope('accuracy'):
            correct_case = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_case, 'float'), name = 'accuracy')

#input arguments
argv_names = ['train_file_name','test_file_name','dict_size','embedding_size',\
        'filter_sizes','num_filters','dropout_keep_prob','batch_size','epochs']
FLAGS = {}
print(sys.argv)
argv = sys.argv[1:]
for i in [0, 1]:
    FLAGS[argv_names[i]] = argv[i]
for i in [2, 3, 5, 7, 8]:
    FLAGS[argv_names[i]] = int(argv[i])
for i in [4]:
    FLAGS[argv_names[i]] = list(map(int, argv[i].split(',')))
for i in [6]:
    FLAGS[argv_names[i]] = float(argv[i])

    

#input training and test data
def load_data(path):
    '''Read data from a csv file'''
    csvfile=open(path, 'r')
    csvreader = csv.reader(csvfile, dialect='excel')
    data = []
    for row in csvreader:
        data.append(list(row))
    csvfile.close()
    return data

def convert(data):
    result = []
    for s in data:
        result.append(s.strip('[]').split(','))
    return result

train_orig = load_data(FLAGS['train_file_name'])

'''
train_size = int(len(train_orig) * 0.9)
train_x, train_y = zip(*train_orig[:train_size])
dev_x, dev_y = zip(*train_orig[train_size:])
'''

train_x, train_y = zip(*train_orig)

test_x = load_data(FLAGS['test_file_name'])

train_x, train_y = np.array(convert(train_x), np.int32), np.array(convert(train_y), np.int32)

#dev_x, dev_y = np.array(convert(dev_x), np.int32), np.array(convert(dev_y), np.int32)
test_x, test_y = np.array(test_x, np.int32), np.array(test_y, np.int32)


cnn = CNN(seq_length = train_x.shape[1], num_classes = 5, dict_size = FLAGS['dict_size'], \
          embedding_size = FLAGS['embedding_size'], filter_sizes = FLAGS['filter_sizes'], \
          num_filters = FLAGS['num_filters'])

global_step = tf.Variable(0, name = 'global_step', trainable = False)
optimizer = tf.train.AdamOptimizer(1e-4)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

sess = tf.Session();

sess.run(tf.global_variables_initializer())

def train_step(x_batch, y_batch):
    feed_dict = {cnn.input_x : x_batch, cnn.input_y : y_batch, \
                 cnn.dropout_keep_prob : FLAGS['dropout_keep_prob']}
    _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

    time_str = datetime.datetime.now().isoformat()
    if(step % 1000 == 0):
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


def test_step(x_batch, y_batch):
    feed_dict = {cnn.input_x : x_batch, cnn.input_y : y_batch, \
                 cnn.dropout_keep_prob : 1.0}
    loss, accuracy = sess.run([cnn.loss, cnn.accuracy], feed_dict = feed_dict)

    time_str = datetime.datetime.now().isoformat()

    print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))

def make_predictions(x_batch, y_batch):
    
    feed_dict = {cnn.input_x : x_batch, cnn.input_y : y_batch, \
                 cnn.dropout_keep_prob : 1.0}
    predictions = sess.run(cnn.predictions, feed_dict = feed_dict)

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

# training
for k in range(FLAGS['epochs']):
    for batch in batches:
        x_batch, y_batch = batch
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
    '''
    if (k + 1) % 100 == 0:
        print("Dev test:")
        test_step(dev_x, dev_y)
    '''
#test_step(dev_x, dev_y)
test_y = []
for test_case in test_x:
    test_y.append(make_predictions([test_case], [[0,0,0,0,0]])[0])
sess.close()
#output result
def save_data(path, data):
    '''Save data to a csv file'''
    csvfile=open(path, 'w', newline='')
    csvwriter = csv.writer(csvfile, dialect='excel')#delimiter=',')
    for result in data:
        csvwriter.writerow(result)
    csvfile.close()
    
output_name = 'result' + \
          '__embedding_' + str(FLAGS['embedding_size']) + \
          '__filters_' + str(','.join(map(str,FLAGS['filter_sizes']))) + \
          '__num_filters_' + str(FLAGS['num_filters']) + \
          '__keep_prob_' + str(FLAGS['dropout_keep_prob']) + \
          '__batch_size_' + str(FLAGS['batch_size']) + \
          '__epochs_' + str(FLAGS['epochs']) + \
          '.csv'

print('saved: ' + output_name)
output = [['PhraseId', 'Sentiment']]
id = len(train_orig)
for i in range(len(test_y)):
    output.append([id + i + 1, test_y[i]])
save_data(output_name, output)
