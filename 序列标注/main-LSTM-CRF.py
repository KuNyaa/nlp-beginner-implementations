# -*- coding: utf-8 -*-

import csv
import nltk 
import numpy as np
import tensorflow as tf
import sklearn
import datetime
import conlleval
from sklearn import metrics

np.random.seed(3)

class FLAGS():
    vocab_freq_threshold = 1
    batch_size = 128
    hidden_size = 256
    embedding_size = 100
    epoches = 40
    learning_rate = 0.001
    num_layers = 1
    dropout_keep_prob = 0.5

class LSTM_CRF(object):
    def __init__(self, vocab_size, tags_size, embedding_size, hidden_size):

        input_sents = tf.placeholder(dtype=tf.int32, shape=[None, None])
        input_tags = tf.placeholder(dtype=tf.int32, shape=[None, None])
        input_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
        dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

        self.input_sents = input_sents
        self.input_tags = input_tags
        self.input_lengths = input_lengths
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob

        batch_size, max_time = input_sents.get_shape().as_list()

        with tf.variable_scope('embedding'):
            #load GloVe
            with open('./glove.6B.100d.txt', 'r', encoding='utf-8') as file:
                vocab_dict = FLAGS.vocab_to_id_map
                GloVe_W = np.random.uniform(-1.0, 1.0, [vocab_size, embedding_size])
                count = 0
                for line in file:
                    line = line.strip().split()
                    word, vec = line[0], line[1:]
                    vec = np.array([float(elem) for elem in vec])
                    assert(len(vec) == embedding_size)
                    if word in vocab_dict:
                        id = vocab_dict[word]
                        GloVe_W[id, :] = vec
                        count += 1
                print("load {} token vec from GloVe.".format(count))
            
            W = tf.get_variable(
                'embedding_W',
                shape=[vocab_size, embedding_size],
                initializer=tf.constant_initializer(GloVe_W))
                #initializer=tf.random_uniform_initializer(-0.1, 0.1))
            input_sents_emb = tf.nn.embedding_lookup(W, input_sents)
            inp_sents_emb_drop = tf.nn.dropout(input_sents_emb, dropout_keep_prob)

        with tf.variable_scope('LSTM'):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            #initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            #initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                inp_sents_emb_drop,
                sequence_length=input_lengths,
                #initial_state_fw=initial_state_fw,
                #initial_state_bw=initial_state_bw,
                dtype=tf.float32,
                time_major=False)
            outputs = tf.concat(outputs, 2) #[batch_size, max_time, hidden_size]
            outputs_drop = tf.nn.dropout(outputs, dropout_keep_prob)
            
            emission_scores = tf.layers.dense(
                outputs_drop, tags_size,
                activation=tf.nn.relu,
                use_bias=False) #[batch_size, max_time, tags_size]
        with tf.variable_scope('CRF'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                emission_scores, input_tags, input_lengths)

            predicted_seqs, viterbi_score = tf.contrib.crf.crf_decode(
                emission_scores, transition_params, input_lengths)

            self.predicted_seqs = predicted_seqs
            
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(-log_likelihood)

            trainable_vars = tf.trainable_variables()
            graidents = tf.gradients(loss, trainable_vars)
            max_grad_norm = 5
            clipped_grads, _ = tf.clip_by_global_norm(graidents, max_grad_norm)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, trainable=False)
            train_op = optimizer.apply_gradients(
                zip(clipped_grads, trainable_vars),
                global_step=global_step)
            self.loss = loss
            self.global_step = global_step
            self.train_op = train_op

        with tf.variable_scope('save-model'):
            self.saver = tf.train.Saver(max_to_keep=1)
            self.save_path = "./checkpoints/model" + \
                             "--hidden-size-" + str(FLAGS.hidden_size) + \
                             "--num_layers-" + str(FLAGS.num_layers) + \
                             "--batch_size-" + str(FLAGS.batch_size) + \
                             "--learning-rate-" + str(FLAGS.learning_rate) + \
                             "--dropout-keep-prob-" + str(FLAGS.dropout_keep_prob) + \
                             "/"
            self.best_fscore = 0
            
                
    def train(self, sess, dataset_train, dataset_dev, lr_rate):
        
        
        iterator = dataset_train.make_one_shot_iterator()
        next_batch_op = iterator.get_next()

        while True:
            try:
                next_batch = sess.run(next_batch_op)
                input_x, input_y, input_lengths = next_batch
            except:
                break
            feed_dict = {self.input_sents:input_x, self.input_tags:input_y,
                         self.input_lengths:input_lengths, self.learning_rate:lr_rate,
                         self.dropout_keep_prob:FLAGS.dropout_keep_prob}
            _, loss, global_step = sess.run(
                [self.train_op, self.loss, self.global_step],
                feed_dict=feed_dict)

            if global_step % 500 == 0:
                time_str = datetime.datetime.now().isoformat()
                train_P, train_R, train_F1 = self.eval(sess, dataset_train)
                dev_P, dev_R, dev_F1 = self.eval(sess, dataset_dev)
                
                print("{}: step:{}  loss:{}".format(time_str, global_step, loss))
                print("train  P:{}  R:{}  F1:{}".format(train_P, train_R, train_F1))
                print("dev  P:{} R:{} F1:{}".format(dev_P, dev_R, dev_F1))

                if dev_F1 > self.best_fscore:
                    self.save(sess)
                    self.best_fscore = dev_F1

    def eval(self, sess, dataset):
        iterator = dataset.make_one_shot_iterator()
        next_batch_op = iterator.get_next()
        IOB_output = []
        while True:
            try:
                next_batch = sess.run(next_batch_op)
                input_x, input_y, input_lengths = next_batch
            except:
                break
            feed_dict = {self.input_sents:input_x, self.input_lengths:input_lengths,
                         self.dropout_keep_prob:1.0}
            predicted_seqs = sess.run(self.predicted_seqs, feed_dict=feed_dict)

            batch_size = predicted_seqs.shape[0]

            for i in range(batch_size):
                seq_length = input_lengths[i]
                to_tag = FLAGS.id_to_tag_map
                for j in range(seq_length):
                    line = ['-'] * 2 + [to_tag[input_y[i, j]]] + [to_tag[predicted_seqs[i, j]]]
                    IOB_output.append(' '.join(line))
                IOB_output.append('\n')

        print(len(IOB_output))
        print(IOB_output[:10])

        return conlleval.evaluate(IOB_output)
                
    def save(self, sess):
        global_step = sess.run(self.global_step)
        self.saved_global_setp = global_step
        self.saver.save(sess, self.save_path + 'ckpt', global_step)
        print('Model saved.')

    def load(self, sess):
        model_file=tf.train.latest_checkpoint(self.save_path)
        self.saver.restore(sess, model_file)
        print('Model restored.')
        
def read_data(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as file:
        sent, tag = [], []
        for line in file:
            line = line.strip().split()
            if not line:
                samples.append((sent, tag))
                sent, tag = [], []
            else:
                sent.append(line[0].lower())
                tag.append(line[3])
    if sent:
        samples.append((sent, tag))
    return samples

def encode(sent, id_map):
    sent_encoded = []
    for word in sent:
        if word not in id_map:
            sent_encoded.append(id_map['<UNK>'])
        else:
            sent_encoded.append(id_map[word])
    return sent_encoded

def make_dataset(path, data, vocab_to_id_map, tag_to_id_map, time_major=False, shuffle=False):
    path = "./temp/" + path
    data_x, data_y = [], []
    for sent, tag in data:
            data_x.append(encode(sent, vocab_to_id_map))
            data_y.append(encode(tag, tag_to_id_map))
    def make_tf_CSVdataset(path, data):
        with open(path + '.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        dataset = tf.data.TextLineDataset(path + '.csv')
        dataset = dataset.map(lambda string: tf.string_split([string], delimiter=',').values)
        dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
        return dataset
    dataset_x = make_tf_CSVdataset(path + '_x', data_x)
    dataset_y = make_tf_CSVdataset(path + '_y', data_y)

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
    dataset = dataset.map(lambda x, y: (x, y, tf.size(x)))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    
    dataset = dataset.padded_batch(
        FLAGS.batch_size,
        padded_shapes=([None], [None], []),
        padding_values=(vocab_to_id_map['<PAD>'], tag_to_id_map['O'], 0))

    if time_major:
        dataset = dataset.map(lambda x, y, z: (tf.transpose(x), y, z))
    
    return dataset


path = './data/conll2003-IOBES/'

train_raw = read_data(path + 'eng.train')
dev_raw = read_data(path + 'eng.testa')
test_raw = read_data(path + 'eng.testb')
glove_raw = read_data('./glove.6B.100d.txt')

#vocab_fdist = nltk.FreqDist([word for (sent, _) in train_raw for word in sent])
#vocab = [word for word, freq in vocab_fdist.items() if freq >= FLAGS.vocab_freq_threshold]
vocab = [word for sent, _ in glove_raw for word in sent]
#vocab = [word for sent, _ in glove_raw for word in sent if word in vocab] 
vocab = ['<PAD>', '<UNK>'] + vocab
tags = sorted(set([tag for (_, sent) in train_raw for tag in sent]))
print(len(glove_raw[0][0]))
print(len(vocab), len(tags))
print(vocab[:10], '\n', vocab[-10:])
print(tags)
vocab_to_id_map = {word:id for id, word in enumerate(vocab)}
id_to_vocab_map = {id:word for id, word in enumerate(vocab)}
tag_to_id_map = {tag:id for id, tag in enumerate(tags)}
id_to_tag_map = {id:tag for id, tag in enumerate(tags)}
FLAGS.vocab_size = len(vocab)
FLAGS.tags_size = len(tags)
FLAGS.vocab_to_id_map = vocab_to_id_map
FLAGS.id_to_vocab_map = id_to_vocab_map
FLAGS.id_to_tag_map = id_to_tag_map

dataset_train = make_dataset('train', train_raw, vocab_to_id_map, tag_to_id_map, shuffle=True)
dataset_dev = make_dataset('dev', dev_raw, vocab_to_id_map, tag_to_id_map)
dataset_test = make_dataset('test', test_raw, vocab_to_id_map, tag_to_id_map)

model = LSTM_CRF(FLAGS.vocab_size, FLAGS.tags_size, FLAGS.embedding_size, FLAGS.hidden_size)

config = tf.ConfigProto(log_device_placement=False)
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
with tf.device('/gpu'):
    sess = tf.Session(config=config)
#sess = tf.Session()
sess.run(tf.global_variables_initializer())


lr_decay_threshold = 1 * FLAGS.epoches // 3
lr_decay_rate = FLAGS.learning_rate / (FLAGS.epoches - lr_decay_threshold)
learning_rate = FLAGS.learning_rate

for k in range(FLAGS.epoches):
    if k > lr_decay_threshold:
        learning_rate -= lr_decay_rate
    model.train(sess, dataset_train, dataset_dev, learning_rate)
    
model.load(sess)
test_P, test_R, test_F = model.eval(sess, dataset_test)
print("Test: P:{}  R:{}  F:{}".format(test_P, test_R, test_F))
sess.close()
