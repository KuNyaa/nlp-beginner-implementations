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
    batch_size = 128
    char_embedding_size = 30
    word_embedding_size = 100
    char_hidden_size = 25
    word_hidden_size = 256
    char_layer_num = 1
    word_layer_num = 3
    epoches = 50
    learning_rate = 0.002
    num_layers = 1
    dropout_keep_prob = 0.5

class GRU_LSTM_CRF(object):
    def __init__(self,chars_size, vocab_size, tags_size, char_embedding_size,
                 word_embedding_size, char_hidden_size, word_hidden_size,
                 char_layer_num, word_layer_num):

        input_sents = tf.placeholder(dtype=tf.int32, shape=[None, None])
        input_words = tf.placeholder(dtype=tf.int32, shape=[None, None, None])
        input_tags = tf.placeholder(dtype=tf.int32, shape=[None, None])
        input_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        input_words_lens = tf.placeholder(dtype=tf.int32, shape=[None, None])
        learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
        dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

        self.input_sents = input_sents
        self.input_words = input_words
        self.input_tags = input_tags
        self.input_lengths = input_lengths
        self.input_words_lens = input_words_lens
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob

        batch_size = tf.shape(input_words)[0]
        max_time = tf.shape(input_words)[1]
        max_word_len = tf.shape(input_words)[2]


        words_length = batch_size * max_time
        
        input_words = tf.reshape(input_words, shape=[words_length, max_word_len])
        input_words_lens = tf.reshape(input_words_lens, shape=[words_length])
        
        with tf.variable_scope('embedding'):
            #char embedding
            char_embedding = tf.get_variable(
                'char_embedding',
                shape=[chars_size, char_embedding_size],
                initializer = tf.random_uniform_initializer(-1.0, 1.0))
            input_words_emb = tf.nn.embedding_lookup(char_embedding, input_words)
            inp_words_emb_drop = tf.nn.dropout(input_words_emb, dropout_keep_prob)
            
            #word embedding
            with open('./glove.6B.100d.txt', 'r', encoding='utf-8') as file:
                vocab_dict = FLAGS.vocab_to_id_map
                GloVe_W = np.random.uniform(-1.0, 1.0, [vocab_size, word_embedding_size])
                UNK_vec = np.zeros(word_embedding_size)
                count = 0
                for line in file:
                    line = line.strip().split()
                    word, vec = line[0], line[1:]
                    vec = np.array([float(elem) for elem in vec])
                    assert(len(vec) == word_embedding_size)
                    if word in vocab_dict:
                        id = vocab_dict[word]
                        GloVe_W[id, :] = vec
                        count += 1
                        UNK_vec += vec
                UNK_vec /= count
                GloVe_W[vocab_dict['<UNK>'], :] = UNK_vec
                print("load {} token vec from GloVe.".format(count))
            
            word_embedding = tf.get_variable(
                'word_embedding',
                shape=[vocab_size, word_embedding_size],
                initializer=tf.constant_initializer(GloVe_W),
                trainable=False)
                #initializer=tf.random_uniform_initializer(-0.1, 0.1))
            input_sents_emb = tf.nn.embedding_lookup(word_embedding, input_sents)
            
            with tf.variable_scope('GRU'):
                cell_fw = tf.nn.rnn_cell.GRUCell(char_hidden_size)
                cell_bw = tf.nn.rnn_cell.GRUCell(char_hidden_size)
                _, char_embs = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    inp_words_emb_drop,
                    sequence_length=input_words_lens,
                    dtype=tf.float32,
                    time_major=False)
                char_embs = tf.concat(char_embs, 1)
                char_embs = tf.reshape(char_embs, shape=[batch_size, max_time, 2 * char_hidden_size])
            input_sents_emb = tf.concat((input_sents_emb, char_embs), 2)
            inp_sents_emb_drop = tf.nn.dropout(input_sents_emb, dropout_keep_prob)
            
        with tf.variable_scope('LSTM'):
            def LSTM_stacked_cells():
                return tf.contrib.rnn.MultiRNNCell(
                    [tf.nn.rnn_cell.BasicLSTMCell(word_hidden_size) for _ in range(word_layer_num)])
            cell_fw = LSTM_stacked_cells()
            cell_bw = LSTM_stacked_cells()
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                inp_sents_emb_drop,
                sequence_length=input_lengths,
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
                             "--hidden-size-" + str(FLAGS.word_hidden_size) + \
                             "--num_layers-" + str(FLAGS.num_layers) + \
                             "--batch_size-" + str(FLAGS.batch_size) + \
                             "--learning-rate-" + str(FLAGS.learning_rate) + \
                             "--dropout-keep-prob-" + str(FLAGS.dropout_keep_prob) + \
                             "/"
            self.best_fscore = 0
            
                
    def train(self, sess, dataset_train, dataset_dev, lr_rate):
        
        for batch in dataset_train:
            input_sents, input_words, input_tags, input_lengths, input_words_lens = batch 
            feed_dict = {self.input_sents:input_sents, self.input_words:input_words,
                         self.input_tags:input_tags, self.input_lengths:input_lengths,
                         self.input_words_lens:input_words_lens, self.learning_rate:lr_rate,
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
        IOB_output = []
        for batch in dataset:
            input_sents, input_words, input_tags, input_lengths, input_words_lens = batch 
            feed_dict = {self.input_sents:input_sents, self.input_words:input_words,
                         self.input_tags:input_tags, self.input_lengths:input_lengths,
                         self.input_words_lens:input_words_lens, self.dropout_keep_prob:1.0}
            
            predicted_seqs = sess.run(self.predicted_seqs, feed_dict=feed_dict)

            batch_size = predicted_seqs.shape[0]

            for i in range(batch_size):
                seq_length = input_lengths[i]
                to_tag = FLAGS.id_to_tags_map
                for j in range(seq_length):
                    line = ['-'] * 2 + [to_tag[input_tags[i, j]]] + [to_tag[predicted_seqs[i, j]]]
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
                sent.append(line[0])
                tag.append(line[3])
    if sent:
        samples.append((sent, tag))
    return samples

def encode(sent, vocab_map, istag=False):
    sent_encoded = []
    words_encoded = []
    char_map = FLAGS.chars_to_id_map
    for word in sent:
        if not istag:
            words_encoded.append([char_map[char] for char in word if char in char_map])
            word = word.lower()
        if word not in vocab_map:
            sent_encoded.append(vocab_map['<UNK>'])
        else:
            sent_encoded.append(vocab_map[word])
            
    return sent_encoded, words_encoded



path = './data/conll2003-IOBES/'

train_raw = read_data(path + 'eng.train')
dev_raw = read_data(path + 'eng.testa')
test_raw = read_data(path + 'eng.testb')
glove_raw = read_data('./glove.6B.100d.txt')

chars = ['<PAD>'] + sorted(set(char for sent, _ in train_raw for word in sent for char in word))
vocab = ['<PAD>', '<UNK>'] + [word for sent, _ in glove_raw for word in sent]
tags = sorted(set([tag for (_, sent) in train_raw for tag in sent]))
print(len(chars), len(vocab), len(tags))
print(chars)
print(tags)

def make_mapping(vocab):
    vocab_to_id_map = {word:id for id, word in enumerate(vocab)}
    id_to_vocab_map = {id:word for id, word in enumerate(vocab)}
    
    return vocab_to_id_map, id_to_vocab_map

FLAGS.chars_to_id_map, FLAGS.id_to_vocab_map = make_mapping(chars)
FLAGS.vocab_to_id_map, FLAGS.id_to_vocab_map = make_mapping(vocab)
FLAGS.tags_to_id_map, FLAGS.id_to_tags_map = make_mapping(tags)
FLAGS.chars_size = len(chars)
FLAGS.vocab_size = len(vocab)
FLAGS.tags_size = len(tags)

def make_dataset(data):
    dataset = []

    for sent, tag in data:
        sent, words = encode(sent, FLAGS.vocab_to_id_map)
        tag, _ = encode(tag, FLAGS.tags_to_id_map, istag=True)
        dataset.append((sent, words, tag))

    return dataset

dataset_train = make_dataset(train_raw) 
dataset_dev = make_dataset(dev_raw)
dataset_test = make_dataset(test_raw)

def make_batch(dataset, batch_size, shuffle=False):
    dataset_batched = []
    data_size = len(dataset)
    if shuffle:
        np.random.shuffle(dataset)
        
    def batch(data):
        def padding(sents, length, padding_value):
            return list(map(lambda x: x + (length - len(x)) * [padding_value], sents))
        
        sent_batch, words_batch, tag_batch = zip(*data)

        sent_lengths = [len(sent) for sent in sent_batch]
        max_sent_len = max(sent_lengths)
        sent_batch_padded = padding(sent_batch, max_sent_len, 0)

        words_lengths = [[len(word) for word in words] for words in words_batch]
        words_lengths = padding(words_lengths, max_sent_len, 0)
        max_words_len = max([max(lengths) for lengths in words_lengths])
        words_batch_padded = list(map(lambda x: padding(x, max_words_len, 0), words_batch))
        words_batch_padded = padding(words_batch_padded, max_sent_len, [0] * max_words_len)

        tag_batch_padded = padding(tag_batch, max_sent_len, 0)

        batch_padded = (sent_batch_padded, words_batch_padded, tag_batch_padded, sent_lengths, words_lengths)
        return tuple(map(lambda x: np.array(x), batch_padded))
    
    for start in range(0, data_size, batch_size):
        dataset_batched.append(batch(dataset[start: min(start + batch_size, data_size)]))

    return dataset_batched


model = GRU_LSTM_CRF(FLAGS.chars_size, FLAGS.vocab_size, FLAGS.tags_size, FLAGS.char_embedding_size,
                     FLAGS.word_embedding_size, FLAGS.char_hidden_size, FLAGS.word_hidden_size,
                     FLAGS.char_layer_num, FLAGS.word_layer_num)
config = tf.ConfigProto(log_device_placement=False)
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
with tf.device('/gpu'):
    sess = tf.Session(config=config)
#sess = tf.Session()
sess.run(tf.global_variables_initializer())


lr_decay_threshold = 1 * FLAGS.epoches // 2
lr_decay_rate = FLAGS.learning_rate / (FLAGS.epoches - lr_decay_threshold)
learning_rate = FLAGS.learning_rate

batch_size = FLAGS.batch_size
dataset_dev_batched = make_batch(dataset_dev, batch_size)
dataset_test_batched = make_batch(dataset_test, batch_size)

for k in range(FLAGS.epoches):
    if k > lr_decay_threshold:
        learning_rate -= lr_decay_rate
    dataset_train_batched = make_batch(dataset_train, batch_size, shuffle=True)
    model.train(sess, dataset_train_batched, dataset_dev_batched, learning_rate)
    
model.load(sess)
test_P, test_R, test_F = model.eval(sess, dataset_test_batched)
print("Test: P:{}  R:{}  F:{}".format(test_P, test_R, test_F))
sess.close()
