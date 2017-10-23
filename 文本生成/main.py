# -*- coding: utf-8 -*-  
import codecs
import datetime
import numpy as np
import tensorflow as tf

    
class RNN(object):

    def __init__(self, vocab_size, hidden_size, learning_rate):

        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        batch_size = tf.shape(self.input_x)[0]

        cell = tf.contrib.rnn.GRUCell(hidden_size)
        init_state = cell.zero_state(batch_size, tf.float32)

        input = tf.one_hot(self.input_x, vocab_size)
        seq_lengths = tf.reduce_sum(tf.reduce_max(tf.sign(input), 2), 1)
        in_state = tf.placeholder_with_default(init_state, shape=[None, hidden_size],
                                               name='in_state')

        output, out_state = tf.nn.dynamic_rnn(cell, input, seq_lengths, in_state)
        self.in_state, self.out_state = in_state, out_state

        output_dropouted = tf.nn.dropout(output, self.dropout_keep_prob)
        scores = tf.contrib.layers.fully_connected(output_dropouted, vocab_size, None)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores[:, :-1],
                                                                           labels=input[:, 1:]))

        self.sample = tf.multinomial(tf.exp(scores[:, -1]), 1)[:, 0]

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
    def train(self, sess, input_x, dropout_keep_prob):
        feed_dict = {self.input_x:input_x, self.dropout_keep_prob:dropout_keep_prob}
        sess.run(self.train_op, feed_dict=feed_dict)

    def step(self, sess):
        return sess.run(self.global_step)

    def eval_loss(self, sess, input_xs):
        total_loss = 0
        for input_x in input_xs:
            feed_dict = {self.input_x:input_x, self.dropout_keep_prob:1.0}
            total_loss += sess.run(self.loss, feed_dict=feed_dict)

        return total_loss / len(input_xs)

    def generate(self, sess, start, genr_length, vocab_size, epsilon):
        sentence = start
        state = None
        for _ in range(genr_length):
            batch = [[sentence[-1]]] 
            feed_dict = {self.input_x:batch, self.dropout_keep_prob:1.0}
            if state is not None:
                feed_dict.update({self.in_state:state})
            id, state = sess.run([self.sample, self.out_state], feed_dict=feed_dict)
            next = id[0]
            if vocab[next - 1].isalpha() and np.random.rand() < epsilon:
                next = np.random.randint(0, vocab_size) + 1
                while not vocab[next - 1].isalpha():
                    next = np.random.randint(0, vocab_size) + 1
            sentence.append(next)

        return sentence

def load_data(path):
    text = []
    with codecs.open(path, 'r', 'utf-8') as file:
        for line in file:
            text.append(line)
    return text


def encode(text, vocab):
    text_encoded = [vocab.index(char) + 1 for char in text]
    return text_encoded

def decode(text, vocab):
    text_decoded = ''.join([vocab[id - 1] for id in text])
    return text_decoded

'''
def pretreatment(text):
    ids = [i for (i, sentence) in enumerate(text) if sentence is '\n']
    cropus = []
    for i in range(len(ids) - 1):
        cropus.append(''.join(text[ids[i] + 1:ids[i + 1] + 1]))
    return cropus
'''

def get_batches(text, seq_length, step_size, batch_size):
    full_batch = []
    for i in range(0, len(text) - seq_length + 1, step_size):
        full_batch.append(text[i:i + seq_length])
    size = len(full_batch)
    num_blocks = size // batch_size
    mini_batches = []
    for i in range(num_blocks):
        begin, end = i * batch_size, (i + 1) * batch_size
        mini_batches.append(full_batch[begin:end])
    if size % batch_size != 0:
        begin, end = num_blocks * batch_size, size + 1
        mini_batches.append(full_batch[begin:end])

    return mini_batches


def generation(sess, rnn, vocab, start, genr_length, epsilon):

    start_encoded = encode(start, vocab)
    sentence = rnn.generate(sess, start_encoded, genr_length, len(vocab), epsilon)
    sentence_decoded = decode(sentence, vocab)

    return sentence_decoded

def training(sess, rnn, text, vocab, seq_length, step_size, epochs, batch_size , genr_length, epsilon, dropout_keep_prob):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(tf.global_variables_initializer())
    vacab_size = len(vocab)

    file = codecs.open("log.txt", "w", "utf-8")  
    
    mini_batches = get_batches(text, seq_length, step_size, batch_size)
    print(len(text), step_size)
    print(len(mini_batches))
    for _ in range(epochs):
        for batch in mini_batches:
            rnn.train(sess, batch, dropout_keep_prob)
            global_step = rnn.step(sess)
            if global_step % 500 == 0:
                saver.save(sess, './checkpoints/checkpoint', global_step) 
                time_str = datetime.datetime.now().isoformat()
                loss = rnn.eval_loss(sess, mini_batches)
                start_char = vocab[np.random.randint(0, vacab_size)]
                genr_sentence = generation(sess, rnn, vocab, start_char, genr_length, epsilon)
                print("\n{}: step:{}  loss: {}".format(time_str, global_step, loss))
                #print("Generate sample: {}".format(genr_sentence).decode('utf-8'))
                file.write("\n{}: step:{}  loss: {}".format(time_str, global_step, loss))
                file.write("Generate sample: {}".format(genr_sentence))

    file.close()



path = "./data/poetryFromTang.txt"
text = ''.join(load_data(path))
vocab = ''.join(sorted(set(list(text))))
text = encode(text, vocab)
                
epochs = 150
hidden_size = 128
learning_rate = 0.001
dropout_keep_prob = 0.45
batch_size = 64
epsilon = 0.0
seq_length = 64
step_size = 8
genr_length = 300
vocab_size = len(vocab)


rnn = RNN(vocab_size=vocab_size, hidden_size=hidden_size, learning_rate=learning_rate)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

training(sess, rnn, text, vocab, seq_length, step_size, epochs, batch_size, 100, epsilon, dropout_keep_prob)

file = codecs.open("generate.txt", "w", "utf-8")  
for _ in range(10):
    start = vocab[np.random.randint(0, vocab_size)]
    file.write("\n\nCase #" + start + ":\n")
    file.write(generation(sess, rnn, vocab, start, genr_length, epsilon))
file.close()

