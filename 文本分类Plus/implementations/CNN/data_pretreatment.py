import csv
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


def update(data):
    words = set()
    seq_length = 0
    for sample in data:
        sentence = sample[2].split(' ')
        seq_length = max(seq_length, len(sentence))
        words.update(set([word.lower() for word in sentence]))
    return words, seq_length
words1, seq_length1 = update(train_orig)
words2, seq_length2 = update(test_orig)

words = words1 | words2
seq_length = max(seq_length1, seq_length2)

words = [' '] + list(words)
print(seq_length)
print(len(words))

train_input = []
for sample in train_orig:
    sentence = sample[2].split(' ')
    while len(sentence) < seq_length:
        sentence.append(' ')
    word_vec = []
    for word in sentence:
        word_vec.append(words.index(word.lower()))
    label = int(sample[3])
    label_vec = [0] * label + [1] + [0] * (4 - label)
    train_input.append([word_vec, label_vec])


test_input = []
for sample in test_orig:
    sentence = sample[2].split(' ')
    while len(sentence) < seq_length:
        sentence.append(' ')
    word_vec = []
    for word in sentence:
        word_vec.append(words.index(word.lower()))
    test_input.append(word_vec)
    
print(len(train_input))
print(len(test_input))
save_data('train.in', train_input)
save_data('test.in', test_input)
