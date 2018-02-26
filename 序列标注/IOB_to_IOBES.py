
path = 'output.txt.IOBES'

data = []

def IOB_to_IOBES(sent):
    word, POS, Chunk, NER = zip(*sent)

    def IOBES(tag, length):
        if length == 1:
            return ['S-' + tag]
        else :
            return ['B-' + tag] + ['I-' + tag] * (length - 2) + ['E-' + tag]

    NER_IOBES = []
    cur_tag = None
    length = 0
    for tag in NER:
        if tag == 'O':
            if cur_tag:
                NER_IOBES += IOBES(cur_tag, length)
                cur_tag = None
                length = 0
            NER_IOBES.append('O')
        else :
            if not cur_tag:
                cur_tag = tag[2:]
                length = 1
            else :
                if cur_tag == tag[2:] and tag[0] == 'I':
                    length += 1
                else:
                    NER_IOBES += IOBES(cur_tag, length)
                    cur_tag = tag[2:]
                    length = 1

    if cur_tag:
        NER_IOBES += IOBES(cur_tag, length)
    assert(len(NER) == len(NER_IOBES))
    sent_IOBES = list(zip(word, POS, Chunk, NER_IOBES))
    
    return ''.join(map(lambda x: ' '.join(x) + '\n', sent_IOBES)) + '\n'

with open(path, 'r') as file:
    sent = []
    cnt = 0
    for line in file:
        cnt += 1
        line = line.strip()
        if not line:
            data.append(IOB_to_IOBES(sent))
            sent = []
        else:
            sent.append(line.split())

print(len(data), cnt)
print(data[:3])
with open(path + '.IOBES', 'w') as file:
    file.write(''.join(data))
