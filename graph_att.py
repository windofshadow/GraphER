import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
import math
import sys
import torch
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


if len(sys.argv) != 2:
    sys.exit("Use: python build_graph.py <dataset>")

datasets = ['Amazon-Google', 'DBLP-GoogleScholar']
# build corpus
dataset = sys.argv[1]
schema = [0,0,1] #0:compostional attribute 1:non-compositional or singular

if dataset not in datasets:
    sys.exit("wrong dataset name")

# solve unknown word by uniform(-0.25,0.25)
def _add_unknown_words_by_uniform(k):
    return np.random.uniform(-0.25, 0.25, k).round(6).tolist()

def _load_embedding(vocab,path, embed_dim):
    glove = []
    glove2id = {}
    print("**************** loading pre_trained word embeddings    *******************")
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            values = line.split(" ")
            word = values[0]
            glove2id[word] = len(glove2id)
            vector = [float(val) for val in values[1:]]
            glove.append(vector)
    embedding = []
    for i in range(len(vocab)):
        word = list(vocab.keys())[list(vocab.values()).index(i)]
        if word in glove2id:
            embedding.append(glove[glove2id[word]])
        else:
            embedding.append(_add_unknown_words_by_uniform(embed_dim))
    pretrained_emb = torch.FloatTensor(embedding)
    return pretrained_emb

def load_fasttext(vocab, fname, embed_dim):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])

    for i in range(len(vocab)):
        word = list(vocab.keys())[list(vocab.values()).index(i)]
        if word in data:
            embedding.append(data[word])
        else:
            embedding.append(_add_unknown_words_by_uniform(embed_dim))
    pretrained_emb = torch.FloatTensor(embedding)
    return pretrained_emb

def _assign_embedding(vocab_size,embed_dim):
    embedding = []
    for i in range(vocab_size):
        embedding.append(_add_unknown_words_by_uniform(embed_dim))
    pretrained_emb = torch.FloatTensor(embedding)
    return pretrained_emb


def readData():
    file_a = "Structured/" + dataset + "/tableA.csv"
    file_b = "Structured/" + dataset + "/tableB.csv"
    file_train = "Structured/" + dataset + "/train.csv"
    file_dev = "Structured/" + dataset + "/valid.csv"
    file_test = "Structured/" + dataset + "/test.csv"

    def remove_stopwords(sent):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sent)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return filtered_sentence

    def readTable(file):
        data = {}
        with open(file, encoding='utf-8') as f:
            reader = csv.reader(f)
            for r in reader:
                data[r[0]] = [remove_stopwords(sent) for sent in r[1:]]
                #for sent in r[1:]:
                #    fsent = remove_stopwords(sent)
                #    data[r[0]].append(fsent)
            del (data['id'])
        return data

    def readMapping(file):
        mapping = []
        with open(file) as f:
            reader = csv.reader(f)
            for r in reader:
                mapping.append((r[0], r[1], r[2]))
        del (mapping[0])
        return mapping

    def check(mapping, ta, tb):
        for i in range(len(mapping)):
            if mapping[i][0] not in ta or mapping[i][1] not in tb:
                del (mapping[i])
                print("!!!")

    table_a = readTable(file_a)
    table_b = readTable(file_b)
    train = readMapping(file_train)
    dev = readMapping(file_dev)
    test = readMapping(file_test)
    check(train, table_a, table_b)
    check(dev, table_a, table_b)
    check(test, table_a, table_b)
    return table_a, table_b, train, dev, test

ta,tb,train,dev,test = readData()

def merge_table(ta, tb):
    all = ta.copy()
    offset = len(ta)
    for (id,value) in tb.items():
        new_id = offset + int(id)
        all[str(new_id)] = value
    return offset, all

offset, all_doc = merge_table(ta, tb)

def convert_mapping(mapping, offset, cosine_loss = True):
    new_mapping = []
    for (x,y,l) in mapping:
        new_y = int(y) + offset
        if cosine_loss == True:
            if l == '0':
                l = '-1'
        new_mapping.append((int(x), new_y, int(l)))
    return new_mapping

train = convert_mapping(train, offset)
dev = convert_mapping(dev, offset)
test = convert_mapping(test, offset)

doc_num = len(all_doc)


def handle(doc_content_list, row, col, weight, previous_layer_length, att_type_mask):

    # build vocab
    word_freq = {}
    word_set = set()
    for (id, doc_words) in doc_content_list.items():
        words = doc_words
        for word in words:
            word = str(word)
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    temp = []
    vocab = list(word_set)
    if str(temp) in vocab:
        vocab.remove(str(temp))
    vocab_size = len(vocab)

    # build a word to doc list
    word_doc_list = {}

    for (id, content) in doc_content_list.items():
        appeared = set()
        for word in content:
            word = str(word)
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(id)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [id]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i  # !!! local id

    doc_num = len(doc_content_list)

    def word2word_pmi(doc_content_list, att_type_mask):
        # word co-occurence with context windows
        window_size = 20
        windows = []

        for (id, doc_words) in doc_content_list.items():
            if att_type_mask[int(id)] == 1:
                continue
            words = doc_words
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                # print(length, length - window_size + 1)
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)
                    # print(window)

        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                word = str(window[i])
                if word in appeared:
                    continue
                if word in word_window_freq:
                    word_window_freq[word] += 1
                else:
                    word_window_freq[word] = 1
                appeared.add(word)

        word_pair_count = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = str(window[i])
                    if word_i in word_id_map:
                        word_i_id = word_id_map[word_i]
                    word_j = str(window[j])
                    if word_j in word_id_map:
                        word_j_id = word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
        # pmi as weights

        num_window = len(windows)

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]
            pmi = log((1.0 * count / num_window) /
                      (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
            if pmi <= 0:
                continue
            row.append(previous_layer_length + i)
            col.append(previous_layer_length + j)
            weight.append(pmi)
        #return row, col, weight

    # handle different type: description -> word2doc; values -> compare
    def word2word_range_window(doc_content_list, att_type_mask):
        # word co-occurence with context windows
        cut_off = 0.5
        prices = []
        price_ids = []

        for (id, doc_words) in doc_content_list.items():
            words = doc_words
            if att_type_mask[int(id)] == 1:
                prices.append(float(words[0]))
                price_ids.append(word_id_map[words[0]])

        for i in range(len(prices)):
            x = prices[i]
            for j in range(i+1, len(prices)):
                y = prices[j]
                diff_ration = pow(x-y, 2) / (x*y) #math.abs(x - y) / math.max(x, y)
                if diff_ration < cut_off:
                    row.append(previous_layer_length + price_ids[i])
                    col.append(previous_layer_length + price_ids[j])
                    weight.append(1.0 - diff_ration)
                    print(x ,y,diff_ration)



    word2word_pmi(doc_content_list, att_type_mask)
    word2word_range_window(doc_content_list, att_type_mask)

    # doc word frequency
    doc_word_freq = {}

    for (doc_id, doc_words) in doc_content_list.items():
        words = doc_words
        for word in words:
            if len(word) == 0:
                continue
            word = str(word)
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for (id, doc_words) in doc_content_list.items():
        words = doc_words
        doc_word_set = set()
        for word in words:
            if len(word) == 0:
                continue
            word = str(word)
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = id + ',' + str(j)
            freq = doc_word_freq[key]
            if int(id) < previous_layer_length:
                row.append(int(id))
            else:
                row.append(int(id) + vocab_size)
            col.append(previous_layer_length + j)
            idf = log(1.0 * len(doc_content_list) /
                      word_doc_freq[vocab[j]])
            #weight.append(freq * idf) !!!!!!!!!!!!!!!!!!
            weight.append(idf)
            doc_word_set.add(word)
    return vocab, vocab_size, word_id_map


row = []
col = []
weight = []

### handle doc-att
att_type_mask = {}
previous_layer_length = len(all_doc)
for i in range(len(all_doc)):
    att_type_mask[i] = 0
att_vocab, vocab_size, att_id_map = handle(all_doc, row, col, weight, previous_layer_length, att_type_mask)
doc_att_embed = _assign_embedding(previous_layer_length+vocab_size, 200)
node_size = previous_layer_length + vocab_size

### handle att-word
att_words = {}
att_type_mask = {}
for (id, atts) in all_doc.items():
    for i in range(len(atts)):
        att = atts[i]
        if len(att) == 0:
            continue
        att_str = str(att)
        id = att_id_map[att_str]
        id = str(previous_layer_length + id)
        att_type_mask[int(id)] = schema[i]
        att_words[id] = att
previous_layer_length = node_size
vocab, vocab_size, word_id_map = handle(att_words, row, col, weight, previous_layer_length, att_type_mask)
token_embed = _load_embedding(word_id_map,"Embedding/glove.6B.200d.txt", 200)
#token_embed = load_fasttext(word_id_map,"Embedding/wiki-news-300d-1M.vec", 300)
node_size = previous_layer_length + vocab_size

embedding = torch.cat((doc_att_embed, token_embed), dim=0)

adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

## doc att
doc_att = {}
for (id, atts) in all_doc.items():
    for att in atts:
        if len(att) == 0:
            att_id = -1
        else:
            att = str(att)
            att_id = att_id_map[att] + len(all_doc)
        if id in doc_att:
            doc_att[id].append(att_id)
        else:
            doc_att[id] = [att_id]

##  att_words
aw = {}
for (id, words) in att_words.items():
    aw[id] = []
    for word in words:
        aw[id].append(word_id_map[word]+previous_layer_length)

    #dump
def save():
    data = {
        'tableA_len': len(ta),
        'tableB_len': len(tb),
        'doc_len': len(ta) + len(tb),
        #'vocab': vocab,
        'vocab_size': vocab_size,
        'data': {
            'doc_content': all_doc,
            'odc_att': doc_att,
            'att_words': aw,
            'train': train,
            'dev': dev,
            'test': test,
            'embedding': embedding
            # 'label': [self.l.word2idx[l] for l in self.valid_labels]
        }
    }
    torch.save(data, 'data/' + dataset + '.info')
save()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()


