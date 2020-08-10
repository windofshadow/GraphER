import torch
import numpy as np
import scipy.sparse as sp
import torch.utils.data as Data
import pickle as pkl
import random


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    '''
        adj += torch.eye(len(adj))
        rowsum = torch.sum(adj, dim=1)
        rowsum = torch.rsqrt(rowsum)
        D = torch.diag(rowsum)
        temp = torch.mm(D, adj)
        return torch.mm(temp, D)
        '''
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    #return sparse_to_tuple(adj_normalized)
    return adj_normalized


def gen_rand_neg(k, isSource, r_id, size, smap, tmap):
    idx = []
    for i in range(k):
        if isSource == True:
            rn = random.randint(0, size)
            while rn in smap[r_id]:
                rn = random.randint(0, size)
            idx.append(rn)
        else:
            rn = random.randint(0, size)
            while rn in tmap[r_id]:
                rn = random.randint(0, size)
            idx.append(rn)
    return idx

def gen_rand_all_neg(k, smap, tmap, s_size, t_size):
    neg = []
    s_idx = []
    t_idx = []
    for i in range(k):
        rn = random.randint(0, s_size)
        s_idx.append(rn)
    for i in range(k):
        rn = random.randint(0, t_size)
        if s_idx[i] in smap:
            while rn in smap[s_idx[i]]:
                rn = random.randint(0, t_size)
        t_idx.append(rn)
    for i in range(k):
        neg.append((s_idx[i],t_idx[i],-1))
    return neg


def check_conflict(anker, isSource, rn, smap, tmap):
    if isSource == True:
        if rn not in smap[anker]:
            return True
        return False
    else:
        if rn not in tmap[anker]:
            return True
        return False

def label_batch_for_lr(batch):
    for i in range(len(batch)):
        if batch[i] == -1:
            batch[i] = 0
    return batch

def hard_negative(anker, encoding, base, k, anIsSource, r_id):
    DAt = [torch.dist(anker, encoding[i], 2) for i in range(len(encoding))]
    DAt = torch.stack(DAt, 0)
    sorted, indices = torch.sort(DAt)
    topk = []
    for i in range(len(encoding)):
        if sorted[i] > base:
            idx = []
            j = i
            for tt in range(k):
                while check_conflict(r_id, anIsSource, indices[j]) == False:
                    j += 1
                idx.append(j)
            topk = [encoding[j] for j in idx]
            break
    if len(topk) == 0:
        randneg = gen_rand_neg(k,anIsSource,r_id,len(encoding))
        topk = [encoding[j] for j in randneg]
    return torch.stack(topk,0)

def toTensor(train):
    S = []
    T = []
    L = []
    #tperm = torch.randperm(len(train))
    for i in range(len(train)):
        (s,t,l) = train[i]
        S.append(int(s))
        T.append(int(t))
        L.append(int(l))
    SI = torch.LongTensor(S)
    TI = torch.LongTensor(T)
    LI = torch.FloatTensor(L)
    return SI, TI, LI

def load_label():
    train = []
    dev = []
    test = []
    smap = {}
    tmap = {}
    with open('data/AG_train','r') as f:
        lines = f.readlines()
        for line in lines:
            pair = line.strip().split('\t')
            tup = (pair[0],pair[1],1)
            train.append(tup)
            if pair[0] not in smap:
                smap[pair[0]] = set()
            smap[pair[0]].add(pair[1])
            if pair[1] not in tmap:
                tmap[pair[1]] = set()
            tmap[pair[1]].add(pair[0])

    with open('data/AG_dev','r') as f:
        lines = f.readlines()
        for line in lines:
            pair = line.strip().split('\t')
            tup = (pair[0], pair[1], 1)
            dev.append(tup)
            if pair[0] not in smap:
                smap[pair[0]] = set()
            smap[pair[0]].add(pair[1])
            if pair[1] not in tmap:
                tmap[pair[1]] = set()
            tmap[pair[1]].add(pair[0])
    with open('data/AG_test','r') as f:
        lines = f.readlines()
        for line in lines:
            pair = line.strip().split('\t')
            tup = (pair[0], pair[1], 1)
            test.append(tup)
            if pair[0] not in smap:
                smap[pair[0]] = set()
            smap[pair[0]].add(pair[1])
            if pair[1] not in tmap:
                tmap[pair[1]] = set()
            tmap[pair[1]].add(pair[0])
    return train, dev, test, smap, tmap

def train_batch(args, info, train, smap, tmap, neg_sample = False, hard_example = False):
    if neg_sample == True:
        temp = []
        temp.extend(train)
        if hard_example == False:
            neg = gen_rand_all_neg(len(train) * args.negative_num, smap, tmap, info['tableA_len'], info['tableB_len'])
        temp.extend(neg)
    else:
        temp = train
    S, T, L = toTensor(temp)
    data = Data.TensorDataset(S, T, L)
    loader = Data.DataLoader(
        dataset=data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
    )
    return loader

def pre_seprate_batch(args, info):
    tableA_len = info['tableA_len']
    tableB_len = info['tableB_len']
    nsample = []
    for i in range(tableA_len):
        for j in range(tableA_len):
            if i == j:
                continue
            nsample.append((i, j, -1))

    for i in range(tableA_len, tableB_len):
        for j in range(tableA_len, tableB_len):
            if i == j:
                continue
            nsample.append((i, j, -1))
    L = torch.zeros(len(nsample))
    L = torch.add(L, -1)
    S, T, L = toTensor(nsample)
    data = Data.TensorDataset(S, T, L)
    loader = Data.DataLoader(
        dataset=data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
    )
    return loader

def compute_f1(tp, tn, fp, fn):
    p = 0
    r = 0
    f1 = 0
    if tp + fp != 0:
        p = tp / (tp + fp)
    if tp + fn != 0:
        r = tp / (tp + fn)
    if p + r != 0:
        f1 = 2 * p * r / (p + r)
    print("tp,tn,fp,fn:", tp, tn, fp, fn)
    return p, r, f1

def load_graph(dataset):
    names = ['adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            objects.append(pkl.load(f))
    adj = objects[0]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = preprocess_adj(adj)
    adj = torch.from_numpy(adj.todense()).float()
    return adj


def load_tdt(info):
    train = info['data']['train']
    dev = info['data']['dev']
    test = info['data']['test']
    smap = {}
    tmap = {}
    def build_stmap(mapping, smap, tmap):
        for (x, y, l) in mapping:
            if l != 1:
                continue
            if x not in smap:
                smap[x] = set()
            smap[x].add(y)
            if y not in tmap:
                tmap[y] = set()
            tmap[y].add(x)
    build_stmap(train, smap, tmap)
    build_stmap(dev, smap, tmap)
    build_stmap(test, smap, tmap)
    return train, dev, test, smap, tmap

def train_batch4lstm(args, train):
    S = []
    T = []
    L = []
    for (x,y,l) in train:
        S.append(torch.LongTensor(x).view(1,-1))
        T.append(torch.LongTensor(y).view(1,-1))
        L.append(int(l))
    S = torch.cat(S,dim=0)
    T = torch.cat(T, dim=0)
    L = torch.LongTensor(L)
    data = Data.TensorDataset(S, T, L)
    loader = Data.DataLoader(
        dataset=data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=1,
    )
    return loader

def train_batch4gcn_hw(args, train, isTest = False):
    S = []
    T = []
    L = []
    for (x,y,l) in train:
        S.append(x)
        T.append(y)
        if l == -1 or l == 0:
            L.append(0)
        else:
            L.append(1)
    S = torch.LongTensor(S)
    T = torch.LongTensor(T)
    L = torch.LongTensor(L)
    if isTest == True:
        return S, T, L
    data = Data.TensorDataset(S, T, L)
    loader = Data.DataLoader(
        dataset=data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=1,
    )
    return loader