import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from highway import HighwayMLP
import math

class GCN_1layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, featureless=True):
        super(GCN_1layer, self).__init__()
        self.weights1 = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        #self.weights1 = nn.Linear(input_dim, hidden_dim)
        #self.weights2 = Parameter(torch.zeros([hidden_dim,output_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.weights1, gain=1)
        #nn.init.xavier_uniform_(self.weights2, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, X):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.weights1)
        #layer1 = self.weights1(self.A)
        #layer2 = F.relu(torch.mm(layer1,self.weights2))
        return layer1

class GCN_2layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, featureless=True):
        super(GCN_2layer, self).__init__()
        self.weights1 = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        #self.weights2 = Parameter(torch.zeros([hidden_dim,output_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.weights1, gain=1)
        #nn.init.xavier_uniform_(self.weights2, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, X):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.weights1)
        layer2 = torch.mm(pre_matrix, layer1)
        #layer2 = F.relu(torch.mm(temp, self.weights2))
        return layer2


class Logstic_Regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj):
        super(Logstic_Regression, self).__init__()
        self.gcn = GCN_1layer(input_dim, hidden_dim, output_dim, adj)
        self.linear = nn.Linear(2, 1)
        nn.init.xavier_uniform_(self.linear.weight, gain=1)
        self.input_dim = input_dim

    def forward(self, doc_att, batch_x, batch_y, use_cuda):
        encoding = self.gcn.forward(torch.eye(self.input_dim))

        def tuple_representation(doc_att, encoding, batch, use_cuda):
            collection0 = []
            collection1 = []
            ## compositional ##
            for ins in batch:
                v = ins.item()
                atts = doc_att[str(v)]
                if atts[1] == -1:
                    atts[1] = atts[0]
                att0_idx = [atts[0]]
                att1_idx = [atts[1]]
                idx0 = torch.LongTensor(att0_idx)
                idx1 = torch.LongTensor(att1_idx)
                if use_cuda == True:
                    idx0 = idx0.cuda()
                    idx1 = idx1.cuda()
                collection0.append(torch.index_select(encoding, 0, idx0))
                collection1.append(torch.index_select(encoding, 0, idx1))
            return torch.cat(collection0, dim=0), torch.cat(collection1, dim=0)

        x0, x1 = tuple_representation(doc_att, encoding, batch_x, use_cuda)
        y0, y1 = tuple_representation(doc_att, encoding, batch_y, use_cuda)
        f0 = F.cosine_similarity(x0, y0).view(-1,1)
        f1 = F.cosine_similarity(x1, y1).view(-1,1)
        feature = torch.cat((f0, f1), dim =1)
        out = self.linear(feature)
        return F.sigmoid(out)


class BiLSTM(nn.Module):
    def __init__(self, vocab, input_dim, hidden_dim, use_cuda, batch_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim//2,batch_first= True,bidirectional=True)
        self.embedding = nn.Embedding(vocab, input_dim,padding_idx=0)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.linear = nn.Linear(input_dim, 2)
        self.highway_layers = nn.ModuleList([HighwayMLP(input_dim, activation_function=F.relu)
                                             for _ in range(2)])
        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, x, y):
        embeds_x = self.embedding(x)
        lstm_out, hc = self.lstm(embeds_x)
        hn = hc[0]
        ex = torch.cat((hn[0], hn[1]), dim=1)

        embeds_y = self.embedding(y)
        lstm_out, hc = self.lstm(embeds_y)
        hn = hc[0]
        ey = torch.cat((hn[0], hn[1]), dim=1)
        input = ex - ey
        #dense2 = self.linear2(F.relu(dense1))
        #predict = F.softmax(dense2,dim=1)
        new_input = input
        for current_layer in self.highway_layers:
            new_input = current_layer(new_input)
        predict = self.softmax(self.linear(new_input))
        return predict


class GCN_hw(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, featureless=True):
        super(GCN_hw, self).__init__()
        self.weights1 = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.weights1, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(hidden_dim, 2)
        self.highway_layers = nn.ModuleList([HighwayMLP(hidden_dim, activation_function=F.relu)
                                             for _ in range(2)])
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X, batch_x, batch_y, doc_att, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = F.relu(torch.mm(pre_matrix,self.weights1))

        ex = self.tuple_representation(doc_att, layer1, batch_x, use_cuda)
        ey = self.tuple_representation(doc_att, layer1, batch_y, use_cuda)
        predict = []
        for i in range(len(ex)):
            temp = ex[i] - ey[i]
            input_pre = torch.mul(temp, temp)
            input = torch.div(input_pre, math.sqrt(len(input_pre))).unsqueeze(dim=0)

            new_input = input
            for current_layer in self.highway_layers:
                new_input = current_layer(new_input)
            predict.append(self.softmax(self.linear(new_input)))
        return torch.cat(predict, dim=0)

    def tuple_representation(self, doc_att, encoding, batch, use_cuda):
        collection = []
        ## compositional ##
        for i in range(len(batch)):
            v = batch[i].item()
            atts = doc_att[str(v)]
            att_idx = [atts[0]]  ## test on the first attribute
            idx = torch.LongTensor(att_idx)
            if use_cuda:
                idx = idx.cuda()
            collection.append(torch.index_select(encoding, 0, idx))
        return torch.cat(collection, dim=0)

class GCN_alignment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, featureless=True):
        super(GCN_alignment, self).__init__()
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(hidden_dim*6, 2)
        self.att_len = 3
        self.highway_layers = self.init_highway(hidden_dim, self.att_len)
        self.softmax = nn.LogSoftmax(dim=1)
    def init_highway(self, hidden_dim, size):
        layers = nn.ModuleList()
        for i in range(size):
            layers.append(
                nn.ModuleList([HighwayMLP(hidden_dim, activation_function=F.relu)
                       for _ in range(1)]))
        return layers

    def soft_attention(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            batch_collection = []
            for i in range(len(batch)):
                doc_collection = []
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                for j in range(att_len):
                    att_idx = atts[j]
                    if att_idx == -1:
                        doc_collection.append(encoding[v].view(1,-1))
                        continue
                    #att_idx = atts[0]  ## test on the first attribute
                    words_idx = att_words[str(att_idx)]
                    idx = torch.LongTensor(words_idx)
                    if use_cuda:
                        idx = idx.cuda()
                    doc_collection.append(torch.index_select(encoding, 0, idx))
                batch_collection.append(doc_collection)
            return batch_collection, att_len

        Q, att_len = _get_words_encoding(batch_x)
        A, _ = _get_words_encoding(batch_y)

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(batch_x)
        EQ = []
        EA = []
        for i in range(batch_size):
            temp_Q = []
            temp_A = []
            for j in range(att_len):
                wq, wa = align_weights(Q[i][j], A[i][j])
                eq = weighted_encoding(A[i][j], wq)
                ea = weighted_encoding(Q[i][j], wa)
                temp_Q.append(eq)
                temp_A.append(ea)
            EQ.append(temp_Q)
            EA.append(temp_A)
        return Q, A, EQ, EA, att_len

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)
        #layer2 = torch.mm(pre_matrix, layer1)

        Q, A, EQ, EA, att_len = self.soft_attention(batch_x, batch_y, layer1, doc_att, att_words, use_cuda)
        batch_size = len(batch_x)
        predict = []

        for i in range(batch_size):
            q = Q[i]
            a = A[i]
            eq = EQ[i]
            ea = EA[i]
            collection = []
            for j in range(att_len):
                q_sub = q[j] - eq[j]
                tq = torch.mul(q_sub, q_sub)
                a_sub = a[j] - ea[j]
                ta = torch.mul(a_sub, a_sub)

                new_input = tq
                for current_layer in self.highway_layers[j]:
                    new_input = current_layer(new_input)
                qr = torch.sum(new_input, dim=0)
                qr = torch.div(qr, math.sqrt(len(Q[i]))).unsqueeze(dim=0)

                new_input = ta
                for current_layer in self.highway_layers[j]:
                    new_input = current_layer(new_input)
                ar = torch.sum(new_input, dim=0)
                ar = torch.div(ar, math.sqrt(len(A[i]))).unsqueeze(dim=0)

                collection.append(qr)
                collection.append(ar)
                #collection.append(torch.abs(qr-ar))
            re = torch.cat(collection, dim=1)
            score = self.linear(re)
            predict.append(self.softmax(score))

        return torch.cat(predict, dim=0)

class GCN_alignment_flat(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, featureless=True):
        super(GCN_alignment_flat, self).__init__()
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(hidden_dim*4, 2)
        self.att_len = 3
        self.highway_layers = nn.ModuleList([HighwayMLP(hidden_dim, activation_function=F.relu) for _ in range(2)])
        self.softmax = nn.LogSoftmax(dim=1)

    def soft_attention(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            batch_collection = []
            for i in range(len(batch)):
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                words_idx = []
                for j in range(att_len): ###only take the first two attibute
                    att_idx = atts[j]
                    if att_idx == -1:
                        continue
                    words_idx.extend(att_words[str(att_idx)])
                idx = torch.LongTensor(words_idx)
                if use_cuda:
                    idx = idx.cuda()
                batch_collection.append(torch.index_select(encoding, 0, idx))
            return batch_collection

        Q = _get_words_encoding(batch_x)
        A = _get_words_encoding(batch_y)

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(batch_x)
        EQ = []
        EA = []
        for i in range(batch_size):
            wq, wa = align_weights(Q[i], A[i])
            eq = weighted_encoding(A[i], wq)
            ea = weighted_encoding(Q[i], wa)
            EQ.append(eq)
            EA.append(ea)
        return Q, A, EQ, EA,

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)
        #layer2 = torch.mm(pre_matrix, layer1)

        Q, A, EQ, EA = self.soft_attention(batch_x, batch_y, layer1, doc_att, att_words, use_cuda)
        batch_size = len(batch_x)
        predict = []

        for i in range(batch_size):
            q_sub = Q[i] - EQ[i]

            tq = torch.mul(q_sub, q_sub)
            a_sub = A[i] - EA[i]
            ta = torch.mul(a_sub, a_sub)

            #mul_q = torch.mul(Q[i], EQ[i])
            #mul_a = torch.mul(A[i], EA[i])
            #qr = torch.sum(mul_q, dim=0)

            mul_q = torch.sum(torch.mul(Q[i], EQ[i]), dim=0)
            mul_a = torch.sum(torch.mul(A[i], EA[i]), dim=0)
            mul_q = torch.div(mul_q, math.sqrt(len(Q[i]))).unsqueeze(dim=0)
            mul_a = torch.div(mul_a, math.sqrt(len(Q[i]))).unsqueeze(dim=0)

            new_input = tq
            for current_layer in self.highway_layers:
                new_input = current_layer(new_input)
            qr = torch.sum(new_input, dim=0)
            qr = torch.div(qr, math.sqrt(len(Q[i]))).unsqueeze(dim=0)


            new_input = ta
            for current_layer in self.highway_layers:
                new_input = current_layer(new_input)
            ar = torch.sum(new_input, dim=0)
            ar = torch.div(ar, math.sqrt(len(A[i]))).unsqueeze(dim=0)


            re = torch.cat((qr, mul_q, ar, mul_a), dim=1)
            score = self.linear(re)
            predict.append(self.softmax(score))

        return torch.cat(predict, dim=0)

class GCN_alignment_cam(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, featureless=True):
        super(GCN_alignment_cam, self).__init__()
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(hidden_dim*4, 2)
        self.att_len = 3
        #self.highway_layers = nn.ModuleList([HighwayMLP(hidden_dim, activation_function=F.relu) for _ in range(2)])
        self.preprocessing = nn.ModuleList([HighwayMLP(hidden_dim, activation_function=F.relu) for _ in range(1)])
        self.transform = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.LogSoftmax(dim=1)


    def _align_weights(self, x, y):
        score = torch.mm(x, y.t())
        return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

    def _weighted_encoding(self, y, w):
        return torch.mm(w, y)

    def _bypass(self, input, layers):
        new_input = input
        for current_layer in layers:
            new_input = current_layer(new_input)
        return new_input

    def _get_words_encoding(slef, batch, encoding, doc_att, att_words, use_cuda):
        batch_collection = []
        for i in range(len(batch)):
            v = batch[i].item()
            atts = doc_att[str(v)]
            att_len = len(atts)
            words_idx = []
            for j in range(att_len):
                att_idx = atts[j]
                if att_idx == -1:
                    continue
                words_idx.extend(att_words[str(att_idx)])
            idx = torch.LongTensor(words_idx)
            if use_cuda:
                idx = idx.cuda()
            batch_collection.append(torch.index_select(encoding, 0, idx))
        return batch_collection


    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)
        #layer2 = torch.mm(pre_matrix, layer1)


        Q = self._get_words_encoding(batch_x, layer1, doc_att, att_words, use_cuda)
        A = self._get_words_encoding(batch_y, layer1, doc_att, att_words, use_cuda)

        batch_size = len(batch_x)
        EQ = []
        EA = []
        PA = []
        PQ = []
        for i in range(batch_size):

            # preprocessing
            #Q_p = self._bypass(Q[i], self.preprocessing)
            #A_p = self._bypass(A[i], self.preprocessing)
            #PA.append(A_p)
            #PQ.append(Q_p)

            PQ = Q
            PA = A
            Q_p = Q[i]
            A_p = A[i]

            # alignment
            wq, wa = self._align_weights(Q_p, A_p)
            eq = self._weighted_encoding(A_p, wq)
            ea = self._weighted_encoding(Q_p, wa)
            EQ.append(eq)
            EA.append(ea)


        batch_size = len(batch_x)
        #features = []
        predict = []

        for i in range(batch_size):

            # Comparison
            # sub
            q_sub = PQ[i] - EQ[i]
            tq = torch.mul(q_sub, q_sub)
            qr = torch.sum(tq, dim=0)
            qr = torch.div(qr, math.sqrt(len(A[i]))).unsqueeze(dim=0)
            a_sub = PA[i] - EA[i]
            ta = torch.mul(a_sub, a_sub)
            ar = torch.sum(ta, dim=0)
            ar = torch.div(ar, math.sqrt(len(A[i]))).unsqueeze(dim=0)
            # mul
            mul_q = torch.sum(torch.mul(PQ[i], EQ[i]), dim=0)
            mul_a = torch.sum(torch.mul(PA[i], EA[i]), dim=0)
            mul_q = torch.div(mul_q, math.sqrt(len(Q[i]))).unsqueeze(dim=0)
            mul_a = torch.div(mul_a, math.sqrt(len(Q[i]))).unsqueeze(dim=0)


            re = torch.cat((qr, ar, mul_q, mul_a), dim=1)
            score = self.linear(re)
            predict.append(self.softmax(score))

        return torch.cat(predict, dim=0)


class GCN_alignment_cnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_num, adj, pre_trained_embedding, featureless=True):
        super(GCN_alignment_cnn, self).__init__()
        #self.embedding = pre_trained_embedding
        #self.embedding.requires_grad = True
        #self.embedding = Parameter(self.embedding)
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.kernel_num = kernel_num
        self.filter_sizes = [1, 2, 3, 4]
        #self.linear = nn.Linear(hidden_dim*4, 300)
        self.dense = nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 2)
        '''
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        '''
        self.softmax = nn.LogSoftmax(dim=1)

        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.kernel_num,
                                       kernel_size=(filter_size, hidden_dim*2+10),
                                       padding=5))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def soft_attention(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            batch_collection = []
            for i in range(len(batch)):
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                words_idx = []
                for j in range(att_len):
                    att_idx = atts[j]
                    if att_idx == -1:
                        continue
                    words_idx.extend(att_words[str(att_idx)])
                idx = torch.LongTensor(words_idx)
                if use_cuda:
                    idx = idx.cuda()
                batch_collection.append(torch.index_select(encoding, 0, idx))
            return batch_collection

        Q = _get_words_encoding(batch_x)
        A = _get_words_encoding(batch_y)

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(batch_x)
        EQ = []
        EA = []
        for i in range(batch_size):
            wq, wa = align_weights(Q[i], A[i])
            eq = weighted_encoding(A[i], wq)
            ea = weighted_encoding(Q[i], wa)
            EQ.append(eq)
            EA.append(ea)
        return Q, A, EQ, EA,

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)
        #layer2 = torch.mm(pre_matrix, layer1)
        #layer3 = torch.mm(pre_matrix, layer2)
        #layer4 = torch.mm(pre_matrix, layer3)

        Q, A, EQ, EA = self.soft_attention(batch_x, batch_y, layer1, doc_att, att_words, use_cuda)
        batch_size = len(batch_x)
        predict = []

        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        for i in range(batch_size):
            q_sub = Q[i] - EQ[i]

            tq = torch.mul(q_sub, q_sub)

            a_sub = A[i] - EA[i]
            ta = torch.mul(a_sub, a_sub)


            mul_q = torch.mul(Q[i], EQ[i])
            mul_a = torch.mul(A[i], EA[i])

            qr = tq

            #t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
            t = torch.cat((qr, mul_q), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            q_re = F.relu(encoding)

            ar = ta

            t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            a_re = F.relu(encoding)


            re = torch.cat((q_re, a_re), dim=1)
            score = self.dense(re)
            F.relu(score)
            predict.append(self.softmax(score))

            #print(self.embedding)

        return torch.cat(predict, dim=0)


class GCN_alignment_iia(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_num, adj, featureless=True):
        super(GCN_alignment_iia, self).__init__()
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.kernel_num = kernel_num
        self.filter_sizes = [1, 2]
        #self.linear = nn.Linear(hidden_dim*4, 300)
        self.dense = nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 2)
        '''
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        '''
        self.softmax = nn.LogSoftmax(dim=1)

        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.kernel_num,
                                       kernel_size=(filter_size, hidden_dim*2+2),
                                       padding=1
                                       ))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def soft_attention(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            record_collection = []
            att_collection = []
            token_collection = []
            for i in range(len(batch)):
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                words_idx = []
                ### record ###
                idx = torch.LongTensor([v])
                if use_cuda:
                    idx = idx.cuda()
                record_collection.append(torch.index_select(encoding, 0, idx))
                atts_idx = []
                #words_idx.append(v)
                for j in range(att_len):
                    att_idx = atts[j]
                    if att_idx == -1:
                        continue
                    atts_idx.append(att_idx)
                    ### token
                    words_idx.append(att_idx)
                    words_idx.extend(att_words[str(att_idx)])

                ### att ###
                idx = torch.LongTensor(atts_idx)
                if use_cuda:
                    idx = idx.cuda()
                att_collection.append(torch.index_select(encoding, 0, idx))
                #token
                idx = torch.LongTensor(words_idx)
                if use_cuda:
                    idx = idx.cuda()
                token_collection.append(torch.index_select(encoding, 0, idx))
            return token_collection, att_collection, record_collection

        Q, aQ, rQ = _get_words_encoding(batch_x)
        A, aA, rA = _get_words_encoding(batch_y)

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(batch_x)
        EQ = []
        EaQ = []
        ErQ = []
        EA = []
        EaA = []
        ErA= []

        for i in range(batch_size):
            wq, wa = align_weights(Q[i], A[i])
            eq = weighted_encoding(A[i], wq)
            ea = weighted_encoding(Q[i], wa)
            EQ.append(eq)
            EA.append(ea)
            waq, waa = align_weights(aQ[i], aA[i])
            eaq = weighted_encoding(aA[i], waq)
            eaa = weighted_encoding(aQ[i], waa)
            EaQ.append(eaq)
            EaA.append(eaa)
            ErA.append(rQ[i])
            ErQ.append(rA[i])
        return Q, A, EQ, EA, aQ, aA, EaQ, EaA, rQ, rA, ErQ, ErA

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)
        #layer2 = torch.mm(pre_matrix, layer1)

        Q, A, EQ, EA, aQ, aA, EaQ, EaA, rQ, rA, ErQ, ErA = self.soft_attention(batch_x, batch_y, layer1, doc_att, att_words, use_cuda)
        batch_size = len(batch_x)
        predict = []

        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        for i in range(batch_size):

            ## TOKEN layer
            q_sub = Q[i] - EQ[i]

            tq = torch.mul(q_sub, q_sub)

            a_sub = A[i] - EA[i]
            ta = torch.mul(a_sub, a_sub)


            mul_q = torch.mul(Q[i], EQ[i])
            mul_a = torch.mul(A[i], EA[i])

            qr = tq

            #t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
            t = torch.cat((qr, mul_q), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            q_re = F.relu(encoding)

            ar = ta

            t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            a_re = F.relu(encoding)
            token_re = torch.cat((q_re, a_re), dim=1)


            ## attribute layer
            q_sub = aQ[i] - EaQ[i]
            tq = torch.mul(q_sub, q_sub)

            a_sub = aA[i] - EaA[i]
            ta = torch.mul(a_sub, a_sub)

            mul_q = torch.mul(aQ[i], EaQ[i])
            mul_a = torch.mul(aA[i], EaA[i])

            qr = tq

            # t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
            t = torch.cat((qr, mul_q), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            # encoding = torch.cat(enc_outs, 1)
            q_re = F.relu(encoding)

            ar = ta

            t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            # encoding = torch.cat(enc_outs, 1)
            a_re = F.relu(encoding)
            att_re = torch.cat((q_re, a_re), dim=1)

            ### record layer
            q_sub = rQ[i] - ErQ[i]
            tq = torch.mul(q_sub, q_sub)

            a_sub = rA[i] - ErA[i]
            ta = torch.mul(a_sub, a_sub)

            mul_q = torch.mul(rQ[i], ErQ[i])
            mul_a = torch.mul(rA[i], ErA[i])

            qr = tq

            # t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
            t = torch.cat((qr, mul_q), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            # encoding = torch.cat(enc_outs, 1)
            q_re = F.relu(encoding)

            ar = ta

            t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            # encoding = torch.cat(enc_outs, 1)
            a_re = F.relu(encoding)
            record_re = torch.cat((q_re, a_re), dim=1)

            re = record_re
            score = self.dense(re)
            F.relu(score)
            predict.append(self.softmax(score))

        return torch.cat(predict, dim=0)

class GCN_alignment_gate_att(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_num, adj, featureless=True):
        super(GCN_alignment_gate_att, self).__init__()
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.kernel_num = kernel_num
        self.filter_sizes = [1, 2]
        #self.linear = nn.Linear(hidden_dim*4, 300)
        self.dense = nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 2)
        '''
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        '''
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )
        self.gate[0].bias.data.fill_(3)
        #self.att_len = 3
        #self.highway_layers = nn.ModuleList([HighwayMLP(hidden_dim, activation_function=F.relu) for _ in range(2)])
        self.softmax = nn.LogSoftmax(dim=1)


        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.kernel_num,
                                       kernel_size=(filter_size, hidden_dim*2)))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def soft_attention(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            batch_collection = []
            for i in range(len(batch)):
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                words_idx = []
                for j in range(att_len):
                    att_idx = atts[j]
                    if att_idx == -1:
                        continue
                    words_idx.extend(att_words[str(att_idx)])
                idx = torch.LongTensor(words_idx)
                if use_cuda:
                    idx = idx.cuda()
                batch_collection.append(torch.index_select(encoding, 0, idx))
            return batch_collection

        Q = _get_words_encoding(batch_x)
        A = _get_words_encoding(batch_y)

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(batch_x)
        EQ = []
        EA = []
        for i in range(batch_size):
            wq, wa = align_weights(Q[i], A[i])
            eq = weighted_encoding(A[i], wq)
            ea = weighted_encoding(Q[i], wa)
            EQ.append(eq)
            EA.append(ea)
        return Q, A, EQ, EA,

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)
        #layer2 = torch.mm(pre_matrix, layer1)

        Q, A, EQ, EA = self.soft_attention(batch_x, batch_y, layer1, doc_att, att_words, use_cuda)
        batch_size = len(batch_x)
        predict = []

        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        for i in range(batch_size):
            intra = torch.diagflat(self.gate(Q[i].detach()))
            Q_s = torch.mm(1 - intra, Q[i]) + torch.mm(intra, EQ[i])
            q_sub = Q[i] - Q_s

            tq = torch.mul(q_sub, q_sub)

            intra = torch.diagflat(self.gate(A[i].detach()))
            A_s = torch.mm(1 - intra, A[i]) + torch.mm(intra, EA[i])
            a_sub = A[i] - A_s
            ta = torch.mul(a_sub, a_sub)


            mul_q = torch.mul(Q[i], Q_s)
            mul_a = torch.mul(A[i], A_s)

            qr = tq

            #t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
            t = torch.cat((qr, mul_q), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            q_re = F.relu(encoding)

            ar = ta

            t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            a_re = F.relu(encoding)


            re = torch.cat((q_re, a_re), dim=1)
            score = self.dense(re)
            F.relu(score)
            predict.append(self.softmax(score))

        return torch.cat(predict, dim=0)

class GCN_alignment_gram(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_num, adj, featureless=True):
        super(GCN_alignment_gram, self).__init__()
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.kernel_num = kernel_num
        self.filter_sizes = [1, 2]
        #self.linear = nn.Linear(hidden_dim*4, 300)
        self.dense = nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 2)
        #self.att_len = 3
        #self.highway_layers = nn.ModuleList([HighwayMLP(hidden_dim, activation_function=F.relu) for _ in range(2)])
        self.softmax = nn.LogSoftmax(dim=1)


        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.kernel_num,
                                       kernel_size=(filter_size, hidden_dim*2)))
            self.encoders.append(self.__getattr__(enc_attr_name))

        self.intra_att_encoders = []
        for i, filter_size in enumerate([1]):
            enc_attr_name = "intra_att_encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv1d(in_channels=hidden_dim,
                                       out_channels=hidden_dim,
                                       kernel_size=filter_size,
                                       padding=filter_size-1))
            self.intra_att_encoders.append(self.__getattr__(enc_attr_name))

    def soft_attention(self, Q, A):

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(Q)
        EQ = []
        EA = []
        for i in range(batch_size):
            wq, wa = align_weights(Q[i], A[i])
            eq = weighted_encoding(A[i], wq)
            ea = weighted_encoding(Q[i], wa)
            EQ.append(eq)
            EA.append(ea)
        return EQ, EA

    def get_encoding(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            batch_collection = []
            for i in range(len(batch)):
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                words_idx = []
                for j in range(att_len):
                    att_idx = atts[j]
                    if att_idx == -1:
                        continue
                    words_idx.extend(att_words[str(att_idx)])
                idx = torch.LongTensor(words_idx)
                if use_cuda:
                    idx = idx.cuda()
                batch_collection.append(torch.index_select(encoding, 0, idx))
            return batch_collection

        Q = _get_words_encoding(batch_x)
        A = _get_words_encoding(batch_y)
        return Q, A

    def multi_gram(self, E):
        #batch_size = len(E)
        multi_gram_encoding = []
        for r in E:
            sum_outs = []
            encoding = []
            filter_s = 0
            r = r.t().unsqueeze(0)
            for encoder in self.intra_att_encoders:
                filter_s += 1
                f_map = encoder(r)
                enc_ = F.relu(f_map)
                enc_ = F.avg_pool2d(enc_, stride=1, kernel_size=(1, filter_s))
                enc_ = enc_.squeeze(0).t()
                encoding.append(enc_.unsqueeze(1))
                sum_outs.append(torch.sum(enc_, dim=1).unsqueeze(0))
            sum_outs = torch.cat(sum_outs, 0)
            encoding = torch.cat(encoding, 1)
            att = F.softmax(sum_outs, dim=0).t().unsqueeze(1)
            final = torch.matmul(att, encoding).squeeze(1)
            multi_gram_encoding.append(final)
        return multi_gram_encoding

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)

        ## n-gram ###
        OQ, OA = self.get_encoding(batch_x, batch_y, layer1, doc_att, att_words, use_cuda)
        Q = self.multi_gram(OQ)
        A = self.multi_gram(OA)

        EQ, EA = self.soft_attention(Q, A)
        batch_size = len(batch_x)
        predict = []

        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        for i in range(batch_size):
            q_sub = Q[i] - EQ[i]

            tq = torch.mul(q_sub, q_sub)
            tq = torch.mul(q_sub, q_sub)
            a_sub = A[i] - EA[i]
            ta = torch.mul(a_sub, a_sub)


            mul_q = torch.mul(Q[i], EQ[i])
            mul_a = torch.mul(A[i], EA[i])

            qr = tq

            t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            q_re = F.relu(encoding)

            ar = ta

            t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            a_re = F.relu(encoding)


            re = torch.cat((q_re, a_re), dim=1)
            score = self.dense(re)
            F.relu(score)
            predict.append(self.softmax(score))

        return torch.cat(predict, dim=0)



class GCN_alignment_joint_att(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_num, adj, featureless=True):
        super(GCN_alignment_joint_att, self).__init__()
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.kernel_num = kernel_num
        self.filter_sizes = [1, 2]
        self.r = 8
        self.linear = nn.Linear(hidden_dim*16, 2)
        self.dense = nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 2)
        '''
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        '''
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )
        self.att_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 350, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(350, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.att_gate_norm = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 350, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(350, 8, bias=False),
            torch.nn.Softmax(dim=0)
        )
        self.gate[0].bias.data.fill_(3)
        #self.att_len = 3
        #self.highway_layers = nn.ModuleList([HighwayMLP(hidden_dim, activation_function=F.relu) for _ in range(2)])
        self.softmax = nn.LogSoftmax(dim=1)


        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.kernel_num,
                                       kernel_size=(filter_size, hidden_dim)))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def soft_attention(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            batch_collection = []
            for i in range(len(batch)):
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                words_idx = []
                ### record ###
                words_idx.append(v)
                for j in range(att_len):
                    att_idx = atts[j]
                    if att_idx == -1:
                        continue
                    words_idx.append(att_idx)
                    words_idx.extend(att_words[str(att_idx)])
                idx = torch.LongTensor(words_idx)
                if use_cuda:
                    idx = idx.cuda()
                batch_collection.append(torch.index_select(encoding, 0, idx))
            return batch_collection

        Q = _get_words_encoding(batch_x)
        A = _get_words_encoding(batch_y)

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(batch_x)
        EQ = []
        EA = []
        for i in range(batch_size):
            wq, wa = align_weights(Q[i], A[i])
            eq = weighted_encoding(A[i], wq)
            ea = weighted_encoding(Q[i], wa)
            EQ.append(eq)
            EA.append(ea)
        return Q, A, EQ, EA,

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)
        #layer2 = torch.mm(pre_matrix, layer1)

        Q, A, EQ, EA = self.soft_attention(batch_x, batch_y, layer1, doc_att, att_words, use_cuda)
        batch_size = len(batch_x)
        predict = []

        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        for i in range(batch_size):


            cnn_flag = False

            if cnn_flag == True:
                # intra = torch.diagflat(self.gate(torch.mul(Q[i], Q[i])))
                intra = torch.diagflat(
                    self.att_gate(Q[i].detach())
                )
                # print(intra)
                Q_s = torch.mm(1 - intra, Q[i]) + torch.mm(intra, EQ[i])
                q_sub = Q[i] - EQ[i]

                tq = torch.mm(intra, torch.mul(q_sub, q_sub))

                # intra = torch.diagflat(self.gate(A[i].detach()))
                intra = torch.diagflat(
                    self.att_gate(A[i].detach())
                )
                A_s = torch.mm(1 - intra, A[i]) + torch.mm(intra, EA[i])
                a_sub = A[i] - EA[i]
                ta = torch.mm(intra, torch.mul(a_sub, a_sub))

                mul_q = torch.mul(Q[i], EQ[i])
                mul_a = torch.mul(A[i], EA[i])

                qr = tq
                # t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
                t = qr.unsqueeze(dim=0)
                t = t.unsqueeze(c_idx)
                enc_outs = []
                for encoder in self.encoders:
                    f_map = encoder(t)
                    enc_ = F.relu(f_map)
                    k_h = enc_.size()[h_idx]
                    enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                    enc_ = enc_.squeeze(w_idx)
                    enc_ = enc_.squeeze(h_idx)
                    enc_outs.append(enc_)
                encoding = self.dropout(torch.cat(enc_outs, 1))
                # encoding = torch.cat(enc_outs, 1)
                q_re = F.relu(encoding)

                ar = ta

                # t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
                t = ar.unsqueeze(dim=0)
                t = t.unsqueeze(c_idx)
                enc_outs = []
                for encoder in self.encoders:
                    f_map = encoder(t)
                    enc_ = F.relu(f_map)
                    k_h = enc_.size()[h_idx]
                    enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                    enc_ = enc_.squeeze(w_idx)
                    enc_ = enc_.squeeze(h_idx)
                    enc_outs.append(enc_)
                encoding = self.dropout(torch.cat(enc_outs, 1))
                # encoding = torch.cat(enc_outs, 1)
                a_re = F.relu(encoding)
                re = torch.cat((q_re, a_re), dim=1)
                score = self.dense(re)

            else:
                intra = self.att_gate_norm(Q[i].detach())
                #print(intra)
                q_sub = Q[i] - EQ[i]
                tq = torch.mm(intra.t(), torch.mul(q_sub, q_sub))
                intra = self.att_gate_norm(A[i].detach())
                a_sub = A[i] - EA[i]
                ta = torch.mm(intra.t(), torch.mul(a_sub, a_sub))

                q_re = tq
                a_re = ta
                re = torch.cat((q_re, a_re), dim=1)
                score = self.linear(re.view(1,-1))

            F.relu(score)
            predict.append(self.softmax(score))

        return torch.cat(predict, dim=0)


class alignment_cnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_num, adj, featureless=True):
        super(alignment_cnn, self).__init__()
        self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.kernel_num = kernel_num
        self.filter_sizes = [1, 2]
        #self.linear = nn.Linear(hidden_dim*4, 300)
        self.dense = nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 2)

        self.softmax = nn.LogSoftmax(dim=1)

        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.kernel_num,
                                       kernel_size=(filter_size, hidden_dim*2)))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def soft_attention(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            batch_collection = []
            for i in range(len(batch)):
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                words_idx = []
                for j in range(att_len):
                    att_idx = atts[j]
                    if att_idx == -1:
                        continue
                    words_idx.extend(att_words[str(att_idx)])
                idx = torch.LongTensor(words_idx)
                if use_cuda:
                    idx = idx.cuda()
                batch_collection.append(torch.index_select(encoding, 0, idx))
            return batch_collection

        Q = _get_words_encoding(batch_x)
        A = _get_words_encoding(batch_y)

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(batch_x)
        EQ = []
        EA = []
        for i in range(batch_size):
            wq, wa = align_weights(Q[i], A[i])
            eq = weighted_encoding(A[i], wq)
            ea = weighted_encoding(Q[i], wa)
            EQ.append(eq)
            EA.append(ea)
        return Q, A, EQ, EA,

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):

        Q, A, EQ, EA = self.soft_attention(batch_x, batch_y, self.embedding, doc_att, att_words, use_cuda)
        batch_size = len(batch_x)
        predict = []

        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        for i in range(batch_size):
            q_sub = Q[i] - EQ[i]

            tq = torch.mul(q_sub, q_sub)

            a_sub = A[i] - EA[i]
            ta = torch.mul(a_sub, a_sub)


            mul_q = torch.mul(Q[i], EQ[i])
            mul_a = torch.mul(A[i], EA[i])

            qr = tq

            #t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
            t = torch.cat((qr, mul_q), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            q_re = F.relu(encoding)

            ar = ta

            t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            a_re = F.relu(encoding)


            re = torch.cat((q_re, a_re), dim=1)
            score = self.dense(re)
            F.relu(score)
            predict.append(self.softmax(score))

        return torch.cat(predict, dim=0)


class GCN_alignment_cnn_nce(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_num, adj, pre_trained_embedding, featureless=True):
        super(GCN_alignment_cnn_nce, self).__init__()
        self.embedding = pre_trained_embedding
        self.embedding.requires_grad = True
        self.embedding = Parameter(self.embedding)
        #self.embedding = Parameter(torch.zeros([input_dim, hidden_dim], dtype=torch.float, requires_grad=True))
        #nn.init.xavier_uniform_(self.embedding, gain=1)
        self.A = adj
        self.featureless = featureless
        self.dropout = 0.5
        self.dropout = nn.Dropout(self.dropout)
        self.kernel_num = kernel_num
        self.filter_sizes = [1, 2]
        #self.linear = nn.Linear(hidden_dim*4, 300)
        self.dense = nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 2)
        '''
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.kernel_num*len(self.filter_sizes)*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        '''
        self.softmax = nn.LogSoftmax(dim=1)

        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.kernel_num,
                                       kernel_size=(filter_size, hidden_dim)))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def soft_attention(self, batch_x, batch_y, encoding, doc_att, att_words, use_cuda):
        def _get_words_encoding(batch):
            batch_collection = []
            for i in range(len(batch)):
                v = batch[i].item()
                atts = doc_att[str(v)]
                att_len = len(atts)
                words_idx = []
                for j in range(att_len):
                    att_idx = atts[j]
                    if att_idx == -1:
                        continue
                    words_idx.extend(att_words[str(att_idx)])
                idx = torch.LongTensor(words_idx)
                if use_cuda:
                    idx = idx.cuda()
                batch_collection.append(torch.index_select(encoding, 0, idx))
            return batch_collection

        Q = _get_words_encoding(batch_x)
        A = _get_words_encoding(batch_y)

        def align_weights(x, y):
            score = torch.mm(x, y.t())
            return F.softmax(score, dim= 1), F.softmax(score.t(), dim= 1)

        def weighted_encoding(y, w):
            return torch.mm(w, y)

        batch_size = len(batch_x)
        EQ = []
        EA = []
        for i in range(batch_size):
            wq, wa = align_weights(Q[i], A[i])
            eq = weighted_encoding(A[i], wq)
            ea = weighted_encoding(Q[i], wa)
            EQ.append(eq)
            EA.append(ea)
        return Q, A, EQ, EA,

    def forward(self, X, batch_x, batch_y, doc_att, att_words, use_cuda):
        if self.featureless != True:
            X = self.dropout(X)
            pre_matrix = torch.mm(self.A, X)
        else:
            pre_matrix = self.A
        layer1 = torch.mm(pre_matrix,self.embedding)
        #layer2 = torch.mm(pre_matrix, layer1)

        Q, A, EQ, EA = self.soft_attention(batch_x, batch_y, layer1, doc_att, att_words, use_cuda)
        batch_size = len(batch_x)
        predict = []

        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        for i in range(batch_size):
            q_sub = Q[i] - EQ[i]

            tq = torch.mul(q_sub, q_sub)

            a_sub = A[i] - EA[i]
            ta = torch.mul(a_sub, a_sub)


            mul_q = torch.mul(Q[i], EQ[i])
            mul_a = torch.mul(A[i], EA[i])

            qr = tq

            #t = torch.cat((qr, mul_q),dim=1).unsqueeze(dim=0)
            t= Q[i].unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            q_re = F.relu(encoding)

            ar = ta

            #t = torch.cat((ar, mul_a), dim=1).unsqueeze(dim=0)
            t = A[i].unsqueeze(dim=0)
            t = t.unsqueeze(c_idx)
            enc_outs = []
            for encoder in self.encoders:
                f_map = encoder(t)
                enc_ = F.relu(f_map)
                k_h = enc_.size()[h_idx]
                enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
                enc_ = enc_.squeeze(w_idx)
                enc_ = enc_.squeeze(h_idx)
                enc_outs.append(enc_)
            encoding = self.dropout(torch.cat(enc_outs, 1))
            #encoding = torch.cat(enc_outs, 1)
            a_re = F.relu(encoding)


            re = torch.cat((q_re, a_re), dim=1)
            score = self.dense(re)
            F.relu(score)
            predict.append(self.softmax(score))

            #print(self.embedding)

        return torch.cat(predict, dim=0)