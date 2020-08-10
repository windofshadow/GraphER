import torch
from model import *
import argparse
from utils import *
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(description='CNN text classification')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs for train')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout (0 = no dropout) [default: 0.5]')
    parser.add_argument('--hidden-dim', type=int, default=200,
                        help='number of embedding dimension [default: 100]')
    parser.add_argument('--output-dim', type=int, default=50,
                        help='number of embedding dimension [default: 50]')
    parser.add_argument('--negative-num', type=int, default=10,
                        help='number of negative example')
    parser.add_argument('--kernel-num', type=int, default=384,
                        help='number of negative example')
    parser.add_argument('--loss', type=str, default='Cosine',
                        help='loss function')
    parser.add_argument('--model', type=str, default='gcn-align-cnn',
                        help='model')
    parser.add_argument('--layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--l2', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--infer_thres', type=float, default=0.5,
                        help='inference threshold [default: 0.5]')
    parser.add_argument('--batch', type=int, default=32,
                        help='batch size [default: 10]')
    parser.add_argument('--cuda-able', action='store_true',
                        help='enables cuda')
    args = parser.parse_args()
    return args


def validation(info, encoding, dev, thres, use_cuda = False):
    values = [0, 0, 0, 0] # tp tn fp fn
    thres = torch.tensor(thres)
    doc_att = info['data']['odc_att']
    temp = dev
    for inst in temp:
        (x, y, l) = inst
        s = tuple_representation(doc_att, encoding,torch.LongTensor([x]), use_cuda)
        t = tuple_representation(doc_att, encoding,torch.LongTensor([y]), use_cuda)
        score = torch.nn.functional.cosine_similarity(s, t, dim=1)
        if score.data[0].cpu() > thres:
            if l == 1:
                values[0] += 1
            else:
                values[2] += 1
        elif l == 1:
            values[3] += 1
        else:
            values[1] += 1
    p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
    return p, r, f1

def validation4LR(info, lr, dev, use_cuda = False):
    values = [0, 0, 0, 0] # tp tn fp fn
    doc_att = info['data']['odc_att']
    S, T, L = toTensor(dev)
    L = label_batch_for_lr(L)
    predict = lr.forward(doc_att, S, T, use_cuda)
    for i in range(len(predict)):
        if predict[i] > 0.5:
            if L[i] == 1:
                values[0] += 1
            else:
                values[2] += 1
        elif L[i] == 1:
            values[3] += 1
        else:
            values[1] += 1
    p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
    return p, r, f1

def validation4lstm(lstm, dev, use_cuda = False):
    values = [0, 0, 0, 0] # tp tn fp fn
    for (x, y, l) in dev:
        x = torch.LongTensor(x).unsqueeze(dim=0)
        y = torch.LongTensor(y).unsqueeze(dim=0)
        #ex = lstm.forward(x)
        #ey = lstm.forward(y)
        if use_cuda == True:
            x = x.cuda()
            y = y.cuda()
        #score = F.cosine_similarity(ex, ey, dim =1)
        score = lstm.forward(x,y)
        _, indices = torch.max(score, dim=1)
        if indices == 1:
            if l == '1':
                values[0] += 1
            else:
                values[2] += 1
        elif l == '1':
            values[3] += 1
        else:
            values[1] += 1
    p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
    return p, r, f1

def validation4test(info, encoding, dev, thres, use_cuda = False):
    values = [0, 0, 0, 0]  # tp tn fp fn
    f = open('output.txt','w',encoding='utf-8')
    docs = info['data']['doc_content']
    doc_att = info['data']['odc_att']
    def list2str(lista):
        strr = ''
        for att in lista:
            for w in att:
                strr += w+' '
        return strr

    thres = torch.tensor(thres)
    temp = dev
    for inst in temp:
        (x, y, l) = inst
        s = tuple_representation(doc_att, encoding, torch.LongTensor([x]), use_cuda)
        t = tuple_representation(doc_att, encoding, torch.LongTensor([y]), use_cuda)
        score = torch.nn.functional.cosine_similarity(s, t, dim=1)
        if score.data[0].cpu() > thres:
            if l == 1:
                values[0] += 1
                f.write(list2str(docs[str(x)])+'\n'+list2str(docs[str(y)])+'\n')
                f.write('1\t'+'1\t'+str(score.item())+'\n\n')
            else:
                values[2] += 1
                f.write(list2str(docs[str(x)]) + '\n' + list2str(docs[str(y)]) + '\n')
                f.write('0\t' + '1\t'+str(score.item())+'\n\n')
        elif l == 1:
            values[3] += 1
            f.write(list2str(docs[str(x)]) + '\n' + list2str(docs[str(y)]) + '\n')
            f.write('1\t' + '0\t'+str(score.item())+'\n\n')
        else:
            values[1] += 1
    p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
    return p, r, f1

def align_validation4test(info, model, dev, use_cuda = False):

    f = open('output.txt','w',encoding='utf-8')
    docs = info['data']['doc_content']
    doc_att = info['data']['odc_att']
    att_words = info['data']['att_words']
    def list2str(lista):
        strr = ''
        for att in lista:
            for w in att:
                strr += w+' '
        return strr

    values = [0, 0, 0, 0]  # tp tn fp fn
    X, Y, L = train_batch4gcn_hw([], dev, True)
    predict = model.forward(torch.eye(10), X, Y, doc_att, att_words, False)
    _, indices = torch.max(predict, dim=1)
    for i in range(len(indices)):
        if indices[i] == 1:
            if L[i] == 1:
                values[0] += 1
                f.write(list2str(docs[str(X[i].item())]) + '\n' + list2str(docs[str(Y[i].item())]) + '\n')
                f.write('1\t' + '1\t' +  '\n\n')
            else:
                values[2] += 1
                f.write(list2str(docs[str(X[i].item())]) + '\n' + list2str(docs[str(Y[i].item())]) + '\n')
                f.write('0\t' + '1\t' + '\n\n')
        elif L[i] == 1:
            values[3] += 1
            #values[3] += 1
            f.write(list2str(docs[str(X[i].item())]) + '\n' + list2str(docs[str(Y[i].item())]) + '\n')
            f.write('1\t' + '0\t' + '\n\n')
        else:
            values[1] += 1
    p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
    return p, r, f1


def tuple_representation(doc_att, encoding, batch, use_cuda):
    flag = 'att'
    if flag == 'att':
        collection = []
        ## compositional ##
        for ins in batch:
            v = ins.item()
            atts = doc_att[str(v)]
            att_idx = [atts[0]]  ## test on the first attribute
            idx = torch.LongTensor(att_idx)
            if use_cuda == True:
                idx = idx.cuda()
            collection.append(torch.index_select(encoding, 0, idx))
        return torch.cat(collection, dim=0)
    else:
        if use_cuda == True:
            batch = batch.cuda()
        return torch.index_select(encoding, 0, batch)


def validation4gcn_hw(gcn_hw, dev, doc_att, use_cuda):
    values = [0, 0, 0, 0]  # tp tn fp fn
    X, Y, L = train_batch4gcn_hw(args, dev, True)
    predict = gcn_hw.forward(torch.eye(10), X, Y, doc_att, use_cuda)
    _, indices = torch.max(predict, dim=1)
    for i in range(len(indices)):
        if indices[i] == 1:
            if L[i]== 1:
                values[0] += 1
            else:
                values[2] += 1
        elif L[i] == 1:
            values[3] += 1
        else:
            values[1] += 1
    p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
    return p, r, f1

def validation4gcn_align(gcn_align, dev, doc_att, att_words, use_cuda):
    values = [0, 0, 0, 0]  # tp tn fp fn
    X, Y, L = train_batch4gcn_hw(args, dev, True)
    predict = gcn_align.forward(torch.eye(10), X, Y, doc_att, att_words, use_cuda)
    _, indices = torch.max(predict, dim=1)
    for i in range(len(indices)):
        if indices[i] == 1:
            if L[i]== 1:
                values[0] += 1
            else:
                values[2] += 1
        elif L[i] == 1:
            values[3] += 1
        else:
            values[1] += 1
    p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
    return p, r, f1



def gcn_align_cnn_model(args):
    info = torch.load('data/' + 'Amazon-Google' + '.info')
    doc_att = info['data']['odc_att']
    att_words = info['data']['att_words']
    embedding = info['data']['embedding']
    #embedding = []

    use_cuda = torch.cuda.is_available() and args.cuda_able

    print('-' * 90)
    A = load_graph("Amazon-Google")

    input_dim = len(A)

    if use_cuda == True:
        A = A.cuda()

    train, dev, test, smap, tmap = load_tdt(info)

    best = 0.0

    criterion = nn.NLLLoss()
    gcn_align = GCN_alignment_cnn(input_dim, args.hidden_dim, args.kernel_num, A, embedding)
    if use_cuda == True:
        gcn_align = gcn_align.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gcn_align.parameters()), lr=args.lr,
                                 weight_decay=args.l2)

    for epoch in range(args.epochs):
        loader = train_batch4gcn_hw(args, train)
        total_loss = torch.zeros(1)
        for step, (batch_x, batch_y, batch_l) in enumerate(loader):
            if use_cuda == True:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_l = batch_l.cuda()
            predict = gcn_align.forward(torch.eye(input_dim), batch_x, batch_y, doc_att, att_words, use_cuda)
            loss = criterion(predict, batch_l)
            total_loss += loss
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p, r, f1 = validation4gcn_align(gcn_align, test, doc_att, att_words, use_cuda)
        print(p,r,f1)
        if best < f1:
            best = f1
            #best_model = "model/model_" + str(epoch) + ".pkl"
            torch.save(gcn_align, "model/model_best.pkl")
    print("best", best)


if __name__=="__main__":
    try:
        args = get_args()

        if args.model == 'gcn-align-cnn':
            gcn_align_cnn_model(args)


    except KeyboardInterrupt:
        print("error")
