from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
# from sympy import true
import torch
import torch.nn.functional as F
import torch.optim as optim
from networkx.drawing.tests.test_pylab import plt
from torch.distributions import Beta
from utils import *
from model import *
import uuid
import os.path as osp
from test import *
import wandb
import logging
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=41, help='Random seed.')#42 41（86.3）
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')#1500 100
parser.add_argument('--lr', type=float, default=0.02, help='learning rate.')#0.01
parser.add_argument('--wd1', type=float, default=5e-4, help='weight decay (L2 loss on convs parameters).')#0.01 1e-4
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on fcs parameters).')#5e-4
parser.add_argument('--layer', type=int, default=32, help='Number of layers.')#64
parser.add_argument('--hidden', type=int, default=256, help='hidden dimensions.')#32
parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate (1 - keep probability).')#0.6
parser.add_argument('--patience', type=int, default=100, help='Patience')#200 30 600
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--model', default='LEGNN', help='model')

parser.add_argument('--bias',  default=True, help='use or not bias.')
parser.add_argument('--norm',  default=False, help='use or not norm.')
parser.add_argument('--bad',  default=False, help='use or not norm.')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--al', type=float, default=0.2, help='initial proportion')#0.1
parser.add_argument('--all', type=float, default=0.1, help='initial proportion')#0.1
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')#False
parser.add_argument('--gama1', type=float, default=1e-3, help='refine')
parser.add_argument('--gama2', type=float, default=1e-2, help='refine')
parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay (L2 loss on parameters).')
parser.add_argument("--normalization", default="AugNormAdj",
                    help="The normalization on the adj matrix.")
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--lal_rate', type=float, default=0, help='label noise ratio')#0.1




args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)

# now it works


if  args.data=='cora' or args.data=='citeseer' or args.data=='pubmed':
    adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data, args.lal_rate, args.seed)
    labels_for_lpa = one_hot_embedding(labels, labels.max().item() + 1).type(torch.FloatTensor).to(device)

cudaid = "cuda:"+str(args.dev) if torch.cuda.is_available() else "cpu"
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)

checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

def train(epoch):

    model.train()
    optimizer.zero_grad()
    output,  Pseudo_label = model(features, adj, labels_for_lpa, idx_train)
    loss_gcn = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_label = F.nll_loss(Pseudo_label[idx_train], labels[idx_train].to(device))

    Pseudo_label = torch.clamp(Pseudo_label, min=1e-7, max=1.0)
    label_one_hot = torch.nn.functional.one_hot(labels, int(labels.max()) + 1).float().to(device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    rce = (-1*torch.sum(Pseudo_label * torch.log(label_one_hot), dim=1)).mean()
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = 1*loss_gcn  + args.gama1*loss_label +args.gama2*rce
    loss_train.backward(retain_graph=True)

    optimizer.step()
    return loss_train.item(),acc_train.item(), \
           output[idx_test].cpu().detach().numpy(), \
           labels[idx_test].cpu().detach().numpy()


def validate():
    model.eval()
    with torch.no_grad():
        output,_= model(features, adj,labels_for_lpa,idx_train)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output,_= model(features, adj,labels_for_lpa,idx_train)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return output[idx_test].cpu().numpy(), loss_test.item(),acc_test.item(), labels[idx_test].cpu().numpy()
ACC = []
for i in range(10):

    run_time =[]


    if args.model == "LEGNN":
        model = LEGNN(nfeat=features.shape[1],
                        nlayers=args.layer,
                        nhidden=args.hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=args.dropout,
                        al = args.al,
                        all = args.all,
                        adj = adj,
                        norm = args.norm).to(device)
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': args.wd1},
            {'params': model.params2, 'weight_decay': args.wd2},
        ], lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=20,eta_min=0.001)

    else:
        raise NotImplementedError("model error.")



    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best1 = 0
    best_epoch = 0
    acc = 0
    train_time = []

    for epoch in range(args.epochs):
        # scheduler.step(epoch)
        t = time.time()
        loss_tra,acc_tra, logits_tra, label_tra = train(epoch)
        epoch_time = time.time()-t
        train_time.append(epoch_time)
        loss_val,acc_val = validate()
        if(epoch+1)%1 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        if args.bad:
            if acc_val > best1:

                best1 = acc_val
                best_epoch = epoch
                acc = acc_val
                torch.save(model.state_dict(), checkpt_file)
                # wandb.save('checkpt_file')
                acc1 = test()[2]
                print('test', acc1 * 100)
                bad_counter = 0
            else:
                bad_counter += 1
        else:
            if loss_val < best:
                best = loss_val
                best_epoch = epoch
                acc = acc_val
                torch.save(model.state_dict(), checkpt_file)
                test_logits, loss_test, acc_test, test_label  = test()
                print('test', acc_test*100)
                bad_counter = 0
            else:
                bad_counter += 1





        if bad_counter == args.patience:
            break

    if args.test:
        acc = test()[2]
    ACC.append(acc*100)

    run_time.append(time.time() - t_total)
    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))
    print('average epoch train time', np.mean(train_time))
    print(ACC)
print(np.mean(ACC))
print(np.std(ACC, ddof=1))



