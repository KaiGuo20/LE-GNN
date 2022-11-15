from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *
import uuid
from layers import *
from networkx.drawing.tests.test_pylab import plt
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=40, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--al', type=float, default=0.9, help='initial proportion')#0.1
parser.add_argument('--all', type=float, default=0.1, help='initial proportion')#0.1
parser.add_argument('--model', default='LEGNN', help='model')
parser.add_argument('--bias', default=False, help='use or not bias.')
parser.add_argument('--gama1', type=float, default=1e-3, help='refine')
parser.add_argument('--gama2', type=float, default=1e-2, help='refine')
parser.add_argument('--norm',  default=False, help='use or not norm.')
parser.add_argument('--lal_rate', type=float, default=0, help='label noise ratio')#0.1
parser.add_argument('--wd1', type=float, default=0, help='weight decay (L2 loss on convs parameters).')#0.01 1e-4
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on fcs parameters).')#5e-4
parser.add_argument("--normalization", default="AugNormAdj",
                    help="The normalization on the adj matrix.")
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Disables CUDA training.')


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)


def train_step(model,optimizer,features,labels,adj,idx_train,labels_for_lpa):

    model.train()
    optimizer.zero_grad()
    if args.model == 'LEGNN':
        output,  Pseudo_label = model(features, adj, labels_for_lpa, idx_train)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train].to(device))

        loss_label = F.nll_loss(Pseudo_label[idx_train], labels[idx_train].to(device))

        Pseudo_label = torch.clamp(Pseudo_label, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, int(labels.max()) + 1).float().to(device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(Pseudo_label * torch.log(label_one_hot), dim=1)).mean()
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))

        loss_train = 1*loss_gcn + args.gama1*loss_label + args.gama2*rce
        loss_train.backward(retain_graph=True)

        optimizer.step()
    return loss_train.item(), acc_train.item(),\
            output[idx_train].cpu().detach().numpy(), \
            labels[idx_train].cpu().detach().numpy()


def validate_step(model,features,labels,adj,idx_val,labels_for_lpa, idx_train):
    model.eval()
    with torch.no_grad():

        output, label= model(features, adj,labels_for_lpa, idx_train)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model,features,labels,adj,idx_test,labels_for_lpa,idx_train):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        if args.model == 'LEGNN':
            output,_= model(features, adj,labels_for_lpa,idx_train)
        else:
            output = model(features, adj,idx_train,labels_for_lpa)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item(),output[idx_test].cpu().numpy(),labels[idx_test].cpu().numpy()
    

def train(datastr,splitstr):
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr, args.lal_rate, args.seed)
    print('labels', idx_test)
    labels_for_lpa = one_hot_embedding(labels, labels.max().item() + 1).type(torch.FloatTensor).to(device)
    features = features.to(device)
    adj = adj.to(device)
    ###########################################################

    if args.model == "LEGNN":
        model = LEGNN(nfeat=features.shape[1],
                          nlayers=args.layer,
                          nhidden=args.hidden,
                          nclass=int(labels.max()) + 1,
                          dropout=args.dropout,
                          al=args.al,
                          all=args.all,
                          adj = adj,
                          norm = args.norm).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr,
        #                         weight_decay=args.weight_decay)
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': args.wd1},
            {'params': model.params2, 'weight_decay': args.wd2},
        ], lr=args.lr)

    else:
        raise NotImplementedError("model error.")

    bad_counter = 0
    best = 999999999
    best1 = 0
    for epoch in range(args.epochs):
        if args.model == 'LEGNN':
            loss_tra, acc_tra, logits_tra, label_tra = train_step(model, optimizer, features, labels, adj, idx_train,labels_for_lpa)
            loss_val, acc_val = validate_step(model, features, labels, adj, idx_val,labels_for_lpa,idx_train)
        if(epoch+1)%1 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))

        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            acc1 = test_step(model,features,labels,adj,idx_test,labels_for_lpa,idx_train)[1]
            bad_counter = 0
            print(acc1)
        else:
            bad_counter += 1

        # if acc_val > best1:
        #     best1 = acc_val
        #     torch.save(model.state_dict(), checkpt_file)
        #     # wandb.save('checkpt_file')
        #     acc1 = test_step(model,features,labels,adj,idx_test,labels_for_lpa,idx_train)[1]
        #     print('test', acc1*100)
        #     bad_counter = 0
        # else:
        #     bad_counter += 1

        if bad_counter == args.patience:
            break
    loss_test, acc,test_logits, test_label = test_step(model,features,labels,adj,idx_test,labels_for_lpa,idx_train)
    return acc*100, test_logits, test_label

t_total = time.time()
acc_list = []
for i in range(10):
    datastr = args.data
    print("-----data", args.data)
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    acc_test,logits,label = train(datastr,splitstr)
    acc_list.append(acc_test)
    print(i,": {:.2f}".format(acc_list[-1]))
    print(acc_list)

print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test acc.:{:.2f}".format(np.mean(acc_list)))
print(acc_list)
print(np.std(acc_list, ddof=1))

