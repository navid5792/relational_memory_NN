# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from relational_rnn_general import RelationalMemory
#from relational_rnn_general_2 import RelationalMemory
from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models_rm_sick import NLINet
from read_data import get_MSRP_data, get_SICK_data, get_AI_data, get_SICK_tree_data, get_QQP_data, get_SICK_binary_data, get_SICK_labels

from metrics import Metrics

parser = argparse.ArgumentParser(description='NLI training')
print("SICK version 3 BS 16 drop 0.4 5 features")
# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle') # 3 max 4 mean
parser.add_argument("--word_emb_path", type=str, default="dataset/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.4, help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=1, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=1, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=512, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=5, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='mean', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)


class RMCArguments:
    def __init__(self):
        self.memslots = 1
        self.numheads = 8
        self.headsize = int((512 * self.memslots) / (self.numheads * self.memslots)) # 128 need to make 1024
        self.input_size = 300  # dimensions per timestep
        self.numblocks = 1
        self.forgetbias = 1.
        self.inputbias = 0.
        self.attmlplayers = 2
        self.clip = 0.1

metrics = Metrics(5)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
#train, valid, test = get_nli(params.nlipath)


train, valid, test = get_SICK_data()
n1 = len(train['s1']) % params.batch_size
n2 = len(test['s1']) % params.batch_size
n3 = len(valid['s1']) % params.batch_size

if n1 != 0:
    train['s1'] = train['s1'][0:-n1]
    train['s2'] = train['s2'][0:-n1]
    train['label'] = train['label'][0:-n1]
if n2 != 0:
    test['s1'] = test['s1'][0:-n2]
    test['s2'] = test['s2'][0:-n2]
    test['label'] = test['label'][0:-n2]
if n3 != 0:
    valid['s1'] = valid['s1'][0:-n3]
    valid['s2'] = valid['s2'][0:-n3]
    valid['label'] = valid['label'][0:-n3]

train_label, valid_label, test_label, valid_org, test_org = get_SICK_labels()

if n1 != 0:
    train_label = train_label[0:-n1]
if n2 != 0:
    test_label = test_label[0:-n2]
    test_org = test_org[0:-n2]
if n3 != 0:
    valid_label =  valid_label[0:-n3]
    valid_org =  valid_org[0:-n3]


#print( len(train['s1']),  len(test['s1']),  len(valid['s1']))


word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])


"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)




args1 = RMCArguments()
relationalMemory = RelationalMemory(mem_slots=args1.memslots, head_size=args1.headsize,
                                  input_size=args1.input_size,
                                  num_heads=args1.numheads, num_blocks=args1.numblocks,
                                  forget_bias=args1.forgetbias, input_bias=args1.inputbias)
nli_net = NLINet(config_nli_model, relationalMemory)
memory = nli_net.relational_memory.initial_state(params.batch_size, trainable=True).to(0)
memory2 = nli_net.relational_memory.initial_state(params.batch_size, trainable=True).to(0)

memory_ = nli_net.relational_memory.initial_state(params.batch_size, trainable=True).to(0)
memory2_ = nli_net.relational_memory.initial_state(params.batch_size, trainable=True).to(0)

print(memory.size())
print(nli_net)


#print(nli_net.encoder.enc_lstm.weight_ih_l0)
#print(nli_net.classifier[4].bias)


for name, x in nli_net.named_parameters():
    print(name)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
#loss_fn = nn.CrossEntropyLoss(weight=weight)
#loss_fn.size_average = False
loss_fn = nn.KLDivLoss()


# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
nli_net.cuda()
loss_fn.cuda()


"""
TRAIN
"""
val_acc_best = 1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None

f = open("norm_check.txt", "w")
f.close()

def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    #target = train['label'][permutation]
    target = train_label[permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))
    global_norm = 0.
    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())

        tgt_batch = Variable(torch.FloatTensor(target[stidx:stidx + params.batch_size])).squeeze(1).cuda()

        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len), memory, memory2, memory_, memory2_)

        #pred = output.data.max(1)[1]
        #correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        #assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        #print(output.size(), tgt_batch.size())

        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()
        #print(nli_net.classifier[4].bias)

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)
        global_norm += total_norm
        if total_norm > params.max_norm:
            #print("shrinking........")
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} '.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time))
                            ))
            
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []

    global_norm /= (len(s1)/params.batch_size)
    with open("norm_check.txt", "a") as f:
        f.write(str(epoch) + "\t" + str(float(global_norm)) + "\n")
    torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
    #train_acc = round(100 * correct.item()/len(s1), 2)
    #print('results : epoch {0} ; mean accuracy train : {1}'.format(epoch, train_acc))
    return None


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    #torch.no_grad()
    #nli_net.cpu()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    indices = torch.arange(1, 5 + 1, dtype=torch.float, device='cpu')
    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    #target = valid['label'] if eval_type == 'valid' else test['label']
    #predictions = torch.zeros(len(s1), dtype=torch.float, device='cpu')
    predictions = []
    target = torch.FloatTensor(valid_org) if eval_type == 'valid' else torch.FloatTensor(test_org)
    idx = 0
    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        #tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        #print(memory.size(), s1_batch.size(), s2_batch.size())
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len), memory, memory2, memory_, memory2_)

        #pred = output.data.max(1)[1]
        #correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        output = output.squeeze().to('cpu')
        for j in range(output.shape[0]):
            predictions = torch.dot(indices, torch.exp(output[j]))
            with open ("out.txt", "a") as f:
                f.write(str(predictions.item()) + "\t" + str(target[idx].item()) + "\n")
            idx += 1
        #print(idx)
    with open("out.txt", "r") as f:
        xx = f.readlines()
    pred = []
    org = []
    for x in xx:
        x = x.strip().split("\t")
        pred.append(float(x[0]))
        org.append(float(x[1]))
    pred = torch.FloatTensor(pred)
    org = torch.FloatTensor(org)

    pearson = metrics.pearson(pred, org)
    mse     = metrics.mse(pred, org)
    
    '''
    predictions = torch.FloatTensor(predictions)
    pearson = metrics.pearson(predictions, target).item()
    mse = metrics.mse(predictions, target).item()  
    '''

    # save model
    eval_acc = mse
    if final_eval:
        print('finalgrep : type {0} : mse {1} pearson: {2}'.format(eval_type, eval_acc, pearson))
    else:
        print('togrep : results : epoch {0} ; type {1} mse {2} pearsom {3}:\
              '.format(epoch, eval_type, mse, pearson))
    
    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc < val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return mse, pearson


"""
Train model on Natural Language Inference task
"""
epoch = 1
#nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))
with open("sick-r.txt", "w") as f:
    pass
while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    with open ("out.txt", "w") as f:
        pass
    mse, pearson = evaluate(epoch, 'test')
    with open("sick-r.txt", "a") as f:
        f.write(str(mse.item()) + "\t" + str(pearson.item()) + "\n")
    epoch += 1

# Run best model on test set.
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
#evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)

# Save encoder instead of full model
torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))