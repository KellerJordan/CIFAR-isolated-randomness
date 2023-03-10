import os
import uuid
import pickle
import hashlib
import argparse
import numpy as np
import torch
from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True, warn_only=True)
from torch import nn
from torch.optim import SGD, lr_scheduler
import torchvision

from model import get_model
from data import get_loaders

def run_training(model_seed, order_seed, aug_seed):
    ## training hyperparams and setup
    EPOCHS = 2
    train_loader, test_loader = get_loaders(order_seed, aug_seed, epochs=EPOCHS)

    n_iters = len(train_loader)
    lr_schedule = np.interp(np.arange(1+n_iters), [0, n_iters], [1, 0]) 

    model = get_model(model_seed).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    loss_fn = nn.CrossEntropyLoss()

    ## train
    model.train()
    losses = []
    it = tqdm(train_loader)
    for inputs, labels in it: 
        outputs = model(inputs.cuda())
        loss = loss_fn(outputs, labels.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_i = loss.item()
        losses.append(loss_i)
        it.set_postfix(loss='{:05.3f}'.format(loss_i))
        scheduler.step()
    
    ## evaluate
    model.eval()
    correct = 0 
    outputs_l = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.cuda())
            outputs_l.append(outputs.clone())
            pred = outputs.argmax(dim=1).cpu()
            correct += (labels == pred).float().sum().item()
    outputs = torch.cat(outputs_l).cpu().numpy()
    
    ## return stats
    stats = {'correct': int(correct),
             'outputs': outputs,
             'losses': losses}
    return stats

parser = argparse.ArgumentParser()
# model_seed controls random initialization of the model
parser.add_argument('--model_seed', type=int, default=0)
# order_seed controls the order of data
# such that the augmentations of the ith training image during the jth
# epoch are the same regardless of the order within the epoch,
# i.e. with a fixed aug_seed, a particular picture of a boat will always
# get augmented the same way at epoch $i$, regardless of when it appears.
parser.add_argument('--order_seed', type=int, default=0)
# aug_seed controls the data augmentations
parser.add_argument('--aug_seed', type=int, default=0)
if __name__ == '__main__':
    args = parser.parse_args()

    kwargs = {'model_seed': args.model_seed,
              'order_seed': args.order_seed,
              'aug_seed': args.aug_seed}
    stats = run_training(**kwargs)
    log = {**kwargs, **stats}
    print('correct=%d' % log['correct'])

    os.makedirs('./logs', exist_ok=True)
    log_path = os.path.join('./logs', str(uuid.uuid4())+'.pkl')
    #s = ','.join(str(a) for a in [args.model_seed, args.order_seed, args.aug_seed])
    #arg_hash = hashlib.sha256(s.encode('utf-8')).hexdigest()
    #log_path = os.path.join(log_dir, '%s.pkl' % arg_hash)
    with open(log_path, 'wb') as f:
        pickle.dump(log, f)

