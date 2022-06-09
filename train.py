import os
import random
import pickle
import hashlib
import argparse
import numpy as np
import torch
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import SGD, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T

from model import create_model

BATCH_SIZE = 512 

def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class DeterministicCIFAR(torch.utils.data.Dataset):
    def __init__(self, order_seed, aug_seed, epochs, transform):
        self.dset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=None)
        # generate order of data during training using order_seed
        set_seed(order_seed)
        self.data_order = []
        for _ in range(epochs):
            epoch_order = list(range(50000))
            random.shuffle(epoch_order)
            self.data_order += epoch_order
        # generate seeds for each image augmentation using aug_seed
        num_iters = 50000*epochs
        set_seed(aug_seed)
        self.aug_seeds = np.random.randint(2**32, size=num_iters)

        self.transform = transform
            
    def set_aug_seed(self, idx):
        epoch_i = idx // 50000
        data_i = self.data_order[idx]
        iter_i = 50000 * epoch_i + data_i
        seed = int(self.aug_seeds[iter_i])
        set_seed(seed)

    def __len__(self):
        return len(self.data_order)

    def __getitem__(self, idx):
        img, lab = self.dset[self.data_order[idx]]
        self.set_aug_seed(idx)
        return self.transform(img), lab 

def get_loaders(order_seed, aug_seed, epochs):
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)

    train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    train_dset = DeterministicCIFAR(order_seed, aug_seed, epochs, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=8)
    test_dset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=8)

    return train_loader, test_loader

def get_model(model_seed):
    set_seed(model_seed)
    model = create_model().cuda()
    return model

def run_training(model_seed, order_seed, aug_seed):
    ## training hyperparams and setup
    EPOCHS = 50
    train_loader, test_loader = get_loaders(order_seed, aug_seed, epochs=EPOCHS)

    ne_iters = int(np.ceil(50000 / BATCH_SIZE))
    lr_schedule = np.interp(np.arange((EPOCHS+1) * ne_iters),
                            [0, 5*ne_iters, EPOCHS*ne_iters],
                            [0, 1, 0])

    model = get_model(model_seed)
    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    ## train
    model.train()
    losses = []
    it = tqdm(train_loader)
    for batch in it: 
        inputs, labels = batch
        with autocast():
            outputs = model(inputs.cuda())
            loss = loss_fn(outputs, labels.cuda())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_i = loss.item()
        losses.append(loss_i)
        it.set_postfix(loss='{:05.3f}'.format(loss_i))
        scheduler.step()
    
    ## evaluate
    model.eval()
    correct = 0 
    outputs_l = []
    it = iter(test_loader)
    with torch.no_grad():
        for i, batch in enumerate(it):
            inputs, labels = batch
            outputs = model(inputs.cuda())
            outputs_l.append(outputs)
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

    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    s = ','.join(str(a) for a in [args.model_seed, args.order_seed, args.aug_seed])
    arg_hash = hashlib.sha256(s.encode('utf-8')).hexdigest()
    log_path = os.path.join(log_dir, '%s.pkl' % arg_hash)
    log_d = {**kwargs, **stats}
    print('correct=%d' % log_d['correct'])
    with open(log_path, 'wb') as f:
        pickle.dump(log_d, f)

