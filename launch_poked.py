"""
Runs a series of 32-epoch trainings, all with the same random seed,
but "poked" at varying points during training.

This is in order to test the sensitivity of training process to small
perturbations.
"""

import os
import uuid
import numpy as np
from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch import nn
from torch.optim import SGD, lr_scheduler

from model import get_model
from data import get_loaders

def poke_model(model):
    w = model[0][0].weight.data
    w[0, 0, 0, 0] *= 1.001

def get_outs(model):
    model.eval()
    correct = 0 
    outputs_l = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs.cuda())
            outputs_l.append(outputs.clone())
    outputs = torch.cat(outputs_l).cpu()
    return outputs

def churn(model1, model2, test_loader):
    outs1 = get_outs(model1)
    outs2 = get_outs(model2)
    return (outs1.argmax(-1) != outs2.argmax(-1)).sum().item()
    
def run_training(poke_step, epochs, seed, train_loader0):
    
    train_loader = train_loader0[:100*epochs]
    n_iters = len(train_loader)
    lr_schedule = np.interp(np.arange(1+n_iters), [0, n_iters], [1, 0]) 

    model = get_model(seed).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    loss_fn = nn.CrossEntropyLoss()

    ## train
    model.train()
    it = tqdm(train_loader)
    for i, (inputs, labels) in enumerate(it):

        if i == poke_step:
            poke_model(model)

        outputs = model(inputs.cuda())
        loss = loss_fn(outputs, labels.cuda())
        it.set_postfix(loss='{:05.3f}'.format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model

if __name__ == '__main__':
    seed = 42
    epochs = 32
    train_loader, test_loader = get_loaders(seed, seed, epochs=epochs)
    # Pre-generate all training batches, for efficiency sake.
    train_loader0 = [batch for batch in tqdm(train_loader)]

    # Train the unperturbed model.
    model1 = run_training(-1, epochs, seed, train_loader0)
    # Run various trainings with a perturbation applied at some step.
    xx = []
    yy = []
    for poke_step in range(0, 100*epochs+1, 100):
        model2 = run_training(poke_step, epochs, seed, train_loader0)
        c = churn(model1, model2, test_loader) # Measure disagreement to unperturbed model
        xx.append(poke_step)
        yy.append(c)
    
    # xx: step that was poked; yy: disagreement to undisturbed model
    obj = {'xx': xx, 'yy': yy, 'seed': seed, 'epochs': epochs}
    os.makedirs('./logs', exist_ok=True)
    out_path = os.path.join('./logs/%s.pt' % uuid.uuid4())
    print('Writing logs to %s' % out_path)
    torch.save(obj, out_path)
    
