#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

from opacus import PrivacyEngine
import torch.optim as optim
from torch.utils.data import DataLoader


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, privacy=1):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pBudget = privacy

    def train(self, net):
        print("Starting local..", self.pBudget)
        #model = net
    #    net.train()
        #model.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        DELTA = 5e-4             #Should be less than 1/# data items 

        # enter PrivacyEngine
#        privacy_engine = PrivacyEngine()
      #  model, optimizer, data_loader = privacy_engine.make_private(
       #     module=net,
       #     optimizer=optimizer,
       #     data_loader=self.ldr_train,
       #     noise_multiplier=1.1,
       #     max_grad_norm=1.0,
        #)

#        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
#            module=net,
#            optimizer=optimizer,
 #           data_loader=self.ldr_train,
#            target_epsilon=self.pBudget,
 #           target_delta=DELTA,
#            max_grad_norm=1.0,
#            epochs=15
#        )

        model = net
        #data_loader = self.ldr_train
        model.train()

        #net.train()

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                #images = images.unsqueeze(0)
                #images = images.double()
                optimizer.zero_grad()
                labels = labels.float()
                images = images.float()
                net.zero_grad()
                log_probs = net(images)
                labels = labels.long()
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
      #  epsilon = privacy_engine.get_epsilon(DELTA)
        epsilon = 0
        print(f"(ε = {epsilon:.2f}, δ = {DELTA})")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), epsilon

