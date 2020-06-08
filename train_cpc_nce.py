from datetime import datetime as dt
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

from hparams import Hparam
from data.dataset import SpeechDataset
from CPC_true_NCE.model import CPCModel_NCE


config = Hparam('./CPC_true_NCE/config.yaml')
gettime = lambda: str(dt.time(dt.now()))[:8]
if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')



if __name__ == "__main__":
    writer = SummaryWriter()

    print('Extracting data')
    dataset = SpeechDataset(config.data.path)
    dataset_train, dataset_test = train_test_split(dataset, test_size=config.train.test_size)
    dataloader_fabric = lambda ds: DataLoader(ds, config.train.batch_size, shuffle=True)
    train_dataloader = dataloader_fabric(dataset_train)
    test_dataloader = dataloader_fabric(dataset_test)

    model = CPCModel_NCE(config).to(config.train.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    criterion = F.binary_cross_entropy_with_logits

    abs_step_train = 0
    abs_step_test = 0
    

    print('Training model')
    for e in range(1, config.train.epochs):
        print('[%s] Epoch %2d started' % (gettime(), e))

        print('  [%s] Train' % gettime())
        for i, batch in tqdm(enumerate(train_dataloader, start=1)):
            opt.zero_grad()

            speakers, utters = batch
            losses, logits, labels = model(utters.unsqueeze(1).to(config.train.device))

            for j in range(config.model.predict_steps):
                loss = losses[j]
                loss.backward(retain_graph=True)
                accuracy = CPCModel_NCE.get_accuracy(logits[j], labels[j])
                roc_auc =  CPCModel_NCE.get_auc_score(logits[j], labels[j])
                
                writer.add_scalar('Train_loss/loss_%d' % j, loss.item(), abs_step_train)
                writer.add_scalar('Train_accuracy/accuracy_%d' % j, accuracy, abs_step_train)
                writer.add_scalar('Train_auc/auc_%d' % j, roc_auc, abs_step_train)

            abs_step_train+=1

            opt.step()
        
        

        for i, batch in tqdm(enumerate(test_dataloader, start=1)):
            with torch.no_grad():
                speakers, utters = batch
                losses, logits, labels = model(utters.unsqueeze(1).to(config.train.device))

                for j in range(config.model.predict_steps):
                    loss = losses[j]
                    roc_auc =  CPCModel_NCE.get_auc_score(logits[j], labels[j])
                    accuracy = CPCModel_NCE.get_accuracy(logits[j], labels[j])
                    
                    writer.add_scalar('Test_loss/loss_%d' % j, loss.item(), abs_step_test)
                    writer.add_scalar('Test_accuracy/accuracy_%d' % j, accuracy, abs_step_test)
                    writer.add_scalar('Test_auc/auc_%d' % j, roc_auc, abs_step_test)

                abs_step_test+=1


        if e % config.train.save_every == 0:
            torch.save(model.state_dict(), config.train.checkpoints_dir + '/' + config.train.save_name + '_%d_epoch.pt' % e)