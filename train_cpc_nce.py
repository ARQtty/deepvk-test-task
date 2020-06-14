from datetime import datetime as dt
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from hparams import Hparam
from data.dataset import AudioDataset
from CPC_true_NCE.model import CPCModel_NCE


config = Hparam('./CPC_true_NCE/config.yaml')
gettime = lambda: str(dt.time(dt.now()))[:8]
if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')



if __name__ == "__main__":
    writer = SummaryWriter()

    print('Extracting data')
    dataset = AudioDataset(config.data.path)
    train_ixs, test_ixs =  dataset.train_test_split_ixs(config.train.test_size)
    train_sampler = SubsetRandomSampler(train_ixs)
    test_sampler = SubsetRandomSampler(test_ixs)

    dataloader_fabric = lambda ds, sampler: DataLoader(ds, config.train.batch_size, sampler=sampler, drop_last=True)
    train_dataloader = dataloader_fabric(dataset, train_sampler)
    test_dataloader = dataloader_fabric(dataset, test_sampler)


    print('Creating model')
    model = CPCModel_NCE(config).to(config.train.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    criterion = F.binary_cross_entropy_with_logits

    abs_step_train = 0
    abs_step_test = 0


    if config.train.start_epoch != 1:
        model.load_state_dict(torch.load(config.train.checkpoints_dir + '/' + config.train.start_checkpoint, device=config.train.device))

    print('Training model')
    for e in range(config.train.start_epoch, config.train.epochs):
        print('[%s] Epoch %2d started' % (gettime(), e))

        print('  [%s] Train' % gettime())
        for i, utters in tqdm(enumerate(train_dataloader, start=1)):
            opt.zero_grad()

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



        for i, utters in tqdm(enumerate(test_dataloader, start=1)):
            with torch.no_grad():
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
