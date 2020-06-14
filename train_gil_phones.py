from datetime import datetime as dt
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from hparams import Hparam
from data.datasets import PhonesDataset
from GIL_model.model import GILModel
from GIL_model.freezers import SimultaneousFreezer, IterativeFreezer
from classifier_models.phone_model import PhonesClassificationModel


config = Hparam('./classifier_models/config_phone_gil.yaml')
gettime = lambda: str(dt.time(dt.now()))[:8]
if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')



if __name__ == "__main__":
    writer = SummaryWriter()

    print('Extracting data')
    dataset = PhonesDataset(config.data.path, config.data.lexicon)
    train_ixs, test_ixs =  dataset.train_test_split_ixs(config.train.test_split)
    train_sampler = SubsetRandomSampler(train_ixs)
    test_sampler = SubsetRandomSampler(test_ixs)

    dataloader_fabric = lambda ds, sampler: DataLoader(ds, config.train.batch_size, sampler=sampler, drop_last=True)
    train_dl = dataloader_fabric(dataset, train_sampler)
    test_dl = dataloader_fabric(dataset, test_sampler)

    train_step = 0
    test_step =  0

    print('Creating model')
    gil_cfg = Hparam(config.model.cpc_config_path)
    model_cpc = GILModel(gil_cfg, writer).to(config.train.device)
    if gil_cfg.train.unfreezing.type == 'iterative':
        freezer = IterativeFreezer(gil_cfg, model_cpc)
    elif gil_cfg.train.unfreezing.type == 'simultaneous':
        freezer = SimultaneousFreezer(gil_cfg, model_cpc)
    updates = 0

    if gil_cfg.train.unfreezing.type == 'iterative':
        # modules are unfreezed since initialization
        for i in range(1, 5):
            model_cpc.freeze_block(i)

    model = PhonesClassificationModel(model_cpc,
                                     config.model.hidden_size,
                                     dataset.n_phones,
                                     config.model.freeze_cpc_model).to(config.train.device)
    if config.train.load_cpc_checkpoint:
        model.load_cpc_checkpoint(config.train.checkpoints_dir + '/' + config.train.cpc_checkpoint)

    opt = torch.optim.Adadelta(model.parameters())#, lr=config.train.lr)
    criterion = torch.nn.CrossEntropyLoss()


    print('Training model')
    for e in range(1, config.train.epochs):
        print('[%s] Epoch %2d started' % (gettime(), e))

        print('  [%s] Train' % gettime())
        for i, batch in enumerate(train_dl, start=1):
            opt.zero_grad()

            labels, utters = batch
            #speakers = F.one_hot(speakers, num_classes=ds.n_speakers).to(config.train.device)
            labels = labels.long().view(-1).to(config.train.device)
            logits = model(utters.unsqueeze(1).to(config.train.device))
            logits = logits.view(-1, dataset.n_phones)
            loss = criterion(logits, labels)

            loss.backward()
            opt.step()

            updates += 1
            freezer.maybe_freeze(updates)

            writer.add_scalar('loss/train', loss.item(), train_step)
            writer.add_scalar('accuracy/train', PhonesClassificationModel.get_acc(logits, labels).item(), train_step)

            train_step += 1


        print('  [%s] Test' % gettime())
        for i, batch in enumerate(test_dl):
            with torch.no_grad():
                labels, utters = batch
                #speakers = F.one_hot(speakers, num_classes=ds.n_speakers).to(config.train.device)
                labels = labels.long().view(-1).to(config.train.device)
                logits = model(utters.unsqueeze(1).to(config.train.device))
                logits = logits.view(-1, dataset.n_phones)
                loss = criterion(logits, labels)

                writer.add_scalar('loss/test', loss.item(), test_step)
                writer.add_scalar('accuracy/test', PhonesClassificationModel.get_acc(logits, labels).item(), test_step)

                test_step += 1


        if e % config.train.save_every == 0:
            torch.save(model.state_dict(), config.train.checkpoints_dir + '/' + config.train.save_name + '_%d_epoch.pt' % e)
