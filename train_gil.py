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
from GIL.model import GILModel
from GIL.freezers import SimultaneousFreezer, IterativeFreezer


config = Hparam('./GIL/config.yaml')
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
    model = GILModel(config, writer).to(config.train.device)

    if config.train.unfreezing.type == 'iterative':
        freezer = IterativeFreezer(config, model)
    elif config.train.unfreezing.type == 'simultaneous':
        freezer = SimultaneousFreezer(config, model)
    updates = 0

    if config.train.unfreezing.type == 'iterative':
        # modules are unfreezed since initialization
        for i in range(1, 5):
            model.freeze_block(i)

    if config.train.start_epoch != 1:
        model.load_state_dict(torch.load(config.train.start_checkpoint, device=config.train.device))

    print('Training model')
    for e in range(config.train.start_epoch, config.train.epochs):
        print('[%s] Epoch %2d started' % (gettime(), e))

        print('  [%s] Train' % gettime())
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader, start=1)):

            utters = batch
            model(utters.unsqueeze(1).to(config.train.device))

            updates += 1
            freezer.maybe_freeze(updates)

        print('  [%s] Test' % gettime())
        model.eval()
        for i, batch in tqdm(enumerate(test_dataloader, start=1)):
            with torch.no_grad():
                utters = batch
                model(utters.unsqueeze(1).to(config.train.device))


        if e % config.train.save_every == 0:
            torch.save(model.state_dict(), config.train.checkpoints_dir + '/' + config.train.save_name + '_%d_epoch.pt' % e)
