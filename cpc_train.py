from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from hparams import Hparam
from data.dataset import SpeechDataset
from cpc_model import CPCModel


config = Hparam('./cpc_config.yaml')
gettime = lambda: str(dt.time(dt.now()))[:8]



if __name__ == "__main__":
    writer = SummaryWriter()

    print('Extracting data')
    dataset = SpeechDataset(config.data.path)
    dataloader = torch.utils.data.DataLoader(dataset, config.train.batch_size, shuffle=True)

    model = CPCModel(config).to(config.train.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    criterion = F.binary_cross_entropy_with_logits

    abs_step = 0


    print('Training model')
    for e in range(1, config.train.epochs):
        print('[%s] Epoch %2d started' % (gettime(), e))

        for i, batch in enumerate(dataloader):
            opt.zero_grad()

            speakers, utters = batch
            logits, labels = model(utters.unsqueeze(1).to(config.train.device))

            for j in range(len(logits)):
                loss = criterion(logits[j], labels[j])
                loss.backward(retain_graph=True)
                accuracy = CPCModel.get_accuracy(logits[j], labels[j])
                writer.add_scalar('Loss/loss_%d' % j, loss.item(), abs_step)
                writer.add_scalar('Accuracy/accuracy_%d' % j, accuracy.item(), abs_step)

            abs_step+=1

            opt.step()

        if e % config.train.save_every == 0:
            torch.save(model.state_dict(), config.train.save_name + '_%d_epoch.pt' % e)
