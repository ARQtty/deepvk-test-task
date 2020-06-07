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
from CPC.model import CPCModel
from CPC_classifiers.speaker_model import SpeakerClassificationCPC


config = Hparam('./CPC_classifiers/config.yaml')
gettime = lambda: str(dt.time(dt.now()))[:8]
if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')



if __name__ == "__main__":
    writer = SummaryWriter()
    
   
    print('Extracting data')
    dataset = SpeechDataset(config.data.path)
    train_dataset, test_dataset = train_test_split(dataset, test_size=config.train.test_split)
    
    batch_size = config.train.batch_size
    dataloader_fabric = lambda ds: DataLoader(ds, 
                                              batch_size=config.train.batch_size, 
                                              shuffle=config.train.shuffle_data, 
                                              drop_last=True)
    train_dl = dataloader_fabric(train_dataset)
    test_dl = dataloader_fabric(test_dataset)
    
    writer = SummaryWriter()
    train_step = 0
    test_step =  0

    model_cpc = CPCModel(Hparam(config.model.cpc_config_path)).to(config.train.device)
    model = SpeakerClassificationCPC(model_cpc, 
                                     config.model.hidden_size, 
                                     dataset.n_speakers,
                                     config.model.freeze_cpc_model).to(config.train.device)
    model.load_cpc_checkpoint('./checkpoints/' + config.train.cpc_checkpoint)

    opt = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    
    print('Training model')
    for e in range(1, config.train.epochs):
        print('[%s] Epoch %2d started' % (gettime(), e))
        
        print('  [%s] Train' % gettime())
        for i, batch in enumerate(train_dl, start=1):
            opt.zero_grad()

            speakers, utters = batch
            #speakers = F.one_hot(speakers, num_classes=ds.n_speakers).to(config.train.device)
            speakers = speakers.long().to(config.train.device)
            logits = model(utters.unsqueeze(1).to(config.train.device))
            logits = logits.squeeze(0)
            loss = criterion(logits, speakers)

            loss.backward()
            opt.step()

            writer.add_scalar('loss/train', loss.item(), train_step)
            writer.add_scalar('accuracy/train', SpeakerClassificationCPC.get_acc(logits, speakers).item(), train_step)

            train_step += 1


        print('  [%s] Test' % gettime())
        for i, batch in enumerate(test_dl):
            with torch.no_grad():
                speakers, utters = batch
                speakers = speakers.long().to(config.train.device)
                logits = model(utters.unsqueeze(1).to(config.train.device))
                logits = logits.squeeze(0)
                loss = criterion(logits, speakers)

                writer.add_scalar('loss/test', loss.item(), test_step)
                writer.add_scalar('accuracy/test', SpeakerClassificationCPC.get_acc(logits, speakers).item(), test_step)

                test_step += 1
                
        
        if e % config.train.save_every == 0:
            torch.save(model.state_dict(), config.train.checkpoints_dir + '/' + config.train.save_name + '_%d_epoch.pt' % e)