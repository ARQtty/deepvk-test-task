import numpy as np
import os
from tqdm import tqdm
import torch
import torchaudio


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.speakers, self.pathes = SpeechDataset.get_s2u(root)
        self.speaker_ixs = self.get_onehot_mapping(list(set(self.speakers)))
        self.n_speakers = len(self.speaker_ixs.values())


    def get_onehot_mapping(self, categories):
        return {cat: i for i, cat in enumerate(sorted(categories))}


    @staticmethod
    def get_s2u(root):
        speakers = []
        pathes = []

        for path, dirs, files in os.walk(root):
            speaker = '/'.join(path.split('/')[-2:])
            files = list(filter(lambda s: s.endswith('.flac'), files))
            if len(files) > 0:
                speakers += [speaker for fname in files]
                pathes += list(map(lambda s: '/%s/%s' % (speaker, s), files))
                
        for i in tqdm(range(len(pathes))):
            # we should check if song longer then 1,28sec
            # it's about 40kB
            
            # sampling ration is 16k in a whole dataset, so if
            # file is larger then 50kB, it is defenetly longer then 1.28sec
            if os.path.getsize(root+pathes[i]) > 50000:
                continue
            
            utter, bitrate = torchaudio.load(root + pathes[i])
            
            if utter.size(1) < 20480:
                speakers[i] = None
                pathes[i] = None
        speakers = list(filter(lambda s: s is not None, speakers))
        pathes = list(filter(lambda s: s is not None, pathes))

        return speakers, pathes


    def __len__(self):
        return len(self.pathes)


    def __getitem__(self, ix):
        speaker = self.speakers[ix]
        speaker_ix = self.speaker_ixs[speaker]
        path = self.root + self.pathes[ix]
        waveform, bitrate = torchaudio.load(path)
        if bitrate != 16000: print('not 16k bitrate!')
        index = np.random.randint(waveform.size(1) - 20480 + 1)
        return speaker_ix, waveform.squeeze(0)[index:index+20480]
