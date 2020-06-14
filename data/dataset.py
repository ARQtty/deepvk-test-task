import numpy as np
import os
from tqdm import tqdm
import torch
import torchaudio
import tgt #textgrid parser



class PhonesDataset(torch.utils.data.Dataset):
    def __init__(self, root, lexicon_path):
        self.root = root
        self.phone2ix = self._get_phones_dict(lexicon_path)
        self.pathes, self.labels_lines = self.get_p2l(root)
        self.n_phones = len(self.phone2ix.keys())
    

    def _get_phones_dict(self, lexicon_path):
        '''Creates phones to phone_ix mapping'''
        # read lexicon file
        data_dict = open(lexicon_path).read().split('\n')
        phones_set = set()

        # read phones of every word and add to phones set
        for line in data_dict:
            if len(line) < 2:
                continue

            word, (*phones) = line.strip().split()
            for phone in phones:
                phones_set.add(phone)

        # add metaphones
        phones_set.add('sil')
        phones_set.add('sp')
        phones_set.add('spn')

        # cast to phone->ix dict
        phone2ix = dict()
        for i, phone in enumerate(list(phones_set)):
            phone2ix[phone] = i
            
        return phone2ix
        
        
    def _get_textgrid_path_from_audio(self, audio_path):
        return audio_path[:-len('flac')] + 'TextGrid'
        
        
    def _get_labels_by_textgrid(self, tg_path):
        tg = tgt.read_textgrid(tg_path)
        tg_len = tg.end_time - tg.start_time
        labels = [0 for x in range(int(tg_len * 100))]

        time_cur = tg.start_time
        tiers = tg.tiers[1]
        cur_tier = tiers[0]
        cur_tier_i = 0
        for i, lab in enumerate(labels):
            labels[i] = self.phone2ix[cur_tier.text]

            time_cur += 0.01
            if cur_tier.end_time < time_cur and cur_tier_i+1 < len(tiers):
                cur_tier_i += 1
                cur_tier = tiers[cur_tier_i]

        return tuple(labels) # more memory efficient
    

    def get_p2l(self, root):
        pathes = [] 
        labels = []

        for path, dirs, files in os.walk(root):
            speaker = '/'.join(path.split('/')[-2:])
            files = list(filter(lambda s: s.endswith('.flac'), files))
            if len(files) > 0:
                pathes += list(map(lambda s: '/%s/%s' % (speaker, s), files))
                
        for i in tqdm(range(len(pathes))):
            textgrid_path = self._get_textgrid_path_from_audio(root+pathes[i])
            label_line = self._get_labels_by_textgrid(textgrid_path)
            labels.append(label_line)
            # we should check if audio is longer then 1,28sec
            # it's about 40kB
            
            # sampling ratio is 16k in a whole dataset, so if
            # file is larger then 50kB, it is defenetly longer then 1.28sec
            if os.path.getsize(root+pathes[i]) > 50000:
                continue
            
            utter, bitrate = torchaudio.load(root + pathes[i])
            if utter.size(1) < 20480:
                labels[i] = None
                pathes[i] = None
  
        pathes = list(filter(lambda s: s is not None, pathes))
        labels = list(filter(lambda s: s is not None, labels))

        return pathes, labels


    def __len__(self):
        return len(self.pathes)


    def __getitem__(self, ix):
        label_line = self.labels_lines[ix]
        
        path = self.root + self.pathes[ix]        
        waveform, bitrate = torchaudio.load(path)
        index_wav = np.random.randint(waveform.size(1) - 20480 + 1)
        index_lab = int(index_wav / 16000 * 100)
        
        sub_wav = waveform.squeeze(0)[index_wav:index_wav+20480]
        sub_lab = torch.LongTensor(label_line[index_lab:index_lab+128])

        return sub_lab, sub_wav

    
    def train_test_split_ixs(self, test_size):
        split_ix = int(len(self) * (1-test_size))
        
        ixs = torch.tensor(list(range(len(self))))
        ixs = ixs[torch.randperm(len(ixs))]
        
        train_ixs = ixs[:split_ix]
        test_ixs = ixs[split_ix:]
        
        return train_ixs, test_ixs



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
        index = np.random.randint(waveform.size(1) - 20480 + 1)

        return speaker_ix, waveform.squeeze(0)[index:index+20480]

    
    def train_test_split_ixs(self, test_size):
        split_ix = int(len(self) * (1-test_size))
        
        ixs = torch.tensor(list(range(len(self))))
        ixs = ixs[torch.randperm(len(ixs))]
        
        train_ixs = ixs[:split_ix]
        test_ixs = ixs[split_ix:]
        
        return train_ixs, test_ixs