import numpy as np
import os
import torch
import torchaudio
import tgt #textgrid parser

from .base_dataset import LibriSpeechBaseDataset
from .predicates import CPCAudioPredicates


class AudioDataset(LibriSpeechBaseDataset):
    '''Dataset for Noise Contrastive Estimation training based on LibriSpeech'''
    def __init__(self, root):
        super().__init__(root)
        self.pathes = self.filter_dataset_items(CPCAudioPredicates.longer_then_sample_length)


    def __getitem__(self, ix):
        '''
        Returns random 20480 size cut of audio waveform
        '''
        path = self.root + self.pathes[ix]
        waveform, bitrate = torchaudio.load(path)
        index = np.random.randint(waveform.size(1) - 20480 + 1)
        waveform = waveform.squeeze(0)[index:index+20480]

        return waveform



class PhonesDataset(LibriSpeechBaseDataset):
    '''Dataset for speaker classification based on LibriSpeech'''
    def __init__(self, root, lexicon_path):
        super().__init__(root)
        self.pathes = self.filter_dataset_items(CPCAudioPredicates.longer_then_sample_length)
        self.phone2ix = self.map_onehot(self._get_phones(lexicon_path))
        self.n_phones = len(self.phone2ix.keys())

        self.labels_lines = self.map_dataset_items(self._get_label_line)


    def _get_phones(self, lexicon_path):
        '''Returns list of unique phones of dataset'''
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

        return list(phones_set)


    def _get_label_line(self, path):
        '''
        Returns aligned labels list where each element corresponds to phone on
        audio on i-th timestep. By the paper setting, timestep is 0.01sec
        '''
        get_textgrid_path = lambda p: p[:-len('flac')] + 'TextGrid'
        textgrid_path = get_textgrid_path(path)

        textgrid = tgt.read_textgrid(self.root + textgrid_path)
        tg_len = textgrid.end_time - textgrid.start_time
        labels = [0 for x in range(int(tg_len * 100))]

        time_cur = textgrid.start_time
        tiers = textgrid.tiers[1]
        cur_tier = tiers[0]
        cur_tier_i = 0
        for i, lab in enumerate(labels):
            labels[i] = self.phone2ix[cur_tier.text]

            time_cur += 0.01
            if cur_tier.end_time < time_cur and cur_tier_i+1 < len(tiers):
                cur_tier_i += 1
                cur_tier = tiers[cur_tier_i]

        return tuple(labels) # tuple is more memory efficient


    def __getitem__(self, ix):
        '''
        Returns random 20480 size cut of audio waveform and corresponding
        cut of phones labels
        '''
        path = self.root + self.pathes[ix]

        waveform, bitrate = torchaudio.load(path)
        index_wav = np.random.randint(waveform.size(1) - 20480 + 1)
        sub_wav = waveform.squeeze(0)[index_wav:index_wav+20480]

        label_line = self.labels_lines[ix]
        index_labels = int(index_wav / 16000 * 100)
        sub_labels = torch.LongTensor(label_line[index_labels:index_labels+128])

        return sub_labels, sub_wav



class SpeakersDataset(LibriSpeechBaseDataset):
    '''Dataset for speaker classification based on LibriSpeech'''
    def __init__(self, root):
        super().__init__(root)
        self.pathes = self.filter_dataset_items(CPCAudioPredicates.longer_then_sample_length)

        self.speakers = self.map_dataset_items(self._get_speaker_by_path)
        self.n_speakers = len(set(self.speakers))

        self.speaker_ixs = self.map_onehot(list(set(self.speakers)))


    def _get_speaker_by_path(self, path):
        '''Extracts speaker id from path to audiofile
           Ex.: from path
           dev-clean/1272/128104/1272-128104-0000.flac
           extracts "1272" '''
        speaker = path[1:path[1:].index('/')+1] # faster then .split(/)[0]
        return speaker


    def __getitem__(self, ix):
        '''
        Returns random 20480 size cut of audio waveform and speaker label
        '''
        path = self.root + self.pathes[ix]

        speaker = self.speakers[ix]
        speaker_ix = self.speaker_ixs[speaker]

        waveform, bitrate = torchaudio.load(path)
        index = np.random.randint(waveform.size(1) - 20480 + 1)
        waveform = waveform.squeeze(0)[index:index+20480]

        return speaker_ix, waveform
