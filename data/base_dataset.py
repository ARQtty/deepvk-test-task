import os
import torch
from torch.utils.data import Dataset



class LibriSpeechBaseDataset(Dataset):
    '''
    Base dataset object provides vectorized functions for manipulating its audio files
    '''
    def __init__(self, root):
        self.root = root
        self.pathes = self._get_files_pathes()


    def _get_files_pathes(self):
        '''Fetches all audio files pathes of dataset'''
        pathes = []

        # Walk over the dataset
        for path, dirs, files in os.walk(self.root):
            speaker = '/'.join(path.split('/')[-2:])
            # select only sound files
            files = list(filter(lambda s: s.endswith('.flac'), files))
            if len(files) > 0:
                pathes += list(map(lambda s: '/%s/%s' % (speaker, s), files))

        pathes = list(filter(lambda s: s is not None, pathes))
        return pathes


    def filter_dataset_items(self, predicate):
        '''
        Applies given predicate to every dataset item (file path)
        and filtering not suitable items by that predicate
        Returns filtred dataset items
        '''
        for i, path in enumerate(self.pathes):
            suitable = predicate(self.root + path)
            if not suitable:
                self.pathes[i] = None

        pathes = list(filter(lambda p: p is not None, self.pathes))
        return pathes


    def map_dataset_items(self, predicate):
        result = []
        for i, path in enumerate(self.pathes):
            item = predicate(path)
            result.append(item)

        return result


    def map_onehot(self, categories):
        return {cat: i for i, cat in enumerate(sorted(categories))}


    def train_test_split_ixs(self, test_size):
        '''
        Returns two lists of randomly shuffled indexes: for train and test part

        Used cause of sklearn train_test_split forces dataset items to be loaded
        into memory as <list>. It costs too much for audio dataset
        '''
        split_ix = int(len(self) * (1-test_size))

        ixs = torch.tensor(list(range(len(self))))
        ixs = ixs[torch.randperm(len(ixs))]

        train_ixs = ixs[:split_ix]
        test_ixs = ixs[split_ix:]

        return train_ixs, test_ixs


    def __len__(self):
        return len(self.pathes)


    def __getitem__(self, ix):
        raise NotImplementedError
