import os.path as osp

import torch
import pandas as pd
import torchaudio
from tqdm import tqdm

from torch.utils.data import Dataset


# generic sofa dataset
class NsynthDataset(Dataset):
    def __init__(self, dataset_path, transform=None, sr=16000, duration=4,
                 pitches=None, velocities=None, instrument_sources=None, instrument_families=None, label='onehot'):
        self.dataset_path = dataset_path
        self.transform = transform
        self.sr = sr
        self.duration = int(sr * duration)
        self.pitches = pitches
        self.velocities = velocities
        self.instrument_sources = instrument_sources
        self.instrument_families = instrument_families
        self.label = label
        self.df = None
        self.load_data()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        filepath = osp.join(self.dataset_path, 'audio', f'{item.name}.wav')
        sample, sr = torchaudio.load(filepath)
        assert sr == self.sr
        assert sample.shape[0] == 1
        paddings = (0, self.duration - sample.shape[1])
        sample = torch.nn.functional.pad(sample, paddings)
        if self.transform:
            sample = self.transform(sample)
        if self.label == 'onehot':
            label = self.onehot[idx]
        else:
            label = item.drop(['qualities_str', 'qualities']).to_dict()
        return sample, label

    def load_data(self):
        filepath_cache = osp.join(self.dataset_path, 'examples_cache.pkl')
        if osp.exists(filepath_cache):
            #print(f'Loading cached data: {filepath_cache}')
            _df = pd.read_pickle(filepath_cache)
        else:
            filepath = osp.join(self.dataset_path, 'examples.json')
            #print(f'Caching data: {filepath}')
            _df = pd.read_json(filepath).T
            _df.to_pickle(filepath_cache)
        # filter data
        if self.pitches:
            _df = _df[_df['pitch'].isin(self.pitches)]
        if self.velocities:
            _df = _df[_df['velocity'].isin(self.velocities)]
        if self.instrument_sources:
            _df = _df[_df['instrument_source'].isin(self.instrument_sources)]
        if self.instrument_families:
            _df = _df[_df['instrument_family'].isin(self.instrument_families)]
        _df['instrument'] = _df['instrument_source_str'].str.cat(_df['instrument_family_str'], sep=' ')
        self.onehot = pd.get_dummies(_df['instrument']).to_numpy()
        self.df = _df
        #print(f'Data: {_df.shape}')
    
    def get_statistics(self):
        all_data = []
        for i in tqdm(range(self.__len__())):
            sample, _ = self.__getitem__(i)
            all_data.append(sample)
        all_data = torch.stack(all_data)
        return all_data.mean(), all_data.std()

    def get_n_classes(self):
        return len(self.df['instrument'].unique())