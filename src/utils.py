import os
import random
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Normalize, Compose


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def set_seed(seed):
    """
    Fix all possible sources of randomness
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_preprocessing(train, sr=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=80, data_mean=None, data_std=None, log_transform=False):
    preprocessings = [MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)]
    if log_transform:
        preprocessings += [AmplitudeToDB()]
    if data_mean is not None:
        preprocessings += [Normalize(data_mean, data_std)]
    return Compose(preprocessings)


def get_dataloader(dataset, batch_size, num_workers=0, shuffle=False, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


def chart_dependencies(model, input_shape):
    """
    Use backprop to chart dependencies
    (see http://karpathy.github.io/2019/04/25/recipe/)
    """
    model.eval()
    inputs = torch.randn(input_shape)
    inputs.requires_grad = True
    outputs = model(inputs)
    random_index = random.randint(0, input_shape[0])
    loss = outputs[random_index].sum()
    loss.backward()
    assert (torch.cat([inputs.grad[i] == 0 for i in range(input_shape[0]) if i != random_index])).all() and (
        inputs.grad[random_index] != 0
    ).any(), f"Only index {random_index} should have non-zero gradients"


def get_now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")