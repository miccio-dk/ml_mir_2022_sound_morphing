from cProfile import label
import os
import random
import datetime
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa.display as lrd
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Normalize, Compose
from sklearn.manifold import TSNE


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
    outputs = model(inputs, label=[])
    random_index = random.randint(0, input_shape[0])
    loss = outputs[0][random_index].sum()
    loss.backward()
    assert (torch.cat([inputs.grad[i] == 0 for i in range(input_shape[0]) if i != random_index])).all() and (
        inputs.grad[random_index] != 0
    ).any(), f"Only index {random_index} should have non-zero gradients"


def get_now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def store_checkpoint(model, optimizer, current_epoch):
    ckpt_path = f'checkpoints/epoch_{current_epoch:04}.pth'
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    print(f'# Stored checkpoint at {ckpt_path}')


def plot_reconstructions(x_true, x_reconst, current_epoch, sr=16000):
    batch_size = x_true.shape[0]
    nrows = 4
    ncols = batch_size // nrows
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 8))
    for i, (ax, xt, xh) in enumerate(zip(axs.flatten(), x_true, x_reconst)):
        xx = torch.dstack([xt, xh]).squeeze(0).detach().cpu().numpy()
        lrd.specshow(xx, ax=ax, cmap='magma', sr=sr, n_fft=1024, win_length=1024, hop_length=256)
        ax.set_title(f'{i}')
    fig.tight_layout()
    figure_path = os.path.join('figures', f'reconstructions_epoch_{current_epoch:04}.png')
    plt.savefig(figure_path)
    plt.close(fig)
    return figure_path


def plot_latentspace(mu, labels, current_epoch):
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    mu_embs = TSNE(n_components=2, n_iter=5000).fit_transform(mu)
    labels['emb_1'] = mu_embs[:, 0]
    labels['emb_2'] = mu_embs[:, 1]
    sns.scatterplot(x='emb_1', y='emb_2', hue='instrument_family_str', size='velocity', style='instrument_source_str', data=labels, ax=ax)
    fig.tight_layout()
    figure_path = os.path.join('figures', f'latentspace_epoch_{current_epoch:04}.png')
    plt.savefig(figure_path)
    plt.close(fig)
    return figure_path