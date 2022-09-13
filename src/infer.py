import os
import sys
import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa.display as lrd
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from tqdm import tqdm

from utils import load_configs, set_seed, get_preprocessing, get_postprocessing
from models.r18_transconv_pixshuffle2 import VaeModel as VaeModelPixs2
from models.r18_transconv_pixshuffle import VaeModel as VaeModelPixs
from models.r18_transconv_pixshuffle_lrelu import VaeModel as VaeModelLeaky
from models.r18_transconv_pixshuffle_lrelu_nomel import VaeModel as VaeModelLeakyNoMel



def infer(cfg):
    print("# Sound Morphing VAE inference script")
    print("# Configutation: ", cfg.__dict__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(cfg.seed)
    if cfg.num_workers > 0:
        torch.set_num_threads(cfg.num_workers)

    print('# Creating pre- and post-processing')
    preproc = get_preprocessing(n_mels=cfg.n_mels, data_mean=cfg.data_mean, data_std=cfg.data_std, log_transform=True, train=False)
    postproc = get_postprocessing(n_mels=cfg.n_mels, data_mean=cfg.data_mean, data_std=cfg.data_std, log_transform=True, train=False)
    print(preproc)

    print('# Creating model and loading checkpoint')
    # cfg.model_type = 'pix_shuffle2'
    # cfg.fc_hidden3 = 896
    ModelClass = {
        'pix_shuffle2': VaeModelPixs2,
        'pix_shuffle': VaeModelPixs,
        'leakyrelu': VaeModelLeaky,
        'leakyrelu_nomel': VaeModelLeakyNoMel,
    }[cfg.model_type]
    model = ModelClass(loss=None, fc_hidden1=cfg.fc_hidden1, fc_hidden2=cfg.fc_hidden2, fc_hidden3=cfg.fc_hidden3, lspace_size=cfg.lspace_size)
    state_dict = torch.load(cfg.ckpt_path, map_location=device)
    print(model.load_state_dict(state_dict['model_state_dict'], strict=False))
    model = model.eval()
    print(model.get_infos())

    print(f'# Starting inference on task: {cfg.task}')
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.task == 'random':
        infer_random(model, cfg.n_samples, postproc, cfg.output_dir, cfg.lspace_size, device=device, sr=cfg.sr)
    elif cfg.task == 'interpolate':
        infer_interpolate(model, cfg.n_samples, preproc, postproc, cfg.output_dir, cfg.path1, cfg.path2, device=device, sr=cfg.sr)
    print('# Done.')


def infer_random(model, n_samples, postproc, output_dir, lspace_size, device='cpu', sr=16000):
    # random latent vectors
    z = torch.randn(n_samples, lspace_size).to(device)
    # decode
    with torch.no_grad():
        x_new = model.decode(z)
    x_new = x_new.cpu()
    melspecs = postproc.transforms[0](x_new)
    melspecs = melspecs.squeeze(1).numpy()
    waveforms = postproc(x_new)
    waveforms = waveforms.squeeze(1).numpy()
    # plots
    figure_path = os.path.join(output_dir, f'generated_specs.png')
    plot_specs(melspecs, figure_path)
    for i, waveform in tqdm(enumerate(waveforms)):
        sf.write(os.path.join(output_dir, f'generated_{i:04}.wav'), waveform, samplerate=sr, subtype='PCM_24')
    combined_waveform = np.concatenate(waveforms)
    sf.write(os.path.join(output_dir, f'generated_combined.wav'), combined_waveform, samplerate=sr, subtype='PCM_24')


def infer_interpolate(model, n_samples, preproc, postproc, output_dir, path1, path2, device='cpu', sr=16000):
    # load samples
    x1, _sr = torchaudio.load(path1)
    assert sr == _sr
    x1 = x1
    x2, _sr = torchaudio.load(path2)
    assert sr == _sr
    x2 = x2
    # preprocess and encode
    x_in = torch.stack([x1, x2])
    x_in = preproc(x_in).to(device)
    with torch.no_grad():
        z_in, _ = model.encode(x_in)
    z_in = z_in.cpu()
    # interpolate between points
    z = torch.stack([torch.linspace(a.item(), b.item(), n_samples) for a, b in z_in.T], dim=-1)
    z = z.to(device)
    # decode and postprocess
    with torch.no_grad():
        x_new = model.decode(z)
    x_new = x_new.cpu()
    melspecs = postproc.transforms[0](x_new)
    melspecs = melspecs.squeeze(1).numpy()
    waveforms = postproc(x_new)
    waveforms = waveforms.squeeze(1).numpy()
    # plots
    figure_path = os.path.join(output_dir, f'interpolated_specs.png')
    plot_specs(melspecs, figure_path)
    for i, waveform in tqdm(enumerate(waveforms)):
        sf.write(os.path.join(output_dir, f'interpolated_{i:04}.wav'), waveform, samplerate=sr, subtype='PCM_24')
    combined_waveform = np.concatenate(waveforms)
    sf.write(os.path.join(output_dir, f'interpolated_combined.wav'), combined_waveform, samplerate=sr, subtype='PCM_24')


def plot_specs(x_specs, figure_path, sr=16000):
    batch_size = x_specs.shape[0]
    ncols = 8
    nrows = batch_size // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 2 * nrows), sharex='col', sharey='row')
    for i, (ax, x) in enumerate(zip(axs.flatten(), x_specs)):
        lrd.specshow(x, ax=ax, cmap='magma', sr=sr, n_fft=1024, win_length=1024, hop_length=256, x_axis='s', y_axis='mel')
        ax.set_title(f'{i:04}', y=1.0, pad=-14, color='w', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.tight_layout()
    plt.savefig(figure_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description='Sound Morphing VAE inference script')
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('task', choices=['random', 'interpolate'], type=str)
    parser.add_argument('-a', '--path1', type=str, required='interpolate' in sys.argv)
    parser.add_argument('-b', '--path2', type=str, required='interpolate' in sys.argv)
    parser.add_argument('-n', '--n_samples', type=int, default=16)
    parser.add_argument('-o', '--output_dir', type=str, default='./outputs/')
    parser.add_argument('-c','--configs_path', type=str, default='./configs.yaml')
    args = parser.parse_args()
    cfg = load_configs(**vars(args))
    infer(cfg)
