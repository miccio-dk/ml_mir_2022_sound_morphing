import sys
import math
import wandb
import torch
import pandas as pd

from argparse import ArgumentParser
from tqdm import tqdm
from utils import (load_configs, set_seed, get_preprocessing, get_dataloader, chart_dependencies, kld_scheduler,
        get_now, store_checkpoint, plot_reconstructions, plot_latentspace, plot_latentspace_pca)
from dataset import NsynthDataset
from loss import VaeLoss, VaeLossClasses

from models.r18_transconv_pixshuffle2 import VaeModel as VaeModelPixs2
from models.r18_transconv_pixshuffle import VaeModel as VaeModelPixs
from models.r18_transconv_pixshuffle_lrelu import VaeModel as VaeModelLeaky
from models.r18_transconv_pixshuffle_lrelu_nomel import VaeModel as VaeModelLeakyNoMel

def train(cfg):
    print("# Sound Morphing VAE training script")
    print("# Configutation: ", cfg.__dict__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(cfg.seed)
    if cfg.num_workers > 0:
        torch.set_num_threads(cfg.num_workers)

    print('# Creating preprocessing, datasets, dataloaders')
    preproc_train = get_preprocessing(
        n_mels=cfg.n_mels, n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length, 
        data_mean=cfg.data_mean, data_std=cfg.data_std, 
        log_transform=True, train=True, db_range=None)
    preproc_train_augm = get_preprocessing(
        n_mels=cfg.n_mels, n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length, 
        data_mean=cfg.data_mean, data_std=cfg.data_std, 
        log_transform=True, train=True, db_range=cfg.db_range)
    preproc_valid = get_preprocessing(
        n_mels=cfg.n_mels, n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length, 
        data_mean=cfg.data_mean, data_std=cfg.data_std, 
        log_transform=True, train=False)
    ds_train = NsynthDataset(cfg.datapath_train, sr=cfg.sr, duration=cfg.duration, pitches=[60], transform=preproc_train, augm_transform=preproc_train_augm, label='both')
    ds_valid = NsynthDataset(cfg.datapath_valid, sr=cfg.sr, duration=cfg.duration, pitches=[60], transform=preproc_valid, label='full')
    dl_train = get_dataloader(ds_train, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, shuffle=True)
    dl_valid = get_dataloader(ds_valid, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, shuffle=False)
    n_channels, n_features, n_timeframes = ds_train[0][0].shape
    n_classes = ds_train.get_n_classes()
    print('Preprocessing for clean_train/augm_train/validation:')
    print(preproc_train)
    print(preproc_train_augm)
    print(preproc_valid)
    print(f'Training: {len(ds_train)} datapoints, {len(dl_train)} batches, {n_classes} classes')
    print(f'Validation: {len(ds_valid)} datapoints, {len(dl_valid)} batches')

    print('# Creating loss, model, optimizer')
    LossClass = {
        'vae': VaeLoss,
        'classes': VaeLossClasses,
    }[cfg.loss_type]
    ModelClass = {
        'pix_shuffle2': VaeModelPixs2,
        'pix_shuffle': VaeModelPixs,
        'leakyrelu': VaeModelLeaky,
        'leakyrelu_nomel': VaeModelLeakyNoMel,
    }[cfg.model_type]
    loss = LossClass(rec_weight=cfg.rec_weight, kld_weight=cfg.kld_weight, ce_weight=cfg.ce_weight, lspace_size=cfg.lspace_size, n_classes=n_classes)
    model = ModelClass(loss=loss, fc_hidden1=cfg.fc_hidden1, fc_hidden2=cfg.fc_hidden2, fc_hidden3=cfg.fc_hidden3, lspace_size=cfg.lspace_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.start_lr)
    chart_dependencies(model, input_shape=(cfg.batch_size, n_channels, n_features, n_timeframes))
    print(model.get_infos())

    run_name = get_now()
    wandb_run = wandb.init(
        name=run_name,
        project='mir2_soundmorph',
        entity='ml4mir2022',
        config=cfg.__dict__,
        job_type='train'
    )
    wandb_run.watch(model, log='all')

    print('# Starting training loop')
    training_loop(
        epochs=cfg.epochs,
        model=model,
        optimizer=optimizer,
        dl_train=dl_train,
        dl_valid=dl_valid,
        kld_weight=cfg.kld_weight,
        kld_exp=cfg.kld_exp,
        val_every=cfg.val_every,
        device=device,
        wandb_run=wandb_run,
        log_pictures=cfg.log_pictures,
    )

    print('# Starting testing')
    ds_test = NsynthDataset(cfg.datapath_test, sr=cfg.sr, duration=cfg.duration, pitches=[60], transform=preproc_valid, label='full')
    dl_test = get_dataloader(ds_test, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, shuffle=False, drop_last=False)
    print(f'Testing: {len(ds_test)} datapoints, {len(dl_test)} batches')
    test(
        model=model,
        dl_test=dl_test,
        device=device,
        wandb_run=wandb_run,
        log_pictures=cfg.log_pictures,
    )
    wandb_run.finish()


def training_loop(epochs, model, optimizer, dl_train, dl_valid, val_every, kld_weight, kld_exp, device='cpu', wandb_run=None, log_pictures=False):
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        run_validation = (epoch % val_every == 0 or epoch == 1 or epoch == epochs)
        model.loss.kld_weight = kld_scheduler(kld_weight, kld_exp, epoch)
        train_one_epoch(
            model, 
            dl_train, 
            optimizer, 
            current_epoch=epoch, 
            total_epochs=epochs, 
            device=device, 
            wandb_run=wandb_run, 
            log_pictures=(run_validation and log_pictures))
        if run_validation:
            validate(
                model, 
                dl_valid, 
                current_epoch=epoch, 
                total_epochs=epochs, 
                device=device, 
                wandb_run=wandb_run, 
                log_pictures=log_pictures)
            ckpt_path = store_checkpoint(model, optimizer, current_epoch=epoch)
            print(f'# Stored checkpoint at {ckpt_path}')
    wandb_run.save(ckpt_path)
            

def train_one_epoch(model, dl_train, optimizer, current_epoch, total_epochs, device='cpu', wandb_run=None, log_pictures=False):
    model.train()
    # For each batch
    step = 1
    all_labels, all_mu = [], []
    epoch_loss, epoch_loss_rec, epoch_loss_kld, epoch_loss_ce = 0, 0, 0, 0
    for x_noisy, x_clean, (labels, onehot) in tqdm(dl_train, desc=f"Training epoch {current_epoch}/{total_epochs}"):
        onehot = onehot.float().to(device)
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        x_reconst, mu, logvar, z, losses = model(x_noisy, x_clean=x_clean, label=onehot)
        loss = losses[0]
        epoch_loss += loss.detach().cpu().numpy()
        epoch_loss_rec += losses[1].detach().cpu().numpy()
        epoch_loss_kld += losses[2].detach().cpu().numpy()
        epoch_loss_ce += losses[3].detach().cpu().numpy() if len(losses) == 4 else 0
        # Stop if loss is not finite
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)
        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_labels.append(labels)
        all_mu.append(mu)
        step += 1
    # Get metrics
    metrics = dict()
    dl_len = len(dl_train)
    metrics[f"train/avg_loss"] = epoch_loss / dl_len
    metrics[f"train/avg_loss_rec"] = epoch_loss_rec / dl_len
    metrics[f"train/avg_loss_kld"] = epoch_loss_kld / dl_len
    metrics[f"train/avg_loss_ce"] = epoch_loss_ce / dl_len
    if log_pictures:
        all_labels = pd.concat([pd.DataFrame(l) for l in all_labels])
        all_mu = torch.vstack(all_mu)
        latentspace_figure_path = plot_latentspace(all_mu.detach().cpu().numpy(), all_labels, current_epoch=current_epoch, n_iter=2500)
        latentspace_pca_figure_path, explvariance_pca_figure_path = plot_latentspace_pca(all_mu.detach().cpu().numpy(), all_labels, current_epoch=current_epoch)
        metrics["train/latent_space"] = wandb.Image(latentspace_figure_path)
        metrics["train/latent_space_pca"] = wandb.Image(latentspace_pca_figure_path)
        metrics["train/expl_variance_pca"] = wandb.Image(explvariance_pca_figure_path)
    print('Training metrics: \n', pd.Series(metrics))
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


def validate(model, dl_valid, current_epoch, total_epochs, device='cpu', wandb_run=None, log_pictures=False):
    model.eval()
    # For each batch
    step = 1
    all_labels, all_mu = [], []
    epoch_loss, epoch_loss_rec, epoch_loss_kld, epoch_loss_ce = 0, 0, 0, 0
    for i, (x_noisy, x_clean, labels) in enumerate(tqdm(dl_valid, desc=f"Validation epoch {current_epoch}/{total_epochs}", colour="blue")):
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        x_reconst, mu, logvar, z, losses = model(x_noisy, x_clean=x_clean, label=[])
        if i == 0:
            reconstructrions_figure_path = plot_reconstructions(x_clean, x_reconst, labels, current_epoch=current_epoch)
        loss = losses[0]
        epoch_loss += loss.detach().cpu().numpy()
        epoch_loss_rec += losses[1].detach().cpu().numpy()
        epoch_loss_kld += losses[2].detach().cpu().numpy()
        epoch_loss_ce += losses[3].detach().cpu().numpy() if len(losses) == 4 else 0
        all_labels.append(labels)
        all_mu.append(mu)
        step += 1
    # Get metrics and return them
    metrics = dict()
    dl_len = len(dl_valid)
    metrics["valid/avg_loss"] = epoch_loss / dl_len
    metrics["valid/avg_loss_rec"] = epoch_loss_rec / dl_len
    metrics["valid/avg_loss_kld"] = epoch_loss_kld / dl_len
    metrics["valid/avg_loss_ce"] = epoch_loss_ce / dl_len
    if log_pictures:
        all_labels = pd.concat([pd.DataFrame(l) for l in all_labels])
        all_mu = torch.vstack(all_mu)
        latentspace_figure_path = plot_latentspace(all_mu.detach().cpu().numpy(), all_labels, current_epoch=current_epoch)
        latentspace_pca_figure_path, explvariance_pca_figure_path = plot_latentspace_pca(all_mu.detach().cpu().numpy(), all_labels, current_epoch=current_epoch)
        metrics["valid/reconstructions"] = wandb.Image(reconstructrions_figure_path)
        metrics["valid/latent_space"] = wandb.Image(latentspace_figure_path)
        metrics["valid/latent_space_pca"] = wandb.Image(latentspace_pca_figure_path)
        metrics["valid/expl_variance_pca"] = wandb.Image(explvariance_pca_figure_path)
    print('Validation metrics: \n', pd.Series(metrics))
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


def test(model, dl_test, device='cpu', wandb_run=None, log_pictures=False):
    model.eval()
    # For each batch
    step = 1
    all_labels, all_mu = [], []
    epoch_loss, epoch_loss_rec, epoch_loss_kld, epoch_loss_ce = 0, 0, 0, 0
    for i, (x_noisy, x_clean, labels) in enumerate(tqdm(dl_test, desc=f"Testing", colour="green")):
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        x_reconst, mu, logvar, z, losses = model(x_noisy, x_clean=x_clean, label=[])
        if i == 0:
            reconstructrions_figure_path = plot_reconstructions(x_clean, x_reconst, labels, current_epoch=0)
        loss = losses[0]
        epoch_loss += loss.detach().cpu().numpy()
        epoch_loss_rec += losses[1].detach().cpu().numpy()
        epoch_loss_kld += losses[2].detach().cpu().numpy()
        epoch_loss_ce += losses[3].detach().cpu().numpy() if len(losses) == 4 else 0
        all_labels.append(labels)
        all_mu.append(mu)
        step += 1
    # Get metrics and return them
    metrics = dict()
    dl_len = len(dl_test)
    metrics["test/avg_loss"] = epoch_loss / dl_len
    metrics["test/avg_loss_rec"] = epoch_loss_rec / dl_len
    metrics["test/avg_loss_kld"] = epoch_loss_kld / dl_len
    metrics["test/avg_loss_ce"] = epoch_loss_ce / dl_len
    if log_pictures:
        all_labels = pd.concat([pd.DataFrame(l) for l in all_labels])
        all_mu = torch.vstack(all_mu)
        latentspace_figure_path = plot_latentspace(all_mu.detach().cpu().numpy(), all_labels, current_epoch=0)
        latentspace_pca_figure_path, explvariance_pca_figure_path = plot_latentspace_pca(all_mu.detach().cpu().numpy(), all_labels, current_epoch=0)
        metrics["test/reconstructions"] = wandb.Image(reconstructrions_figure_path)
        metrics["test/latent_space"] = wandb.Image(latentspace_figure_path)
        metrics["test/latent_space_pca"] = wandb.Image(latentspace_pca_figure_path)
        metrics["test/expl_variance_pca"] = wandb.Image(explvariance_pca_figure_path)
    print('Testing metrics: \n', pd.Series(metrics))
    if wandb_run is not None:
        wandb_run.log(metrics)


if __name__ == "__main__":
    parser = ArgumentParser(description='Sound Morphing VAE training script')
    parser.add_argument('-c','--configs_path', type=str, default='./configs.yaml')
    args = parser.parse_args()
    cfg = load_configs(**vars(args))
    train(cfg)
