import sys
import math
import wandb
import torch
import pandas as pd

from tqdm import tqdm
from utils import (Bunch, set_seed, get_preprocessing, get_dataloader, chart_dependencies, 
        get_now, store_checkpoint, plot_reconstructions, plot_latentspace)
from dataset import NsynthDataset
from model import VaeModel
from loss import VaeLoss


def train(cfg):
    print("# Sound Morphing VAE training script")
    print("# Configutation: ", cfg.__dict__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(cfg.seed)
    if cfg.num_workers > 0:
        torch.set_num_threads(cfg.num_workers)

    print('# Creating preprocessing, datasets, dataloaders')
    preproc_train = get_preprocessing(n_mels=cfg.n_mels, data_mean=cfg.data_mean, data_std=cfg.data_std, log_transform=True, train=True)
    preproc_valid = get_preprocessing(n_mels=cfg.n_mels, data_mean=cfg.data_mean, data_std=cfg.data_std, log_transform=True, train=False)
    ds_train = NsynthDataset('/home/rmiccini/stanford_mir2/data/nsynth-train', sr=cfg.sr, duration=cfg.duration, pitches=[60], transform=preproc_train, label='onehot')
    ds_valid = NsynthDataset('/home/rmiccini/stanford_mir2/data/nsynth-valid', sr=cfg.sr, duration=cfg.duration, pitches=[60], transform=preproc_valid, label='full')
    dl_train = get_dataloader(ds_train, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, shuffle=True)
    dl_valid = get_dataloader(ds_valid, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, shuffle=False)
    *_, n_timeframes = ds_train[0][0].shape
    n_classes = ds_train.get_n_classes()
    print(preproc_train)
    print(f'Training: {len(ds_train)} datapoints, {len(dl_train)} batches, {n_classes} classes')
    print(f'Validation: {len(ds_valid)} datapoints, {len(dl_valid)} batches')

    print('# Creating loss, model, optimizer')
    loss = VaeLoss(rec_weight=cfg.rec_weight, kld_weight=cfg.kld_weight)
    model = VaeModel(loss=loss, fc_hidden1=cfg.fc_hidden1, fc_hidden2=cfg.fc_hidden2, fc_hidden3=cfg.fc_hidden3, lspace_size=cfg.lspace_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.start_lr)
    chart_dependencies(model, input_shape=(cfg.batch_size, 1, cfg.n_mels, n_timeframes))
    print(model.get_infos())

    run_name = get_now()
    wandb_run = wandb.init(
        name=run_name,
        project='sound_morphing',
        entity='miccio',
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
        val_every=cfg.val_every,
        device=device,
        wandb_run=wandb_run,
    )

    print('# Starting testing')
    ds_test = NsynthDataset('/home/rmiccini/stanford_mir2/data/nsynth-test', sr=cfg.sr, duration=cfg.duration, pitches=[60], transform=preproc_valid, label='full')
    dl_test = get_dataloader(ds_test, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, shuffle=False)
    print(f'Testing: {len(ds_test)} datapoints, {len(dl_test)} batches')
    test(
        model=model,
        dl_test=dl_test,
        device=device,
        wandb_run=wandb_run,
    )
    wandb_run.finish()


def training_loop(epochs, model, optimizer, dl_train, dl_valid, val_every, device='cpu', wandb_run=None):
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, dl_train, optimizer, current_epoch=epoch, total_epochs=epochs, device=device, wandb_run=wandb_run)
        if epoch % val_every == 0 or epoch == 1 or epoch == epochs:
            validate(model, dl_valid, current_epoch=epoch, total_epochs=epochs, device=device, wandb_run=wandb_run)
            store_checkpoint(model, optimizer, current_epoch=epoch)
            

def train_one_epoch(model, dl_train, optimizer, current_epoch, total_epochs, device='cpu', wandb_run=None):
    model.train()
    # For each batch
    step = 1
    epoch_loss, epoch_loss_rec, epoch_loss_kld, epoch_loss_ce = 0, 0, 0, 0
    for x, label in tqdm(dl_train, desc=f"Training epoch {current_epoch}/{total_epochs}"):
        label = label.to(device)
        x = x.to(device)
        x_reconst, mu, logvar, z, losses = model(x, label=label)
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
        step += 1
    # Get metrics
    metrics = dict()
    dl_len = len(dl_train)
    metrics[f"train/avg_loss"] = epoch_loss / dl_len
    metrics[f"train/avg_loss_rec"] = epoch_loss_rec / dl_len
    metrics[f"train/avg_loss_kld"] = epoch_loss_kld / dl_len
    metrics[f"train/avg_loss_ce"] = epoch_loss_ce / dl_len
    print('Training metrics: \n', pd.Series(metrics))
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


def validate(model, dl_valid, current_epoch, total_epochs, device='cpu', wandb_run=None):
    model.eval()
    # For each batch
    step = 1
    all_labels, all_mu = [], []
    epoch_loss, epoch_loss_rec, epoch_loss_kld, epoch_loss_ce = 0, 0, 0, 0
    for i, (x, labels) in enumerate(tqdm(dl_valid, desc=f"Validation epoch {current_epoch}/{total_epochs}", colour="blue")):
        x = x.to(device)
        x_reconst, mu, logvar, z, losses = model(x, label=[])
        if i == 0:
            reconstructrions_figure_path = plot_reconstructions(x, x_reconst, current_epoch=current_epoch)
        loss = losses[0]
        epoch_loss += loss.detach().cpu().numpy()
        epoch_loss_rec += losses[1].detach().cpu().numpy()
        epoch_loss_kld += losses[2].detach().cpu().numpy()
        epoch_loss_ce += losses[3].detach().cpu().numpy() if len(losses) == 4 else 0
        all_labels.append(labels)
        all_mu.append(mu)
        step += 1
    all_labels = pd.concat([pd.DataFrame(l) for l in all_labels])
    all_mu = torch.vstack(all_mu)
    latentspace_figure_path = plot_latentspace(all_mu.detach().cpu().numpy(), all_labels, current_epoch=current_epoch)
    # Get metrics and return them
    metrics = dict()
    dl_len = len(dl_valid)
    metrics["valid/avg_loss"] = epoch_loss / dl_len
    metrics["valid/avg_loss_rec"] = epoch_loss_rec / dl_len
    metrics["valid/avg_loss_kld"] = epoch_loss_kld / dl_len
    metrics["valid/avg_loss_ce"] = epoch_loss_ce / dl_len
    metrics["valid/reconstructions"] = wandb.Image(reconstructrions_figure_path)
    metrics["valid/latent_space"] = wandb.Image(latentspace_figure_path)
    print('Validation metrics: \n', pd.Series(metrics))
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


def test(model, dl_test, device='cpu', wandb_run=None):
    model.eval()
    # For each batch
    step = 1
    all_labels, all_mu = [], []
    epoch_loss, epoch_loss_rec, epoch_loss_kld, epoch_loss_ce = 0, 0, 0, 0
    for i, (x, labels) in enumerate(tqdm(dl_test, desc=f"Testing", colour="green")):
        x = x.to(device)
        x_reconst, mu, logvar, z, losses = model(x, label=[])
        if i == 0:
            reconstructrions_figure_path = plot_reconstructions(x, x_reconst, current_epoch=0)
        loss = losses[0]
        epoch_loss += loss.detach().cpu().numpy()
        epoch_loss_rec += losses[1].detach().cpu().numpy()
        epoch_loss_kld += losses[2].detach().cpu().numpy()
        epoch_loss_ce += losses[3].detach().cpu().numpy() if len(losses) == 4 else 0
        all_labels.append(labels)
        all_mu.append(mu)
        step += 1
    all_labels = pd.concat([pd.DataFrame(l) for l in all_labels])
    all_mu = torch.vstack(all_mu)
    latentspace_figure_path = plot_latentspace(all_mu.detach().cpu().numpy(), all_labels, current_epoch=0)
    # Get metrics and return them
    metrics = dict()
    dl_len = len(dl_test)
    metrics["test/avg_loss"] = epoch_loss / dl_len
    metrics["test/avg_loss_rec"] = epoch_loss_rec / dl_len
    metrics["test/avg_loss_kld"] = epoch_loss_kld / dl_len
    metrics["test/avg_loss_ce"] = epoch_loss_ce / dl_len
    metrics["test/reconstructions"] = wandb.Image(reconstructrions_figure_path)
    metrics["test/latent_space"] = wandb.Image(latentspace_figure_path)
    print('Testing metrics: \n', pd.Series(metrics))
    if wandb_run is not None:
        wandb_run.log(metrics)



if __name__ == "__main__":
    cfg = Bunch({
        # general
        'seed': 42,
        # data
        'sr': 16000,
        'duration': 4,
        'batch_size': 32,
        'num_workers': 4,
        'n_mels': 80,
        'data_mean': -41.5759,  # in db
        'data_std': 38.5646,  # in db
        # training
        'start_lr': 1e-4,
        'epochs': 4,
        'val_every': 2,
        # model
        'fc_hidden1': 512,
        'fc_hidden2': 1024,
        'fc_hidden3': 4096,
        'lspace_size': 256,
        # loss
        'rec_weight': 1.0,
        'kld_weight': 0.01,
        'ce_weigth': 1.0,
    })
    train(cfg)