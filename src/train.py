import sys
import math
import wandb
import torch
import pandas as pd
from tqdm import tqdm
from utils import Bunch, set_seed, get_preprocessing, get_dataloader, chart_dependencies, get_now
from dataset import NsynthDataset
from model import VaeModel
from loss import VaeLoss


def train(cfg):
    print("# Sound Morphing VAE training script")
    print("# Configutation: ", cfg.__dict__)
    set_seed(cfg.seed)
    if cfg.num_workers > 0:
        torch.set_num_threads(cfg.num_workers)

    print('# Creating preprocessing, datasets, dataloaders')
    preproc_train = get_preprocessing(n_mels=cfg.n_mels, train=True)
    preproc_valid = get_preprocessing(n_mels=cfg.n_mels, train=False)
    ds_train = NsynthDataset('/home/rmiccini/stanford_mir2/data/nsynth-train', sr=cfg.sr, duration=cfg.duration, pitches=[60], transform=preproc_train)
    ds_valid = NsynthDataset('/home/rmiccini/stanford_mir2/data/nsynth-valid', sr=cfg.sr, duration=cfg.duration, pitches=[60], transform=preproc_valid)
    dl_train = get_dataloader(ds_train, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, shuffle=True)
    dl_valid = get_dataloader(ds_valid, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, shuffle=False)
    *_, n_timeframes = ds_train[0][0].shape

    print('# Creating loss, model, optimizer')
    loss = VaeLoss(rec_weight=cfg.rec_weight, kld_weight=cfg.kld_weight)
    model = VaeModel(loss=loss)
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

    print('# Starting training loop')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        x_reconst, losses = model(x, label=label)
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
    epoch_loss, epoch_loss_rec, epoch_loss_kld, epoch_loss_ce = 0, 0, 0, 0
    for x, label in tqdm(dl_valid, desc=f"Validation epoch {current_epoch}/{total_epochs}", colour="blue"):
        label = label.to(device)
        x = x.to(device)
        x_reconst, losses = model(x, label=label)
        loss = losses[0]
        epoch_loss += loss.detach().cpu().numpy()
        epoch_loss_rec += losses[1].detach().cpu().numpy()
        epoch_loss_kld += losses[2].detach().cpu().numpy()
        epoch_loss_ce += losses[3].detach().cpu().numpy() if len(losses) == 4 else 0
        step += 1
    # Get metrics and return them
    metrics = dict()
    dl_len = len(dl_valid)
    metrics[f"valid/avg_loss"] = epoch_loss / dl_len
    metrics[f"valid/avg_loss_rec"] = epoch_loss_rec / dl_len
    metrics[f"valid/avg_loss_kld"] = epoch_loss_kld / dl_len
    metrics[f"valid/avg_loss_ce"] = epoch_loss_ce / dl_len
    print('Validation metrics: \n', pd.Series(metrics))
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


def store_checkpoint(model, optimizer, current_epoch):
    ckpt_path = f'checkpoints/epoch_{current_epoch:04}.pth'
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)


if __name__ == "__main__":
    cfg = Bunch({
        # general
        'seed': 42,
        # data
        'sr': 16000,
        'duration': 4,
        'batch_size': 8,
        'num_workers': 0,
        'n_mels': 80,
        # training
        'start_lr': 1e-4,
        'epochs': 50,
        'val_every': 10,
        # model
        'lspace_size': 64,
        # loss
        'rec_weight': 1.0,
        'kld_weight': 1.0,
        'ce_weigth': 1.0,
    })
    train(cfg)