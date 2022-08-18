# Sound Morphing with VAEs
> Project for "Deep Learning for Music Information Retrieval @ CCRMA" 2022

## Setup


## Usage
```sh
# train
python src/train.py
```


## TODO
- merge code
- retrain on 4 final architectures:
  - (transpose+pixshuffle, leakyrelu) x (vae_loss, vae_loss_classes)
- create inference script
  - generate audio from random points in latent space
  - generate N samples between two sample (encode samples -> interpolate z -> decode z)
- presentation
  - intro (problem statement, background on VAE, previous works)
  - method (training framework, model architecture, dataset, preprocessing/augmentation)
  - results (latent space: pca/tsne, reconstructed examples gif, audio samples)
  - future work
  - conclusions