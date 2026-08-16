# Hybrid Agent Run

**Task:** Generate and reconstruct MNIST digits with a variational autoencoder

## Plan
- Intents: vae, mnist, generation, reconstruction
- Domains: mnist, digit, binarized
- Queries: Generate and reconstruct MNIST digits with a variational autoencoder, vae, variational autoencoder, mnist vae

## Literature (8)
- (2021) VDSM: Unsupervised Video Disentanglement with State-Space Modeling and Deep Mixtures of Experts. Vowels, Camgoz, Bowden — https://arxiv.org/pdf/2103.07292.pdf [score=5.077]
- (2020) Unsupervised representation disentanglement using cross domain features and adversarial learning in variational autoencoder based voice conversion.	Huang, Luo, Hwang, Lo, Peng, Tsao, Wang	 — https://arxiv.org/pdf/2001.07849.pdf [score=4.772]
- (2021) Semi-Supervised Disentanglement of Class-Related and Class-Independent Factors in VAE. Hajimiri, Lotfi, Baghshah — https://arxiv.org/pdf/2102.00892.pdf [score=4.427]
- (2020) S3VAE: self-supervised sequential VAE for representation disentanglement and data generation.	Zhu, Min, Kadav, Graf,	 — https://arxiv.org/pdf/2005.11437.pdf [score=4.377]
- (2019) Variational learning with disentanglement-pytorch.	Abdi, Abolmaesumi, Fels	 — https://openreview.net/pdf?id=rJgUsFYnir [score=4.327]

## Datasets (6)
- `ylecun/mnist` score=10.151 — domain_hits=2, query_overlap=1, downloads=80000 [catalog]
- `zalando-datasets/fashion_mnist` score=7.308 — domain_hits=1, query_overlap=1, downloads=30000 [catalog]
- `Temporarium/HDR_Photos_VAE_Training_DNG` score=3.806 — query_overlap=1, downloads=1713 [hub]
- `yuanchenyang/imagenet-256-flux2-vae-latents` score=3.724 — query_overlap=1, downloads=1353 [hub]
- `LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public` score=3.608 — query_overlap=1, downloads=971 [hub]
- `grspo/imagenet-latents-flux-vae-mjhq512bs128fix-e2e-lr2e-5-400k` score=3.554 — query_overlap=1, downloads=832 [hub]

## Models (6)
- `pszmk/mnist-vae-latent2` score=14.749 loadable=True — known AutoModel + safetensors MNIST VAE (CPU-safe); domain_hits=1, vae_match, loadability=4.5, downloads=8
- `uday9k/Gaussian_MNIST_VAE` score=12.799 loadable=True — custom MNIST VAE weights on Hub; domain_hits=1, vae_match, loadability=4.5, downloads=10
- `uday9k/Binarized_MNIST_VAE` score=10.593 loadable=False — domain_hits=2, vae_match, loadability=1.5, downloads=15 [hub]
- `karthik-2905/VariationalAutoencoders` score=9.7 loadable=False — domain_hits=1, vae_match, loadability=1.5 [hub]
- `jolespin/binary-vae-mnist` score=8.148 loadable=False — domain_hits=1, vae_match, loadability=1.5, downloads=5 [hub]
- `madebyollin/sdxl-vae-fp16-fix` score=7.651 loadable=False — vae_match, loadability=1.5, downloads=297423 [hub]

## Inference
- Model: `local:TinyVAE(trained_on=ylecun/mnist)`
- Dataset: `ylecun/mnist`
- Device: `cpu`
- Metrics: {'recon_mse': 0.025689, 'train_final_loss': 138.2879}
- Artifacts:
  - `runs/mnist-vae-demo/inference/reconstructions.png`
  - `runs/mnist-vae-demo/inference/samples.png`
  - `runs/mnist-vae-demo/inference/latent_means.json`
  - `runs/mnist-vae-demo/inference/tiny_vae.pt`
- Notes: Resolved dataset ylecun/mnist; Loaded Hub model `pszmk/mnist-vae-latent2` via AutoModel; Detected collapsed Hub latents; switching to local TinyVAE training; Trained TinyVAE epochs=5 final_loss=138.2879 samples=60000; Saved checkpoint runs/mnist-vae-demo/inference/tiny_vae.pt
