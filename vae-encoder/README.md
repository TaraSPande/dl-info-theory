# vae-encoder

Tabular (float vector) **beta-VAE** designed for:

1. Train VAE (encoder+decoder) unsupervised.
2. Discard the decoder.
3. Attach a downstream head to the encoder for **regression** or **classification**.

## Files

- `models.py`
  - `TabularVAE`, `TabularVAEEncoder`, `TabularVAEDecoder`
  - `vae_loss` (reconstruction + KL)
  - `RegressionHead`, `ClassificationHead`, `EncoderWithHead`
- `trainer.py`
  - `train_vae`
  - `train_regression_head`, `train_classification_head`
- `demo.py`: end-to-end synthetic sanity check

## Run demo

```bash
python vae-encoder/demo.py
```

## Typical usage

```python
from models import TabularVAE, VAEConfig, RegressionHead, EncoderWithHead

vae = TabularVAE(VAEConfig(input_dim=128, latent_dim=16))
# ... train vae ...

encoder = vae.encoder  # keep this
head = RegressionHead(latent_dim=encoder.cfg.latent_dim, num_targets=1)
model = EncoderWithHead(encoder, head, use_mean=True)
pred = model(x)["out"]
```
