# Urdu/Arabic - Text Recognition AI Tool

ORATOR is a deep learning–based neural network OCR tool built specifically for recognizing **Urdu & Arabic** scripts.
It is built using a modern architecture: **Densnet** for feature extraction, **GRU** for sequence modeling, and **Attn** for transcription.
Also include architecture: **Resnet** for feature extraction, **LSTM** for sequence modeling, and **CTC** for transcription.
---

## Requirements

- Python 3.11.5
- [`uv`](https://github.com/astral-sh/uv) for dependency and environment management
- PyTorch, torchvision, lmdb, numpy, and related libraries


Expected structure:

```
UA-TRAIT/
└── dataset/
    ├── train/
    │   ├── test/
    │   └── gt.txt
    └── test/
        ├── test/
        └── gt.txt
```

---
### Install Dependencies


```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync
```
## Creating LMDB Datasets

### For Training

```bash
uv run create_lmdb_dataset.py \
  --inputPath dataset/train/ \
  --gtFile dataset/train/gt.txt \
  --outputPath ./lmdb_train
```

### For Testing

```bash
uv run create_lmdb_dataset.py \
  --inputPath dataset/test/ \
  --gtFile dataset/test/gt.txt \
  --outputPath ./lmdb_test
```

---

## Training the Model

```bash
uv run train.py \
  --train_data ./lmdb_train \
  --valid_data ./lmdb_test \
  --FeatureExtraction Densnet \
  --SequenceModeling GRU \
  --Prediction Attn \
  --exp_name UA-TRAIT \
  --num_epochs 100 \
  --batch_size 8 

```

---




## References

* Python EOL: [https://devguide.python.org/versions/](https://devguide.python.org/versions/)
* HRNet Paper: [https://arxiv.org/abs/1904.04514](https://arxiv.org/abs/1904.04514)
* CTC Explanation: [https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)
* UTRNet Repo: [https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git)
* uv Packaging Tool: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

---