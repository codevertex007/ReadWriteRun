
# Word2Vec from Scratch (CBOW & Skip-Gram)

This project implements the **Word2Vec** algorithm from scratch using PyTorch. Both **CBOW** and **Skip-Gram** architectures are supported.

---

## 🧠 Supported Architectures

| Model     | Objective                                      |
|-----------|-----------------------------------------------|
| Skip-Gram | Predict surrounding context words from center |
| CBOW      | Predict center word from surrounding context  |

---

## 📁 Project Structure

```

word2vec/
├── data/                             
├── notebooks/                        # Notebooks only for inference and t-SNE plots
│   ├── 01\_skipgram\_visualization.ipynb
│   └── 02\_cbow\_visualization.ipynb
├── utils/                            # Core code
│   ├── cbow\_model.py                 # CBOW implementation
│   ├── skipgram\_model.py             # Skip-Gram implementation
│   ├── dataloader.py                 # dataloader
│   ├── trainer.py                    # Training class
│   ├── helper.py                     # Save/load vocab, config, etc.
│   └── constants.py                  # Global constants
├── weights/                          # Saved models, loss logs, and vocab
│   └── cbow\_wikitext-103-v1/
│       ├── model.pth
│       ├── vocab.pt
│       ├── loss.json
│       └── config.yaml
├── config.yaml                       # Training configuration
├── train.py                          
└── README.md                         # You're here

````

---

## 🔧 How to Train

Edit `config.yaml` to select model and hyperparameters:

```yaml
model_name: cbow                     # or skipgram
data_dir: data/wikitext-103-v1
train_batch_size: 96
val_batch_size: 96
embedding_dim: 100
optimizer: Adam
learning_rate: 0.01
epochs: 10
model_dir: weights/cbow_wikitext-103-v1
````

Run training:

```bash
python train.py --config config.yaml
```

Artifacts saved in `{model_dir}`:

* `model.pth`: Trained embedding model
* `vocab.pt`: Word-to-index mapping
* `loss.json`: Training/validation loss history
* `config.yaml`: Copy of config used for run

---

## 📊 Embedding Visualization (t-SNE)

Use notebooks to visualize word embeddings in 2D using t-SNE:

* `notebooks/01_skipgram_visualization.ipynb`
* `notebooks/02_cbow_visualization.ipynb`

---


## ✅ TODO

* [x] CBOW model
* [x] Skip-Gram model
* [x] t-SNE word embedding visualization
* [x] Word analogy evaluation (e.g., king - man + woman ≈ queen)
* [ ] Negative sampling

---