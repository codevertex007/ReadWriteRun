import os
import yaml
import argparse
import torch
import os
import shutil
import torch.nn as nn
from utils.trainer import Trainer
from utils.dataloader import get_dataloader_and_vocab
from utils.helper import save_vocab, save_config, get_model_class, get_optimizer_class, get_lr_scheduler

def train(config):
    
    model_dir = config["model_dir"]
    embedding_dim = config["embedding_dim"]

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    train_dataloader, val_dataloader, vocab = get_dataloader_and_vocab(train_batch_size=config['train_batch_size'],
                                                                       val_batch_size=config['val_batch_size'],
                                                                       model_name=config['model_name'],
                                                                       data_dir=config['data_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size=vocab_size, embedding_dim=embedding_dim)
    print("=====MODEL=====")
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)
    
    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()

    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config file')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    train(config)