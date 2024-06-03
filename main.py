from models import LightningResNet18
import torch
import pytorch_lightning as pl
#from datasets import get_cifar10_debug_loaders as get_cifar10_loaders
#print("WARNING: USING DEBUG LOADERS!")
from datasets import get_cifar10_loaders
from pytorch_lightning.loggers import WandbLogger
import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def weight_average(module_class, state_dicts, train_loader, alphas=None):
    """
    Averages the weights of multiple state dictionaries.
    
    Args:
    - module_class: The class of the model to be averaged.
    - state_dicts: List of state dictionaries to average.
    - alphas: List of weighting factors for each state dictionary. If None, equal weighting is used.
    
    Returns:
    - A new model with averaged weights.
    """
    if alphas is None:
        alphas = [1 / len(state_dicts)] * len(state_dicts)
    else:
        assert len(alphas) == len(state_dicts), "Length of alphas must match the number of state dictionaries"

    # Initialize an empty state dictionary to store the weighted average
    averaged_state_dict = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in state_dicts[0].items()}

    # Iterate over each state dictionary and alpha
    for state_dict, alpha in zip(state_dicts, alphas):
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] += alpha * state_dict[key]

    # Create a new model instance and load the averaged state dictionary
    averaged_model = module_class()
    averaged_model.load_state_dict(averaged_state_dict)

    model.train()
    for batch in train_loader:
        images, _ = batch
        averaged_model(images)        
    
    return averaged_model


if __name__ == '__main__':
    # Load the model state dictionaries
    N_MODELS = 2
    state_dicts = []
    os.makedirs('./results/pl_logs', exist_ok=True)
    os.makedirs('./results/checkpoints', exist_ok=True)

    train_loader, val_loader, test_loader = get_cifar10_loaders()
    for i in range(N_MODELS):
        model_checkpoint_min = ModelCheckpoint(
            monitor='val_loss',
            dirpath='./results/checkpoints',
            filename=f'run_{i}_best.ckpt',
            save_top_k=1,
            mode='min',
        )
        model_checkpoint_last = ModelCheckpoint(
            dirpath='./results/checkpoints',
            filename=f'run_{i}_last.ckpt',
            save_last=True,
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
        )

        trainer = pl.Trainer(
            max_epochs=100,
            default_root_dir='./results/pl_logs',
            logger=WandbLogger(project='weight_averaging', name=f'model_{i}', log_model=False),
            callbacks=[model_checkpoint_min, model_checkpoint_last, early_stopping],
        )
        trainer.fit(LightningResNet18(), train_loader, val_loader)
        model = LightningResNet18.load_from_checkpoint(model_checkpoint_min.best_model_path)
        state_dicts.append(model.state_dict())
    
    alphas = [(i, 1-i) for i in np.linspace(0, 1, 11)]
    linear_tunnel_model_results = [trainer.validate(weight_average(LightningResNet18, state_dicts, train_loader, alpha), val_loader) for alpha in alphas]

    plt.plot([alpha[0] for alpha in alphas], [result[0]['val_loss'] for result in linear_tunnel_model_results], label='Validation Loss')
    plt.savefig('./results/linear_tunnel_loss.png')
    plt.close()
    plt.plot([alpha[0] for alpha in alphas], [result[0]['val_acc'] for result in linear_tunnel_model_results], label='Validation Accuracy')
    plt.savefig('./results/linear_tunnel_acc.png')
    plt.close()
    print('Results saved to ./results')

    


