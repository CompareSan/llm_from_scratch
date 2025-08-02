import torch 
import torch.nn as nn
from cs336_basics.optimizers.adam import AdamW
from cs336_basics.layers.transformer_llm import Transformer
from cs336_basics.train import Trainer
import numpy as np



def main():   
    vocab_size = 10000
    context_len = 128
    num_layers = 6
    d_model = 768
    num_heads = 12
    d_ff = int(8/3 * d_model)
    theta = 10000.0
    batch_size = 4
    n_epochs = 10
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Transformer(vocab_size, context_len, num_layers, d_model, num_heads, d_ff, theta)
    optimizer = AdamW(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, device=device)


    dataset = np.load('./data/valid_ids.npy')  # Load your dataset here

    train_losses = trainer.train(dataset, batch_size, context_len, n_epochs)

    print("Training complete. Final training loss:", train_losses[-1])


if __name__ == "__main__":
    main()