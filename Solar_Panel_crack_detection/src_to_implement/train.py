# train.py

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import your Dataset, Model, and Trainer
from data import ChallengeDataset
from model import ResNet
from trainer import Trainer

def main():
    # 1) Read the CSV file
    #    Make sure 'data.csv' is in the correct location or provide a full path.
    df = pd.read_csv('D:\\1_Files\\Pycharm\\DL24\\exercise4_codes\\src_to_implement\\data.csv', sep=';')

    # 2) Split into train and validation sets
    #    Adjust test_size to whatever split ratio you prefer (e.g., 80/20).
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

    # 3) Create Dataset and DataLoader objects
    train_data = ChallengeDataset(train_df, mode="train")
    val_data = ChallengeDataset(val_df, mode="val")

    #    Adjust the batch size as needed.
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False,num_workers=4)

    # 4) Instantiate the ResNet model
    model = ResNet()

    # 5) Choose loss function.
    #    Since our model already includes a sigmoid, we should use BCELoss (not BCEWithLogitsLoss).
    criterion = torch.nn.BCELoss()

    # 6) Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,weight_decay=1e-3)
    

    # 7) Create the Trainer
    #    Specify the patience for early stopping and whether to use GPU.
    trainer = Trainer(
        model=model,
        crit=criterion,
        optim=optimizer,
        train_dl=train_loader,
        val_test_dl=val_loader,
        cuda=True,                    # Set to False if you don't have a GPU
        early_stopping_patience=10     # Stop if no improvement after 5 epochs
    )

    # 8) Train the model
    #    Set a max number of epochs or rely entirely on early stopping.
    train_losses, val_losses = trainer.fit(epochs=70)

    # 9) Optionally, save the final model or do more steps as needed.
    print("Training finished!")
    print("Train losses:", train_losses)
    print("Validation losses:", val_losses)


if __name__ == "__main__":
    main()
