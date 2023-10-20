#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:15:35 2023

@author: diego

"""

import torch  # 🔥 Import PyTorch for deep learning
from torch.utils.data import random_split  # 📂 For splitting datasets
from torch_geometric.data import Data, Dataset  # 📊 PyTorch Geometric for graph data handling
from torch_geometric.loader import DataLoader  # 🚚 DataLoader for batching data
import pandas as pd  # 🐼 Pandas for data manipulation
from pathlib import Path  # 📁 pathlib for handling file paths
import numpy as np  # 🧮 NumPy for numerical operations
from sklearn.preprocessing import OneHotEncoder  # 🧬 Scikit-Learn for one-hot encoding
import polars as pl  # 📊 Polars for data manipulation
import re  # 🧵 Regular expressions for text processing
from tqdm import tqdm  # 🔄 tqdm for progress bar display


DATA_DIR = Path("/home/maqiao/Data/kaggle/stanford-ribonanza-rna-folding")  # 📂 Directory for dataset
TRAIN_CSV = DATA_DIR / "train_data.csv"  # 🚆 Training data in CSV format
TRAIN_PARQUET_FILE = "train_data.parquet"  # 📦 Training data in Parquet format
TEST_CSV = DATA_DIR / "test_sequences.csv"  # 🚀 Test sequences in CSV format
TEST_PARQUET_FILE = "test_sequences.parquet"  # 📦 Test sequences in Parquet format
PRED_CSV = "submission.csv"  # 📄 Output file for predictions

def to_parquet(csv_file, parquet_file):
    # 📊 Read CSV data using Polars
    dummy_df = pl.scan_csv(csv_file)

    # 🔍 Define a new schema mapping for specific columns
    new_schema = {}
    for key, value in dummy_df.schema.items():
        if key.startswith("reactivity"):
            new_schema[key] = pl.Float32  # 📊 Convert 'reactivity' columns to Float32
        else:
            new_schema[key] = value

    # 📊 Read CSV data with the new schema and write to Parquet
    df = pl.scan_csv(csv_file, schema=new_schema)
    
    # 💾 Write data to Parquet format with specified settings
    df.sink_parquet(
        parquet_file,
        compression='uncompressed',  # No compression for easy access
        row_group_size=10,  # Adjust row group size as needed
    )
    
    
    
to_parquet(TRAIN_CSV, TRAIN_PARQUET_FILE)  # 🚆 Training data
to_parquet(TEST_CSV, TEST_PARQUET_FILE)    # 🚀 Test data



def nearest_adjacency(sequence_length, n=2, loops=True):
    base = np.arange(sequence_length)
    connections = []

    for i in range(-n, n + 1):
        if i == 0 and not loops:
            continue
        elif i == 0 and loops:
            stack = np.vstack([base, base])
            connections.append(stack)
            continue

        # 🔄 Wrap around the sequence for circular connections
        neighbours = base.take(range(i, sequence_length + i), mode='wrap')
        stack = np.vstack([base, neighbours])

        # Separate connections for positive and negative offsets
        if i < 0:
            connections.append(stack[:, -i:])
        elif i > 0:
            connections.append(stack[:, :-i])

    # Combine connections horizontally
    return np.hstack(connections)


EDGE_DISTANCE = 4 #Edge distance for generating adjacency matrix.


class SimpleGraphDataset(Dataset):
    def __init__(self, parquet_name, edge_distance=5, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # 📄 Set the Parquet file name
        self.parquet_name = parquet_name
        # 📏 Set the edge distance for generating the adjacency matrix
        self.edge_distance = edge_distance
        # 🧮 Initialize the one-hot encoder for node features
        self.node_encoder = OneHotEncoder(sparse=False)
        # 🧮 Fit the one-hot encoder to possible values (A, G, U, C)
        self.node_encoder.fit(np.array(['A', 'G', 'U', 'C']).reshape(-1, 1))
        # 📊 Load the Parquet dataframe
        self.df = pl.read_parquet(self.parquet_name)
        # 📊 Filter the dataframe by 'SN_filter' column where the value is 1.0
        self.df = self.df.filter(pl.col("SN_filter") == 1.0)
        # 🧬 Get reactivity column names using regular expression
        reactivity_match = re.compile('(reactivity_[0-9])')
        reactivity_names = [col for col in self.df.columns if reactivity_match.match(col)]
        # 📊 Select only the reactivity columns
        self.reactivity_df = self.df.select(reactivity_names)
        # 📊 Select the 'sequence' column
        self.sequence_df = self.df.select("sequence")

    def parse_row(self, idx):
        # 📊 Read the row at the given index
        sequence_row = self.sequence_df.row(idx)
        reactivity_row = self.reactivity_df.row(idx)
        # 🧬 Get the sequence string and convert it to an array
        sequence = np.array(list(sequence_row[0])).reshape(-1, 1)
        # 🧬 Encode the sequence array using the one-hot encoder
        encoded_sequence = self.node_encoder.transform(sequence)
        # 📏 Get the sequence length
        sequence_length = len(sequence)
        # 📊 Get the edge index using nearest adjacency function
        edges_np = nearest_adjacency(sequence_length, n=self.edge_distance, loops=False)
        # 📏 Convert the edge index to a torch tensor
        edge_index = torch.tensor(edges_np, dtype=torch.long)
        # 🧬 Get reactivity targets for nodes
        reactivity = np.array(reactivity_row, dtype=np.float32)[0:sequence_length]
        # 🔒 Create valid masks for nodes
        valid_mask = np.argwhere(~np.isnan(reactivity)).reshape(-1)
        torch_valid_mask = torch.tensor(valid_mask, dtype=torch.long)
        # 🧬 Replace nan values for reactivity with 0.0 (not super important as they get masked)
        reactivity = np.nan_to_num(reactivity, copy=False, nan=0.0)
        # 📊 Define node features as the one-hot encoded sequence
        node_features = torch.Tensor(encoded_sequence)
        # 🎯 Define targets
        targets = torch.Tensor(reactivity)
        # 📊 Create a PyTorch Data object
        data = Data(x=node_features, edge_index=edge_index, y=targets, valid_mask=torch_valid_mask)
        return data

    def len(self):
        # 📏 Return the length of the dataset
        return len(self.df)

    def get(self, idx):
        # 📊 Get and parse data for the specified index
        data = self.parse_row(idx)
        return data
    
    
full_train_dataset = SimpleGraphDataset(parquet_name=TRAIN_PARQUET_FILE, edge_distance=EDGE_DISTANCE)  # 🚆 Full training dataset
generator1 = torch.Generator().manual_seed(42)  # 🌱 Initialize random seed generator

# Assuming full_train_dataset is a valid PyTorch dataset
train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],generator1)


train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=8)  # 📦 Training data loader
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=8)  # 📦 Validation data loader



# 📉 Define loss functions for training and evaluation
import torch.nn.functional as F

def loss_fn(output, target):
    # 🪟 Clip the target values to be within the range [0, 1]
    clipped_target = torch.clip(target, min=0, max=1)
    # 📉 Calculate the mean squared error loss
    mses = F.mse_loss(output, clipped_target, reduction='mean')
    return mses

def mae_fn(output, target):
    # 🪟 Clip the target values to be within the range [0, 1]
    clipped_target = torch.clip(target, min=0, max=1)
    # 📉 Calculate the mean absolute error loss
    maes = F.l1_loss(output, clipped_target, reduction='mean')
    return maes



from torch_geometric.nn.models import EdgeCNN

# 🛠️ Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 🏗️ Initialize the EdgeCNN model with specified parameters
model = EdgeCNN(
    in_channels=full_train_dataset.num_features,  # 📊 Input features determined by the dataset
    hidden_channels=128,  # 🕳️ Number of hidden channels in the model
    num_layers=4,  # 🧱 Number of layers in the model
    out_channels=1  # 📤 Number of output channels
).to(device)  # 🏗️ Move the model to the selected device (GPU or CPU)



n_epochs = 15

# 📈 Define the optimizer with learning rate and weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)

# 🚂 Iterate over epochs
for epoch in range(n_epochs):
    train_losses = []
    train_maes = []
    model.train()
    
    # 🚞 Iterate over batches in the training dataloader
    for batch in (pbar := tqdm(train_dataloader, position=0, leave=True)):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        out = torch.squeeze(out)
        loss = loss_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
        mae = mae_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
        loss.backward()
        train_losses.append(loss.detach().cpu().numpy())
        train_maes.append(mae.detach().cpu().numpy())
        optimizer.step()
        pbar.set_description(f"Train loss {loss.detach().cpu().numpy():.4f}")
    
    # 📊 Print average training loss and MAE for the epoch
    print(f"Epoch {epoch} train loss: ", np.mean(train_losses))
    print(f"Epoch {epoch} train mae: ", np.mean(train_maes))
    
    val_losses = []
    val_maes = []
    model.eval()
    
    # 🚞 Iterate over batches in the validation dataloader
    for batch in (pbar := tqdm(val_dataloader, position=0, leave=True)):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        out = torch.squeeze(out)
        loss = loss_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
        mae = mae_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
        val_losses.append(loss.detach().cpu().numpy())
        val_maes.append(mae.detach().cpu().numpy())
        pbar.set_description(f"Validation loss {loss.detach().cpu().numpy():.4f}")
    
    # 📊 Print average validation loss and MAE for the epoch
    print(f"Epoch {epoch} val loss: ", np.mean(val_losses))
    print(f"Epoch {epoch} val mae: ", np.mean(val_maes))
