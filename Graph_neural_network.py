#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:15:35 2023

@author: diego

"""
import RNA
import csv
import torch  # Import PyTorch for deep learning
from torch.utils.data import random_split  #  For splitting datasets
from torch_geometric.data import Data, Dataset  #  PyTorch Geometric for graph data handling
from torch_geometric.loader import DataLoader  # DataLoader for batching data
import pandas as pd  #  Pandas for data manipulation
from pathlib import Path  #  pathlib for handling file paths
import numpy as np  #  NumPy for numerical operations
from sklearn.preprocessing import OneHotEncoder  # 🧬 Scikit-Learn for one-hot encoding
import polars as pl  #  Polars for data manipulation
import re  #  Regular expressions for text processing
from tqdm import tqdm  #  tqdm for progress bar display
import os
#  Define loss functions for training and evaluation
import torch.nn.functional as F
from torch_geometric.nn.models import EdgeCNN
from torch.utils.tensorboard import SummaryWriter

DATA_DIR = Path("/home/diego/Documents/master/fundamentals_of_data_analytics/kaggle/RNA_GNN/")  #  Directory for dataset
TRAIN_CSV = DATA_DIR / "train_data_QUICK_START.csv"  #  Training data in CSV format
TRAIN_PARQUET_FILE = "mini_train_data.parquet"  #  Training data in Parquet format

TEST_CSV = DATA_DIR / "mini_test_sequences.csv"  #  Test sequences in CSV format
TEST_PARQUET_FILE = "mini_test_sequences.parquet"  #  Test sequences in Parquet format

PRED_CSV = "submission.csv"  #  Output file for predictions
TRAIN=1
model_path_dms='best_model_dms.pth.tar'
model_path_2a3='best_model_2a3.pth.tar'
reactivity_2a3=0
reactivity_dms=1
n_epochs = 2
batch_size=2


if(reactivity_2a3):
    model_path=model_path_2a3
else:
    model_path=model_path_dms

def to_parquet(csv_file, parquet_file):
    #  Read CSV data using Polars
    dummy_df = pl.scan_csv(csv_file)

    #  Define a new schema mapping for specific columns
    new_schema = {}
    for key, value in dummy_df.schema.items():
        if key.startswith("reactivity"):
            new_schema[key] = pl.Float32  #  Convert 'reactivity' columns to Float32
        else:
            new_schema[key] = value

    # Read CSV data with the new schema and write to Parquet
    df = pl.scan_csv(csv_file, schema=new_schema)
    
    # Write data to Parquet format with specified settings
    df.sink_parquet(
        parquet_file,
        compression='uncompressed',  # No compression for easy access
        row_group_size=10,  # Adjust row group size as needed
    )
    
    
    
to_parquet(TRAIN_CSV, TRAIN_PARQUET_FILE)  #  Training data
to_parquet(TEST_CSV, TEST_PARQUET_FILE)    #  Test data


def generate_adjancecies(structure):
    a = []
    b = []
    c = []
    d = []
    
    for i in range(2):
        a.append([])
        b.append([])
        c.append([])
        d.append([])
    
    stack = []
    
    idx = 0
    for char in structure:
        if char == '(':
            stack.append(idx)
        elif char == ')':
            topIdx = stack.pop()
            c[0].append(idx)
            c[1].append(topIdx)
            d[0].append(topIdx)
            d[1].append(idx)
        
        if idx < len(structure) - 1:
            a[0].append(idx+1)
            a[1].append(idx)
            b[0].append(idx)
            b[1].append(idx+1)
        
        idx +=1
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    return np.hstack([a,b,c,d])


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

        #  Wrap around the sequence for circular connections
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
    def __init__(self,
                 parquet_name,
                 edge_distance=5,
                 root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 train=1,
                 reactivity_2a3=1,
                 reactivity_dms=0):
        
        super().__init__(root, transform, pre_transform, pre_filter)
        #  Set the Parquet file name
        self.parquet_name = parquet_name
        #  Set the edge distance for generating the adjacency matrix
        self.edge_distance = edge_distance
        #  Initialize the one-hot encoder for node features
        self.node_encoder = OneHotEncoder(sparse=False)
        #  Fit the one-hot encoder to possible values (A, G, U, C)
        self.node_encoder.fit(np.array(['A', 'G', 'U', 'C']).reshape(-1, 1))
        #  Load the Parquet dataframe
        self.df = pl.read_parquet(self.parquet_name)
        #  Filter the dataframe by 'SN_filter' column where the value is 1.0
        if(train==1):
            self.df = self.df.filter(pl.col("SN_filter") == 1.0) #ACTIVATE TO TRAIN REAL DATASET!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if(reactivity_2a3==1):
                self.df = self.df.filter(pl.col("experiment_type") == '2A3_MaP')
            if(reactivity_dms==1):
                self.df = self.df.filter(pl.col("experiment_type") == 'DMS_MaP')
                
            self.mode=1
            #  Get reactivity column names using regular expression
            reactivity_match = re.compile('(reactivity_[0-9])')
            reactivity_names = [col for col in self.df.columns if reactivity_match.match(col)]
            #  Select only the reactivity columns
            self.reactivity_df = self.df.select(reactivity_names)
        else:
            self.idx_start=self.df.select("id_min")
            self.idx_end=self.df.select("id_max")
            self.mode=0
       
        # Select the 'sequence' column
        self.sequence_df = self.df.select("sequence")
        

    def parse_row(self, idx):
        if(self.mode==1):
            #  Read the row at the given index
            sequence_row = self.sequence_df.row(idx)
            reactivity_row = self.reactivity_df.row(idx)
            #  Get the sequence string and convert it to an array
            sequence = np.array(list(sequence_row[0])).reshape(-1, 1)
            #  Encode the sequence array using the one-hot encoder
            encoded_sequence = self.node_encoder.transform(sequence)
            #  Get the sequence length
            sequence_length = len(sequence)
            #  Get the edge index using nearest adjacency function
            #edges_np = nearest_adjacency(sequence_length, n=self.edge_distance, loops=False)
            rna_seq = RNA.fold_compound(sequence_row[0])
            structure, _ = rna_seq.mfe()
            edges_np=generate_adjancecies(structure)
            #  Convert the edge index to a torch tensor
            edge_index = torch.tensor(edges_np, dtype=torch.long)
            #  Get reactivity targets for nodes
            reactivity = np.array(reactivity_row, dtype=np.float32)[0:sequence_length]
            #  Create valid masks for nodes
            valid_mask = np.argwhere(~np.isnan(reactivity)).reshape(-1)
            torch_valid_mask = torch.tensor(valid_mask, dtype=torch.long)
            #  Replace nan values for reactivity with 0.0 (not super important as they get masked)
            reactivity = np.nan_to_num(reactivity, copy=False, nan=0.0)
            #  Define node features as the one-hot encoded sequence
            node_features = torch.Tensor(encoded_sequence)
            #  Define targets
            targets = torch.Tensor(reactivity)
            start_end_idx=np.array([(valid_mask[0],valid_mask[-1])])
            start_end_idx_torch = torch.from_numpy(start_end_idx).to(dtype=torch.int)
            x_length=torch.from_numpy(np.array([len(node_features)-1])).to(dtype=torch.int)
            #  Create a PyTorch Data object
            
            
            data = Data(x=node_features,
                        edge_index=edge_index,
                        y=targets,
                        valid_mask=torch_valid_mask,
                        valid_mask_start_end_idx=start_end_idx_torch,
                        x_length=x_length)
            return data
        else:
            #  Read the row at the given index
            sequence_row = self.sequence_df.row(idx)
            
            #  Get the sequence string and convert it to an array
            sequence = np.array(list(sequence_row[0])).reshape(-1, 1)
            #  Encode the sequence array using the one-hot encoder
            encoded_sequence = self.node_encoder.transform(sequence)
            #  Get the sequence length
            sequence_length = len(sequence)
            #  Get the edge index using nearest adjacency function
            #edges_np = nearest_adjacency(sequence_length, n=self.edge_distance, loops=False)
            rna_seq = RNA.fold_compound(sequence_row[0])
            structure, _ = rna_seq.mfe()
            edges_np=generate_adjancecies(structure)
            
            #  Convert the edge index to a torch tensor
            edge_index = torch.tensor(edges_np, dtype=torch.long)
            
            idx_start_row = self.idx_start.row(idx)
            idx_end_row = self.idx_end.row(idx)
            node_features = torch.Tensor(encoded_sequence)
            start=torch.from_numpy(np.array(idx_start_row)).to(dtype=torch.int)
            end=torch.from_numpy(np.array(idx_end_row)).to(dtype=torch.int)
            
            data = Data(x=node_features,
                        edge_index=edge_index,
                        start_idx=start,
                        end_idx=end
                        )
            return data

    def len(self):
        #  Return the length of the dataset
        return len(self.df)

    def get(self, idx):
        #  Get and parse data for the specified index
        data = self.parse_row(idx)
        return data
    
if(TRAIN==1):  
    full_train_dataset = SimpleGraphDataset(parquet_name=TRAIN_PARQUET_FILE,
                                            edge_distance=EDGE_DISTANCE,
                                            reactivity_2a3=reactivity_2a3,
                                            reactivity_dms=reactivity_dms)  
    generator1 = torch.Generator().manual_seed(42)  # 🌱 Initialize random seed generator
    
    # Assuming full_train_dataset is a valid PyTorch dataset
    train_size = int(0.7 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],generator1)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)  #  Training data loader
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)  #  Validation data loader
    
    
    

    
    def loss_fn(output, target):
        #  Clip the target values to be within the range [0, 1]
        clipped_target = torch.clip(target, min=0, max=1)
        #  Calculate the mean squared error loss
        mses = F.mse_loss(output, clipped_target, reduction='mean')
        return mses
    
    def mae_fn(output, target):
        #  Clip the target values to be within the range [0, 1]
        clipped_target = torch.clip(target, min=0, max=1)
        # 📉 Calculate the mean absolute error loss
        maes = F.l1_loss(output, clipped_target, reduction='mean')
        return maes
    
    def create_valid_mask(batch):
        
        acum=0
        valid_mask=[]
        for i in range(batch.batch_size):
            start_idx,end_idx=batch.valid_mask_start_end_idx[i][0],batch.valid_mask_start_end_idx[i][1]
            valid_idx=list(range(acum+start_idx+i, (acum+end_idx+i+1)))
            acum=acum+batch.x_length[i]
            valid_mask+=valid_idx
            
        return valid_mask
    
    
    
    #  Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the EdgeCNN model with specified parameters
    model = EdgeCNN(
        in_channels=full_train_dataset.num_features,  #  Input features determined by the dataset
        hidden_channels=128,  #  Number of hidden channels in the model
        num_layers=4,  #  Number of layers in the model
        out_channels=1  #  Number of output channels
    ).to(device)  #  Move the model to the selected device (GPU or CPU)
    
    
    
    
    
    #  Define the optimizer with learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)
    # Set up TensorBoard
    writer = SummaryWriter('logs')
    #  Iterate over epochs
    min_MAE=float('inf')
    for epoch in range(n_epochs):
        train_losses = []
        train_maes = []
        model.train()
        
        #  Iterate over batches in the training dataloader
        for batch in (pbar := tqdm(train_dataloader, position=0, leave=True)):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            out = torch.squeeze(out)
            valid_mask=create_valid_mask(batch)
    
            loss = loss_fn(out[valid_mask], batch.y[valid_mask])
            mae = mae_fn(out[valid_mask], batch.y[valid_mask])
            loss.backward()
            train_losses.append(loss.detach().cpu().numpy())
            train_maes.append(mae.detach().cpu().numpy())
            optimizer.step()
            pbar.set_description(f"Train loss {loss.detach().cpu().numpy():.4f}")
        
        # Write loss to TensorBoard
        writer.add_scalar('Loss vs epoch /train', np.mean(train_losses), epoch )
        writer.add_scalar('MAE vs epoch /train', np.mean(train_maes), epoch )
            
            
            
        #  Print average training loss and MAE for the epoch
        print(f"Epoch {epoch} train loss: ", np.mean(train_losses))
        print(f"Epoch {epoch} train mae: ", np.mean(train_maes))
        
        val_losses = []
        val_maes = []
        model.eval()
        
        #  Iterate over batches in the validation dataloader
        for batch in (pbar := tqdm(val_dataloader, position=0, leave=True)):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            out = torch.squeeze(out)
            valid_mask=create_valid_mask(batch)
            
            loss = loss_fn(out[valid_mask], batch.y[valid_mask])
            mae = mae_fn(out[valid_mask], batch.y[valid_mask])
            val_losses.append(loss.detach().cpu().numpy())
            val_maes.append(mae.detach().cpu().numpy())
            pbar.set_description(f"Validation loss {loss.detach().cpu().numpy():.4f}")
            
        writer.add_scalar('Loss vs epoch / test', np.mean(val_losses), epoch )
        writer.add_scalar('MAE vs epoch / test', np.mean(val_maes), epoch )   
            
        if(np.mean(val_maes) <= min_MAE):
            torch.save({"epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(DATA_DIR,model_path))
            min_MAE=np.mean(val_maes)
        
        #  Print average validation loss and MAE for the epoch
        print(f"Epoch {epoch} val loss: ", np.mean(val_losses))
        print(f"Epoch {epoch} val mae: ", np.mean(val_maes))
    # Close the TensorBoard writer
    writer.close()
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  Initialize the EdgeCNN model with specified parameters
    model_dms = EdgeCNN(
        in_channels=4,  #  Input features determined by the dataset
        hidden_channels=128,  #  Number of hidden channels in the model
        num_layers=4,  #  Number of layers in the model
        out_channels=1  #  Number of output channels
    ).to(device)  #  Move the model to the selected device (GPU or CPU)
    
    
    model_2a3 = EdgeCNN(
        in_channels=4,  #  Input features determined by the dataset
        hidden_channels=128,  #  Number of hidden channels in the model
        num_layers=4,  #  Number of layers in the model
        out_channels=1  #  Number of output channels
    ).to(device)  #  Move the model to the selected device (GPU or CPU)
    

    model_metadata_dms = torch.load(model_path_dms)
    model_dms.load_state_dict(model_metadata_dms['state_dict'])
    
    
    

    model_metadata_2a3 = torch.load(model_path_2a3)
    model_2a3.load_state_dict(model_metadata_2a3['state_dict'])
    
    final_test_dataset = SimpleGraphDataset(parquet_name=TEST_PARQUET_FILE, edge_distance=EDGE_DISTANCE,train=0)
    
    final_test_dataloader=DataLoader(final_test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    
    with open(PRED_CSV, 'a', newline='') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)
        
        csv_writer.writerow(['id', 'reactivity_DMS_MaP','reactivity_2A3_MaP'])
        
        for test_batch in (pbar := tqdm(final_test_dataloader, position=0, leave=True)):
            test_batch = test_batch.to(device)
            reactivity_prediction_dms = model_dms(test_batch.x, test_batch.edge_index)
            reactivity_prediction_dms = torch.squeeze(reactivity_prediction_dms)
            
            reactivity_prediction_2a3 = model_2a3(test_batch.x, test_batch.edge_index)
            reactivity_prediction_2a3 = torch.squeeze(reactivity_prediction_2a3)
            
            
            
            
            reactivity_predictions_dms=reactivity_prediction_dms.tolist()
            reactivity_predictions_2a3=reactivity_prediction_2a3.tolist()
            
            ids=[]
            for i in range(test_batch.batch_size):
                ids=ids+list(range(test_batch.start_idx[i],test_batch.end_idx[i]+1))
        
        
            rows = zip(ids, reactivity_predictions_dms,reactivity_predictions_2a3)

            # Write the rows to the CSV file
            csv_writer.writerows(rows)
            

        
   
