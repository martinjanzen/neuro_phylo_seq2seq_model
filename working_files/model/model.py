"""
08_model.py
FIXED: Reduces the decoder to 1 layer to prevent redundant initialization 
and overfitting on low-resource data (Issue 15). Also includes Sigmoid 
trust scaling (Issue 8A) and relaxed PhonologicalLoss ratios (Issue 9).
"""
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import Dataset

class RomanceDataset(Dataset):
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x = self.data[idx]["X"].transpose(0, 1) 
        y = self.data[idx]["Y"] 
        return x, y

class NeuroPhyloLSTM(nn.Module):
    def __init__(self, hidden_dim=256, encoder_layers=2, dropout=0.3):
        super(NeuroPhyloLSTM, self).__init__()
        self.feat_dim = 24

        path = "model/phylo_weights.pt"
        if os.path.exists(path):
            initial_w = torch.load(path)
        else:
            initial_w = torch.ones((5, 24))
            
        self.register_buffer('phylo_prior', initial_w)
        self.phylo_weights = nn.Parameter(initial_w.clone())

        # Encoder remains deep to process the complex 5-language input
        self.encoder = nn.LSTM(120, hidden_dim, encoder_layers, 
                               batch_first=True, bidirectional=True, dropout=dropout)
                               
        self.h_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # CRITICAL FIX 15: Restrict decoder to 1 layer to match initialization math.
        # Dropout must be 0.0 for a 1-layer LSTM to avoid PyTorch runtime warnings.
        self.decoder = nn.LSTM(24, hidden_dim, num_layers=1, 
                               batch_first=True, dropout=0.0)
                               
        self.fc_out = nn.Linear(hidden_dim, 24)

    def forward(self, x, use_phylo=True):
        B, S, L, D = x.shape
        
        if use_phylo:
            w_normalized = torch.sigmoid(self.phylo_weights) * 2.0
            w_gate = w_normalized.view(1, 1, L, D)
            x = x * w_gate
        
        x = x.reshape(B, S, -1)
        _, (h, c) = self.encoder(x)
        
        # CRITICAL FIX 15: Removed .repeat(). 
        # The bridge inherently outputs (1, Batch, Hidden), perfectly sizing the 1-layer decoder.
        h_d = self.h_bridge(torch.cat((h[-2], h[-1]), -1)).unsqueeze(0)
        c_d = self.c_bridge(torch.cat((c[-2], c[-1]), -1)).unsqueeze(0)
        
        curr = torch.zeros((B, 1, 24)).to(x.device)
        preds = []
        for _ in range(S):
            out, (h_d, c_d) = self.decoder(curr, (h_d, c_d))
            step_pred = torch.tanh(self.fc_out(out))
            preds.append(step_pred)
            curr = step_pred
            
        return torch.cat(preds, dim=1)

class PhonologicalLoss(nn.Module):
    def __init__(self, mode='baseline'):
        super().__init__()
        w = torch.ones(24)

        if mode in ['phylo', 'constrained']:
            w[0:5]   = 3.0   # Skeleton 
            w[5:14]  = 2.0   # Articulation 
            w[14:24] = 1.0   # Color 
            
        self.register_buffer('weights', w)

    def forward(self, pred, target):
        mask = (target.abs().sum(-1, keepdim=True) > 0).float()
        sq_error = (pred - target)**2
        weighted_error = sq_error * self.weights
        return (weighted_error * mask).mean()