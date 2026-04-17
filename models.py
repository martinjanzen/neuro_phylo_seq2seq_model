# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset

class RomanceDataset(Dataset):
    def __init__(self, data_path, exclude_idx=None):
        if not os.path.exists(data_path): raise FileNotFoundError(f"Dataset not found at {data_path}")
        self.data = torch.load(data_path, weights_only=False)
        self.exclude_idx = exclude_idx

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        x, y = item["X"].clone(), item["Y"]
        if self.exclude_idx is not None: x[self.exclude_idx, :, :] = 0.0
        return x.transpose(0, 1), y

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim * 2)
        self.W_k = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1)

    def forward(self, query, keys):
        attn_weights = F.softmax(self.v(torch.tanh(self.W_q(query) + self.W_k(keys))), dim=1)
        return torch.bmm(attn_weights.transpose(1, 2), keys), attn_weights

class PhonologicalLoss(nn.Module):
    def forward(self, pred, target):
        pred_feats, pred_stop = pred[..., :24], pred[..., 24:]
        target_feats, target_stop = target[..., :24], target[..., 24:25]
        valid_mask = 1.0 - target_stop
        phon = (torch.abs(pred_feats - target_feats) * valid_mask).sum() / (valid_mask.sum().clamp(min=1.0) * 24)
        return phon + F.binary_cross_entropy(pred_stop, target_stop, reduction='mean')

# Model 1: ReconstructionLSTM (Baseline)
class ReconstructionLSTM(nn.Module):
    def __init__(self, hidden_dim=64, encoder_layers=1, dropout=0.3):
        super().__init__()
        self.n_langs, self.hidden_dim, self.encoder_layers = 5, hidden_dim, encoder_layers
        self.lang_input_embed = nn.Embedding(5, 8)
        self.encoder = nn.LSTM(32, hidden_dim, encoder_layers, batch_first=True, bidirectional=True, dropout=dropout if encoder_layers > 1 else 0.0)
        self.h_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.decoder = nn.LSTM(24 + (hidden_dim * 2), hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        self.fc_out = nn.Linear(hidden_dim, 25)

    def forward(self, x):
        B, S, L, D = x.shape
        lang_active = (x.abs().sum(dim=(1, 3)) > 0).float()
        le = self.lang_input_embed(torch.arange(self.n_langs, device=x.device)).unsqueeze(0).unsqueeze(0).expand(B, S, L, -1) * lang_active.unsqueeze(1).unsqueeze(-1)
        x_flat = torch.cat([x, le], dim=-1).transpose(1, 2).reshape(B * L, S, D + 8)
        enc_out_flat, (h_flat, c_flat) = self.encoder(x_flat)

        enc_out = enc_out_flat.view(B, L, S, self.hidden_dim * 2).mean(dim=1)
        h = h_flat.view(self.encoder_layers, 2, B, L, -1).mean(dim=3)
        c = c_flat.view(self.encoder_layers, 2, B, L, -1).mean(dim=3)

        h_d = self.h_bridge(torch.cat((h[-1, 0], h[-1, 1]), -1)).unsqueeze(0)
        c_d = self.c_bridge(torch.cat((c[-1, 0], c[-1, 1]), -1)).unsqueeze(0)

        curr, preds = torch.zeros((B, 1, 24), device=x.device), []
        for _ in range(S):
            context, _ = self.attention(h_d.transpose(0, 1), enc_out)
            out, (h_d, c_d) = self.decoder(torch.cat((curr, context), dim=-1), (h_d, c_d))
            raw = self.fc_out(out)
            feats, stop_bit = torch.tanh(raw[..., :24]), torch.sigmoid(raw[..., 24:])
            preds.append(torch.cat((feats, stop_bit), dim=-1))
            curr = feats
        return torch.cat(preds, dim=1)

# Model 2: AttentionLSTM
class AttentionLSTM(nn.Module):
    def __init__(self, hidden_dim=64, encoder_layers=1, dropout=0.3):
        super().__init__()
        self.n_langs, self.hidden_dim, self.encoder_layers = 5, hidden_dim, encoder_layers
        self.lang_input_embed = nn.Embedding(5, 8)
        self.encoder = nn.LSTM(32, hidden_dim, encoder_layers, batch_first=True, bidirectional=True, dropout=dropout if encoder_layers > 1 else 0.0)
        self.h_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.decoder = nn.LSTM(24 + (hidden_dim * 2), hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        self.fc_out = nn.Linear(hidden_dim, 25)

    def forward(self, x, return_attn=False):
        B, S, L, D = x.shape
        lang_active = (x.abs().sum(dim=(1, 3)) > 0).float()
        le = self.lang_input_embed(torch.arange(self.n_langs, device=x.device)).unsqueeze(0).unsqueeze(0).expand(B, S, L, -1) * lang_active.unsqueeze(1).unsqueeze(-1)
        x_flat = torch.cat([x, le], dim=-1).transpose(1, 2).reshape(B * L, S, D + 8)
        enc_out_flat, (h_flat, c_flat) = self.encoder(x_flat)

        enc_out = enc_out_flat.view(B, L, S, self.hidden_dim * 2).reshape(B, L * S, self.hidden_dim * 2)
        h = h_flat.view(self.encoder_layers, 2, B, L, -1).mean(dim=3)
        c = c_flat.view(self.encoder_layers, 2, B, L, -1).mean(dim=3)

        h_d = self.h_bridge(torch.cat((h[-1, 0], h[-1, 1]), -1)).unsqueeze(0)
        c_d = self.c_bridge(torch.cat((c[-1, 0], c[-1, 1]), -1)).unsqueeze(0)

        curr, preds, step_attns = torch.zeros((B, 1, 24), device=x.device), [], []
        for _ in range(S):
            context, attn_w = self.attention(h_d.transpose(0, 1), enc_out)
            step_attns.append(attn_w.squeeze(-1).view(B, L, S).sum(dim=2))
            out, (h_d, c_d) = self.decoder(torch.cat((curr, context), dim=-1), (h_d, c_d))
            raw = self.fc_out(out)
            feats, stop_bit = torch.tanh(raw[..., :24]), torch.sigmoid(raw[..., 24:])
            preds.append(torch.cat((feats, stop_bit), dim=-1))
            curr = feats
            
        pred_tensor = torch.cat(preds, dim=1)
        return (pred_tensor, torch.stack(step_attns, dim=1).mean(dim=1)) if return_attn else pred_tensor