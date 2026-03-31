import torch
import torch.nn as nn
import os

class NeuroPhyloLSTM(nn.Module):
    def __init__(self, input_langs=5, feat_dim=24, hidden_dim=256, num_layers=2, dropout=0.3, phylo_path="model/phylo_weights.pt"):
        super(NeuroPhyloLSTM, self).__init__()
        self.feat_dim = feat_dim
        self.input_dim = input_langs * feat_dim
        
        # AIM 2.1: Transition from 'Static Buffer' to 'Learnable Parameter'
        # We initialize with Strategy A (Literature-backed weights)
        if os.path.exists(phylo_path):
            initial_weights = torch.load(phylo_path)
        else:
            # Fallback to uniform trust if file is missing
            initial_weights = torch.ones((input_langs, feat_dim))
            
        # Registering as Parameter allows the optimizer to 'nudge' trust scores
        self.phylo_weights = nn.Parameter(initial_weights)

        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.h_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.LSTM(input_size=feat_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, feat_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, target_len=None, use_phylo=False):
        batch_size, seq_len, num_langs, feat_dim = x.shape
        if target_len is None: target_len = seq_len
        
        # Learned Gating: Broadcasting the 5x24 matrix over (B, S, L, F)
        if use_phylo:
            w = self.phylo_weights.view(1, 1, num_langs, feat_dim).to(x.device)
            x = x * w
            
        x = x.reshape(batch_size, seq_len, -1)
        _, (h_n, c_n) = self.encoder(x)
        
        h_cat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        c_cat = torch.cat((c_n[-2, :, :], c_n[-1, :, :]), dim=1)
        
        decoder_h = self.h_bridge(h_cat).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        decoder_c = self.c_bridge(c_cat).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        curr_input = torch.zeros((batch_size, 1, self.feat_dim)).to(x.device)
        predictions = []
        
        for _ in range(target_len):
            out, (decoder_h, decoder_c) = self.decoder(curr_input, (decoder_h, decoder_c))
            pred_feat = self.tanh(self.fc_out(out)) 
            predictions.append(pred_feat)
            curr_input = pred_feat 
            
        return torch.cat(predictions, dim=1)

class PhonologicalLoss(nn.Module):
    def __init__(self, weight_profile='baseline'):
        super(PhonologicalLoss, self).__init__()
        self.register_buffer('weights', torch.ones(24))
        
        if weight_profile == 'constrained':
            self.weights[0:5] = 10.0   
            self.weights[5:13] = 5.0
            self.weights[13:] = 2.0
        elif weight_profile == 'extreme':
            self.weights[0:5] = 100.0  
            self.weights[5:13] = 10.0
            self.weights[13:] = 1.0    
            
    def forward(self, pred, target):
        sq_error = (pred - target) ** 2
        return torch.mean(sq_error * self.weights)