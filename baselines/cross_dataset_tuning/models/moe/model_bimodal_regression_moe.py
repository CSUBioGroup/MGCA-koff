# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:05:02 2021

@author: Qilei Liu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class moe(nn.Module):
    def __init__(self, num_experts, drop_r, res_list_num, x1_dim, x2_dim, hid_dim):
        super(moe, self).__init__()
        
        self.embed = bimodal_regression_embed(res_list_num, x1_dim, x2_dim, hid_dim)
        
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim), 
            nn.GELU(),
            nn.Dropout(drop_r),
            nn.Linear(hid_dim, 1),
            nn.GELU()
        ) for i in range(num_experts)])
        
        self.gating = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, num_experts),
            nn.Softmax(dim = 1)
        )
        
    def forward(self, x1, x2):
        x = self.embed(x1, x2)

        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim = 1)
        
        gating_outputs = self.gating(x)
        
        final_outputs = torch.sum(expert_outputs * gating_outputs.unsqueeze(-1), dim = 1)
        
        return final_outputs

class bimodal_regression_embed(nn.Module):
    def __init__(self, res_list_num, x1_dim, x2_dim, hid_dim):
        super(bimodal_regression_embed, self).__init__()
        
        self.V_mask = nn.Linear(x1_dim, hid_dim, bias = False)    
        self.V_mask.weight.requires_grad = False
        self.V = nn.Linear(x1_dim, hid_dim)
        self.ag1 = att_gate_layer(hid_dim, hid_dim)
        
        self.embed = nn.Embedding(res_list_num + 1, x2_dim, padding_idx = 0)
        self.embed.weight.requires_grad = False
        self.cnn_mask = nn.Conv1d(x2_dim, hid_dim, kernel_size = 3, stride = 3, bias = False)
        self.cnn_mask.weight.requires_grad = False
        self.cnn = nn.Conv1d(x2_dim, hid_dim, kernel_size = 3, stride = 3)
        self.max = nn.MaxPool1d(kernel_size = 3, stride = 3)
        self.ag2 = att_gate_layer(hid_dim, hid_dim)

    def forward(self, x1, x2):
        x1_mask = self.V_mask(x1)
        x1 = self.V(x1)
        x1 = torch.where(x1_mask != 0, x1, x1_mask)
        x1 = self.ag1(x1)
        x1_sum = x1.sum(1)
        
        x2 = self.embed(x2.long())
        x2 = x2.permute(0, 2, 1) 
        x2_mask = self.cnn_mask(x2)
        x2_mask = self.max(x2_mask)
        x2 = self.cnn(x2)
        x2 = self.max(x2)
        x2 = torch.where(x2_mask != 0, x2, x2_mask)
        x2 = x2.permute(0, 2, 1)
        x2 = self.ag2(x2)
        x2_sum = x2.sum(1)
        
        x = torch.cat([x1_sum, x2_sum], -1)
        return x

class att_gate_layer(nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(att_gate_layer, self).__init__()
        
        self.gelu = nn.GELU()
        self.W_mask = nn.Linear(n_in_feature, n_out_feature, bias = False)
        self.W_mask.weight.requires_grad = False
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.gate = nn.Linear(n_out_feature * 2, 1)    
        
    def forward(self, x):
        x_mask = self.W_mask(x)
        x_p = self.W(x)
        x_p = torch.where(x_mask != 0, x_p, x_mask)
        e = torch.einsum('ijl,ikl->ijk', (x_p, x_p))
        e = e + e.permute(0, 2, 1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(e != 0, e, zero_vec)
        attention = F.softmax(attention, dim = -1)
        x_pp = self.gelu(torch.einsum('aij,ajk->aik', (attention, x_p)))
        x_pp = torch.where(x_mask != 0, x_pp, x_mask)
        return x_pp