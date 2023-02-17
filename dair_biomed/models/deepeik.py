import torch
import torch.nn as nn

class AttentionDecoder(nn.Module):
    def __init__(self, config, all_hidden_size):
        super(AttentionDecoder, self).__init__()
        self.attn_meta2text = nn.MultiheadAttention(all_hidden_size - config["drug"]["text"]["hidden_size"], 1, batch_first=True, kdim=768, vdim=768)
        self.fc1 = nn.Linear(768, config["drug"]["text"]["hidden_size"])
        # all_hidden_size = 768
        self.fc2 = nn.Sequential(
            nn.Linear(all_hidden_size, all_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(all_hidden_size // 2, all_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(all_hidden_size // 4, config["decoder"]["output_size"]),
        )

    def forward(self, h):
        meta_feat = torch.stack(h[:3] + h[4:], dim=1).flatten(start_dim=1)
        k, v, attn_mask = h[3][0], h[3][0], h[3][1]
        _, attn = self.attn_meta2text(meta_feat.unsqueeze(1), k, v)
        attn = attn * attn_mask.unsqueeze(1)

        h = torch.matmul(attn, h[3][0]).squeeze(1)
        h = self.fc1(h)
        h = torch.cat((meta_feat, h), dim=-1)
        h = self.fc2(h).squeeze(1)
        return h

class DeepEIK4DTI(nn.Module):

class DeepEIK4DP(nn.Module):

class DeepEIK4PPI(nn.Module):