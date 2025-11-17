import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Attention pooling
class AttentionPool(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attn_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        # x: (B, N, D)
        attn_logits = self.attn_v(x).squeeze(-1)  # (B, N)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn_w = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # (B, N, 1)
        bag = (attn_w * x).sum(dim=1)  # (B, D)
        return bag, attn_w.squeeze(-1)

class InstanceEncoder(nn.Module):
    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(res.children())[:-1])  # (B, C, 1, 1)
        self.proj = nn.Linear(res.fc.in_features, out_dim)

    def forward(self, x):
        f = self.backbone(x).view(x.size(0), -1)
        return self.proj(f)  # (B, out_dim)

class MILModel(nn.Module):
    def __init__(self, emb_dim=512, attn_hidden=256, n_classes=2, encoder_pretrained=True):
        super().__init__()
        self.encoder = InstanceEncoder(out_dim=emb_dim, pretrained=encoder_pretrained)
        self.attn = AttentionPool(in_dim=emb_dim, hidden_dim=attn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, bag_images, bag_sizes):
        # bag_images: (sum_N, C, H, W)
        feats = self.encoder(bag_images)  # (sum_N, D)

        B = len(bag_sizes)
        Nmax = max(bag_sizes)
        D = feats.size(-1)

        bags = feats.new_zeros((B, Nmax, D))
        mask = torch.zeros((B, Nmax), dtype=torch.bool, device=feats.device)

        idx = 0
        for i, n in enumerate(bag_sizes):
            bags[i, :n] = feats[idx: idx + n]
            mask[i, :n] = True
            idx += n

        bag_emb, attn_w = self.attn(bags, mask=mask)
        logits = self.classifier(bag_emb)
        return logits, attn_w
