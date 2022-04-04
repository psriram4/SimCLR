import torch
import torch.nn as nn
import torch.nn.functional as F

from pl_bolts.models.self_supervised.evaluator import Flatten

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            # Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        out = self.model(x)
        out = F.normalize(out, dim=1)
        return out