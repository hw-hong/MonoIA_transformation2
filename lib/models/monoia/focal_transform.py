import torch
import torch.nn as nn


class FocalFeatureTransform(nn.Module):
    """Predicts how an object-level RoI feature changes when its effective focal length changes.

    Input:  roi_vec [N, hidden_dim]  - RoIAlign+GAP feature at a query's predicted box
            delta_f [N] or [N, 1]    - log(target_focal / source_focal) for that query's image
    Output: correction [N, hidden_dim] - additive correction to be applied to the query's
            decoder hidden state (hs) before the size/angle/depth heads consume it.
    """

    def __init__(self, hidden_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([hidden_dim + 1] + h, h + [hidden_dim])
        )
        # zero-init the last layer so correction starts at 0 (no behavior change at init)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, roi_vec, delta_f):
        if delta_f.dim() == 1:
            delta_f = delta_f.unsqueeze(-1)
        x = torch.cat([roi_vec, delta_f], dim=-1)
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
