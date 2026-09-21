"""The network family, its footprint, and int8 weight quantisation.

Footprint is the quantity matched across compression routes, so it has to be
computed one way for every model.
"""
import copy
import torch
import torch.nn as nn

WIN = 50          # samples, 2 s at 25 Hz
CH = 3            # tri-axial


class AccNet(nn.Module):
    """1D CNN over a window of tri-axial acceleration.

    Each entry of `widths` is one block: convolution, batch norm, ReLU, max pool by 2.
    Global average pooling then a linear head. Because the head sees only the last
    width, the same class serves as teacher, reduced-capacity network and pruned
    network, and width is the only thing that changes down the ladder.

    Constructing a network draws its initial weights from torch's global generator.
    Where that generator stands at construction decides the initialisation, so the
    order in which networks are built is part of what makes a run repeatable.
    """

    def __init__(self, widths=(32, 64, 64), n_classes=4, win=WIN, k=5):
        super().__init__()
        self.widths = tuple(widths)
        blocks, c_in = [], CH
        for c in widths:
            blocks += [nn.Conv1d(c_in, c, k, padding=k // 2),
                       nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c_in, n_classes)
        self.win = win

    def forward(self, x):
        z = self.features(x)
        return self.head(self.pool(z).squeeze(-1))

    def n_params(self):
        """Every trainable parameter, batch norm included."""
        return sum(p.numel() for p in self.parameters())

    def peak_activation(self):
        """Working buffer for one window, in elements.

        A layer needs its input and its output alive at the same time, so the buffer
        is the largest sum of two consecutive tensors in the chain (input, then after
        each convolution and each pool), not the largest single tensor. Taking the
        single largest understates RAM by roughly half. Pooling floors odd lengths.
        """
        sizes, L = [CH * self.win], self.win
        for c in self.widths:
            sizes.append(c * L)
            L //= 2
            sizes.append(c * L)
        return max(sizes[i] + sizes[i + 1] for i in range(len(sizes) - 1))

    def footprint_bytes(self, weight_bits=8, act_bits=8):
        """Flash for the weights and RAM for the working buffer, both in bytes.

        `total` is flash plus RAM. The three differ most, proportionally, for the
        smallest networks, so a "KB" figure should say which of them it is. RAM assumes
        8-bit activations, as an integer deployment would use.
        """
        flash = self.n_params() * weight_bits / 8
        ram = self.peak_activation() * act_bits / 8
        return dict(flash=int(flash), ram=int(ram), total=int(flash + ram),
                    params=self.n_params())


def quantise_int8(model):
    """Return a copy with every convolution and linear weight rounded onto a symmetric
    per-tensor int8 grid, step max|w| / 127, clipped to [-127, 127].

    Only those weights are touched: biases, batch norm and activations stay in
    floating point. A deployed int8 graph also quantises activations, and its
    accuracy can differ from the weight-only figure computed here.
    """
    m = copy.deepcopy(model)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.Conv1d, nn.Linear)):
                w = mod.weight
                s = w.abs().max() / 127.0
                if s > 0:
                    mod.weight.copy_(torch.round(w / s).clamp(-127, 127) * s)
    return m
