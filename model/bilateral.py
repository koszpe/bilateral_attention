
from torch import nn


class BilateralModule(nn.Module):

    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.bilateral_grid = nn.Conv2d(in_ch, out_ch * k, kernel_size=k, stride=k)
        self.map = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, inputs):
        bg = self.bilateral_grid(inputs)
        m = self.map(inputs)

        out = bg * m

        return out

