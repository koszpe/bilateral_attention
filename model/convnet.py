from torch import nn

from model.bilateral import BilateralModule


class ConvNet(nn.Module):
    def __init__(self, class_num, ch_in=3,
                 out_ch_nums= (16, 16, 32, 32, 32, 64, 64, 64),
                 kernel_sizes=(3,  3,  3,  3,  3,  3,  3,  1),
                 strides=     (1,  1,  2,  1,  1,  2,  1,  1),
                 bil_ks=      (5,  5,  5,  5,  5,  5,  5,  5),
                 batch_norm=True,
                 ):
        super().__init__()
        relu = nn.ReLU()
        in_ch_nums=[ch_in] + out_ch_nums[:-1]
        self.blocks = []
        for in_ch, out_ch, kernel_size, stride, bil_k in zip(in_ch_nums, out_ch_nums, kernel_sizes, strides, bil_ks):
            modules = [nn.Conv2d(in_ch, out_ch, kernel_size, stride), relu]
            if batch_norm:
                modules.append((nn.BatchNorm2d(out_ch)))
            modules.append(relu)

            if bil_k > 0:
                modules.append(BilateralModule(out_ch, out_ch, bil_k))
                modules.append(relu)

            self.blocks.append(nn.Sequential(*modules))

        self.fc = nn.Linear(in_features=out_ch_nums[-1], out_features=class_num)



