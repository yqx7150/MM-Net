import torch
import torch.nn as nn

from .model import subnet
from .model_mutiple_base import InvNet 



class MultiOutputReversibleGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, num_blocks):
        super(MultiOutputReversibleGenerator, self).__init__()

       
        subnet_constructor = subnet('DBNet')
        self.generator_k12 = InvNet(input_channels, output_channels, subnet_constructor, num_blocks)
        self.generator_k34 = InvNet(input_channels, output_channels, subnet_constructor, num_blocks)


    def forward(self, x, rev=False):
       

        if rev:
          
            num_slices = 4
            k1, k2, k3, k4 = torch.split(x, x.size(1) // num_slices, dim=1)
            k1_k2 = torch.cat([k1, k2], dim=1)
            k3_k4 = torch.cat([k3, k4], dim=1)

            output12 = self.generator_k12(k1_k2, rev=True)
            output34 = self.generator_k34(k3_k4, rev=True)

            length = output12.size(1)
            output1 = output12[:, :length // 2, :, :]
            output2 = output12[:, length // 2:, :, :]
            output3 = output34[:, :length // 2, :, :]
            output4 = output34[:, length // 2:, :, :]

            output = torch.cat([output1, output2, output3, output4], dim=1)
            return output
        else:

            temp_k1_k2 = self.generator_k12(x)
            temp_k3_k4 = self.generator_k34(x)

            length = temp_k1_k2.size(1)
            k1 = temp_k1_k2[:, :length // 2, :, :]
            k2 = temp_k1_k2[:, length // 2:, :, :]
            k3 = temp_k3_k4[:, :length // 2, :, :]
            k4 = temp_k3_k4[:, length // 2:, :, :]
        
            output = torch.cat([k1, k2, k3, k4], dim=1)
            return output
