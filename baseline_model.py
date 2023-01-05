import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torchvision.utils import save_image


class convBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3)
        self.relu = nn.ReLU()
    
    def forward(self, input:torch.tensor):
        input = self.conv1(input)
        input = self.relu(input)
        input = self.conv2(input)
        return input


class UnetEncoder(nn.Module):
    def __init__(self, channel_list:list):
        super(UnetEncoder, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block_list = []

        for i in range(len(channel_list)-1):
            temp = convBlock(channel_list[i], channel_list[i+1])
            if torch.cuda.is_available():
                temp.to('cuda')
            self.block_list.append(temp)
    
    def forward(self, input:torch.tensor):
        layered_encoder_out = []

        for block in self.block_list:
            input = block(input)
            layered_encoder_out.append(input)
            input = self.pool(input)

        return layered_encoder_out 


class UnetDecoder(nn.Module):
    def __init__(self, channel_list:list):
        super(UnetDecoder, self).__init__()
        self.channel_list = channel_list
        self.block_list = []
        self.up_sampler = []

        for i in range(len(channel_list)-1):
            temp = nn.ConvTranspose2d(channel_list[i], channel_list[i+1], 2, 2)
            if torch.cuda.is_available():
                temp.to('cuda')
            self.up_sampler.append(temp)

        for i in range(len(channel_list)-1):
            temp = convBlock(channel_list[i], channel_list[i+1])
            if torch.cuda.is_available():
                temp.to('cuda')
            self.block_list.append(temp)

    # layer concat takes from the encoder layered ouptut for concat
    def forward(self, input, layered_concat):
        for i in range(len(self.channel_list)-1):
            input = self.up_sampler[i](input)
            concat_feature = self.crop(layered_concat[i], input)
            input = torch.concat([input, concat_feature], dim=1)
            input = self.block_list[i](input)
        return input

    def crop(self, concat_feature, input):
            B, C, H, W = input.shape
            concat_feature = torchvision.transforms.CenterCrop([H, W])(concat_feature)
            return concat_feature


class UNET(nn.Module):
    def __init__(self, in_channel_list:list, out_channel_list:list, classes=1, keep_dim=True, output_size=(1024, 1024)):
        super(UNET, self).__init__()
        self.encoder = UnetEncoder(in_channel_list)
        self.decoder = UnetDecoder(out_channel_list)
        self.compressor = nn.Conv2d(out_channel_list[-1], classes, 1)
        self.output_size = output_size
        self.keep_dim = keep_dim

        if torch.cuda.is_available():
            self.encoder.to('cuda')
            self.decoder.to('cuda')
    
    def forward(self, input):
        encoder_output = self.encoder(input)
        encoder_output = list(reversed(encoder_output))
        output = self.decoder(encoder_output[0], encoder_output[1:])
        output = self.compressor(output)
        if self.keep_dim:
            output = F.interpolate(output, self.output_size)
        return output
