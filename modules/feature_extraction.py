import torch.nn as nn
from modules.cnn.hrnet import HRNet
from modules.cnn.densnet import DenseNet

class HRNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=32):
        super(HRNet_FeatureExtractor, self).__init__()
        self.ConvNet = HRNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

    
class DenseNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(DenseNet_FeatureExtractor, self).__init__()
        self.ConvNet = DenseNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)
# x = torch.randn(1, 1, 32, 400)
# model = HRNet_FeatureExtractor()
# out = model(x)