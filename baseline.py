import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    
    def forward(self, input):
        with torch.no_grad():
            output = self.model(input)
            output_predictions = output.argmax(0)
        return output_predictions
