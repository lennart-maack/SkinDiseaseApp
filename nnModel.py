import torch
import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained= None):
        super(SEResnext50_32x4d, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.dense_output = nn.Linear(2048, 9)
    
    def forward(self, image):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        x = self.dense_output(x)

        return x