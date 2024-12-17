import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

OUTPUT_METRICS = ['body_fat', 'muscle_mass', 'bone_mass', 'bone_density']

class MultiInputModel(nn.Module):
    def __init__(self, num_tabular_features, outputs):
        super(MultiInputModel, self).__init__()

        self.feature_extractor = models.mobilenet_v2(pretrained=True).features
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_bn = nn.BatchNorm1d(1280)

        self.tabular_dense1 = nn.Linear(num_tabular_features, 32)
        self.tabular_bn = nn.BatchNorm1d(32)

        combined_features_dim = 1280 * 2 + 32
        self.output_layers = nn.ModuleList([nn.Linear(combined_features_dim, 1) for _ in outputs])

    def process_image(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.image_bn(x)

    def forward(self, front_image_input, back_image_input, tabular_input):
        front_features = self.process_image(front_image_input)
        back_features = self.process_image(back_image_input)
        tabular_features = F.relu(self.tabular_dense1(tabular_input))
        tabular_features = self.tabular_bn(tabular_features)

        combined_features = torch.cat([front_features, back_features, tabular_features], dim=1)
        outputs = [output_layer(combined_features) for output_layer in self.output_layers]
        return outputs
