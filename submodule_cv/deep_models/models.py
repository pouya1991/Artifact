import torch
import torch.nn as nn
import torchvision.models as models
import os
from efficientnet_pytorch import EfficientNet

####################################################
out_channel = {'alexnet': 256, 'vgg16': 512, 'vgg19': 512, 'vgg16_bn': 512, 'vgg19_bn': 512,
               'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnext50_32x4d': 2048,
               'resnext101_32x8d': 2048, 'mobilenet_v2': 1280, 'mobilenet_v3_small': 576,
               'mobilenet_v3_large': 960 ,'mnasnet1_3': 1280, 'shufflenet_v2_x1_5': 1024,
               'squeezenet1_1': 512, 'efficientnet-b0': 1280, 'efficientnet-l2': 5504,
               'efficientnet-b1': 1280, 'efficientnet-b2': 1408, 'efficientnet-b3': 1536,
               'efficientnet-b4': 1792, 'efficientnet-b5': 2048, 'efficientnet-b6': 2304,
               'efficientnet-b7': 2560, 'efficientnet-b8': 2816, 'inception_v3': 2048}

feature_map = {'alexnet': -2, 'vgg16': -2,  'vgg19': -2, 'vgg16_bn': -2,  'vgg19_bn': -2,
               'resnet18': -2, 'resnet34': -2, 'resnet50': -2, 'resnext50_32x4d': -2,
               'resnext101_32x8d': -2, 'mobilenet_v2': 0, 'mobilenet_v3_large': -2,
               'mobilenet_v3_small': -2, 'mnasnet1_3': 0, 'shufflenet_v2_x1_5': -1,
               'squeezenet1_1': 0, 'inception_v3': -3}

diff_fc_layer = ['mobilenet_v2', 'mnasnet1_3', 'shufflenet_v2_x1_5']
####################################################

# https://github.com/fastai/fastai/blob/186e02d2b20ca3ad295b4a0c101632364eeabe5c/fastai/layers.py#L111
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        super().__init__()

        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Model(nn.Module):

    '''
    Amirali
    '''

    def __init__(self, config):
        super().__init__()

        self.base_model  = config["base_model"]
        self.num_classes = config["num_subtypes"]
        self.pretrained  = config["pretrained"]
        self.last_layers = config["last_layers"]
        self.concat_pool = config["concat_pool"]

        # The idea is to divide the model into two parts:
        # 1. Feature Extraction
        # 2. Classifier (pooling layer is considered here!)
        if self.last_layers=="short":

            if "efficientnet" in self.base_model:
                if self.pretrained:
                    # model = EfficientNet.from_pretrained(self.base_model, num_classes=self.num_classes)
                    self.feature_extract = EfficientNet.from_pretrained(self.base_model, include_top=False)
                else:
                    # model = EfficientNet.from_name(self.base_model, num_classes=self.num_classes)
                    self.feature_extract = EfficientNet.from_name(self.base_model, include_top=False)
                # Add same last layers compare to model {avgpool, dropout, Linear}
                layers = [nn.AdaptiveAvgPool2d(output_size=1), nn.Dropout(p=0.2, inplace=False)]
                layers += [nn.Linear(in_features=out_channel[self.base_model], out_features=self.num_classes)]
                self.classifier = nn.Sequential(*layers)

            else:
                model = getattr(models, self.base_model)
                model = model(pretrained=self.pretrained)
                # Modify last layers
                if 'vgg' in self.base_model or 'alexnet' in self.base_model:
                    model.classifier._modules['6'] = torch.nn.Linear(4096, self.num_classes)
                elif self.base_model in ['mobilenet_v2', 'mnasnet1_3']:
                    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=self.num_classes)
                elif 'mobilenet_v3' in self.base_model:
                    model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=self.num_classes)
                elif 'squeezenet' in self.base_model:
                    model.classifier._modules["1"] = torch.nn.Conv2d(512, self.num_classes, kernel_size=(1, 1))
                    model.num_classes = self.num_classes
                else:
                    num_features = model.fc.in_features
                    model.fc = torch.nn.Linear(num_features, self.num_classes)
                # Seperate feature and classifier layers
                self.feature_extract = nn.Sequential(*list(model.children())[0]) if feature_map[self.base_model]==0 \
                                       else nn.Sequential(*list(model.children())[:feature_map[self.base_model]])
                self.classifier = nn.Sequential(*list(model.children())[1:]) if feature_map[self.base_model]==0 \
                                       else nn.Sequential(*list(model.children())[feature_map[self.base_model]:])

        elif self.last_layers=="long":
            # Check if the model is added to the current models
            if self.base_model not in out_channel:
                raise NotImplementedError(f"{self.base_model} is not considered in the model design!")

            if "efficientnet" in self.base_model:
                if self.pretrained:
                    self.feature_extract = EfficientNet.from_pretrained(self.base_model, include_top=False)
                else:
                    self.feature_extract = EfficientNet.from_name(self.base_model, include_top=False)
            else:
                model = getattr(models, self.base_model)
                model = model(pretrained=self.pretrained)
                self.feature_extract = nn.Sequential(*list(model.children())[0]) if feature_map[self.base_model]==0 \
                                       else nn.Sequential(*list(model.children())[:feature_map[self.base_model]])
            # Last layers same as fastai
            pool = AdaptiveConcatPool2d() if self.concat_pool else nn.AdaptiveAvgPool2d(1)
            layers = [pool, nn.Flatten()]
            num_channel = 2*out_channel[self.base_model] if self.concat_pool else out_channel[self.base_model]
            layers += [nn.BatchNorm1d(num_channel), nn.Dropout(p=0.5)]
            layers += [nn.Linear(num_channel, 512), nn.ReLU(inplace=True)]
            layers += [nn.BatchNorm1d(512), nn.Dropout(p=0.5), nn.Linear(512, self.num_classes)]
            self.classifier = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f"{self.last_layers} is not implemented!")

    def forward(self, x):

        if self.last_layers=="short":
            # Extract feature maps
            feature = self.feature_extract(x)
            # squeezenet does not have fc layer, it uses conv
            if 'squeezenet' in self.base_model:
                out = self.classifier(feature).view(feature.size(0), self.num_classes)
            else:
                # some models do not have avg pooling in their __init__ function
                if self.base_model in diff_fc_layer:
                    num_fc_layer = 0
                    feature_pool = nn.functional.adaptive_avg_pool2d(feature, (1, 1))
                else:
                    num_fc_layer = 1
                    feature_pool = self.classifier[0](feature)
                flatten_feature = torch.flatten(feature_pool, 1)
                out = self.classifier[num_fc_layer:](flatten_feature)

        elif self.last_layers=="long":
            feature = self.feature_extract(x)
            out     = self.classifier(feature)

        return out
