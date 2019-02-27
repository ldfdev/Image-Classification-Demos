# code taken from
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torchvision, torch



model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class VGG_modif(nn.Module):

    def __init__(self, features, num_classes=500, init_weights=True):
        super(VGG_modif, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 10),
            nn.ReLU(True)
            # nn.Dropout(),
            # nn.Linear(750, 10),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # initialize weights from pretrained model where possible
        pretrained_model = torchvision.models.vgg11(pretrained=True)
        pretrained_model_dict = pretrained_model.state_dict()

        # print(self.state_dict().keys())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        pretrained_state = { k:v for k,v in pretrained_model_dict.items() if k in self.state_dict() and v.size() == self.state_dict()[k].size() }
        # same calculation as pretrained_state, more verbose
        # for key1, key2 in zip(pretrained_model_dict, self.state_dict()):
        #     # print(key1, key2)
        #     preSize = pretrained_model_dict[key1].size()
        #     Size = self.state_dict()[key2].size()
        #     # print('\t', preSize, Size)
        #     if (preSize == Size):
        #         print(key1)
        #         print('\tsizes match')
        #         self.state_dict()[key2] = pretrained_model_dict[key1]
        self.state_dict().update(pretrained_state)
        self.load_state_dict(self.state_dict())
        # print(type(self.features)) # Sequential
        # for num, elem in enumerate(self.features.children()): #children is preferable over modules as latter return pretrained model + same as shildren
        #     print('Num ',num, elem)
        #     if hasattr(elem, 'weight'):
        #         print('\tWeight', elem.weight)
        #     if hasattr(elem, 'bias'):
        #         print('\tBias', elem.bias)
            # except AttributeError as e:
            #     print(str(e))
        # assert(self.features == pretrained_model.features)

        for num, (self_elem, elem) in enumerate(zip(self.features.children(), pretrained_model.features.children())):
            # print(num, self_elem, elem)
            if hasattr(elem, 'weight'):
                self_elem.weight = elem.weight
                assert(int((self_elem.weight == elem.weight).all()))
            if hasattr(elem, 'bias'):
                self_elem.bias = elem.bias
                assert(int((self_elem.bias == elem.bias).all()))
        print('The 2 models correspond')
        # self.model.load_state_dict(self.state_dict)
        # for num, elem in enumerate(self.features.children()): #children is preferable over modules as latter return pretrained model + same as shildren
        #     print('Num ',num, elem)
        #     if hasattr(elem, 'weight'):
        #         print('\tWeight', elem.weight)
        #     if hasattr(elem, 'bias'):
        #         print('\tBias', elem.bias)
def vgg11_modif():
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG_modif(make_layers(cfg))

if __name__=='__main__':
    model = vgg11_modif()