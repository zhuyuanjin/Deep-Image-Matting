import torch
from torch import nn
import torchvision.models as models
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()
        ##Encoder
        # vgg = models.vgg11_bn(pretrained=True)
        # self.features = vgg
        # self.features.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # weight = torch.zeros([64, 4, 3, 3], dtype=torch.float)
        # weight[:, 0:3, :, :] = vgg.features[0].weight
        # state_dict = vgg.state_dict()
        # state_dict['features.0.weight'] = weight
        # self.features.load_state_dict(state_dict)

        self.layer0 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2 = nn.ReLU(inplace=True)
        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)


        self.layer4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer5 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer6 = nn.ReLU(inplace=True)
        self.layer7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)


        self.layer8 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer9 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer10 = nn.ReLU(inplace=True)
        self.layer11 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer12 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer13 = nn.ReLU(inplace=True)
        self.layer14 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)


        self.layer15 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer16 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer17 = nn.ReLU(inplace=True)
        self.layer18 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer19 =  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20 = nn.ReLU(inplace=True)
        self.layer21 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)


        self.layer22 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer23 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer24 = nn.ReLU(inplace=True)
        self.layer25 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer26 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer27 = nn.ReLU(inplace=True)
        self.layer28 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)


        ##Decoder
        self.deconv6 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        self.deconvBN6 = nn.BatchNorm2d(512)


        self.Unpooling5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5 = nn.Conv2d(512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.deconvBN5 = nn.BatchNorm2d(512)


        self.Unpooling4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4 = nn.Conv2d(512, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.deconvBN4 = nn.BatchNorm2d(256)


        self.Unpooling3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3 = nn.Conv2d(256, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.deconvBN3 = nn.BatchNorm2d(128)


        self.Unpooling2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = nn.Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.deconvBN2 = nn.BatchNorm2d(64)


        self.Unpooling1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.deconvBN1 = nn.BatchNorm2d(64)

        self.conv_final = nn.Conv2d(64, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        self.initialize()


    def forward(self, x):
        #Encoder
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x, indices1 = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x, indices2 = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x, indices3 = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x, indices4 = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        x = self.layer25(x)
        x = self.layer26(x)
        x = self.layer27(x)
        x, indices5 = self.layer28(x)

        #Decoder
        x = self.deconvBN6(F.relu(self.deconv6(x)))
        x = self.Unpooling5(x, indices5)
        x = self.deconvBN5(F.relu(self.deconv5(x)))
        x = self.Unpooling4(x, indices4)
        x = self.deconvBN4(F.relu(self.deconv4(x)))
        x = self.Unpooling3(x, indices3)
        x = self.deconvBN3(F.relu(self.deconv3(x)))
        x = self.Unpooling2(x, indices2)
        x = self.deconvBN2(F.relu(self.deconv2(x)))
        x = self.Unpooling1(x, indices1)
        x = self.deconvBN1(F.relu(self.deconv1(x)))
        x = self.conv_final(x)
        return x


    def initialize(self):
        vgg = models.vgg11_bn(pretrained=True)
        tmp = torch.zeros([64, 4, 3, 3])
        tmp[:, 0:3, :, :] = vgg.features[0].weight
        self.layer0.load_state_dict(OrderedDict([('weight', tmp), ('bias', vgg.features[0].bias)]))
        self.layer1.load_state_dict(vgg.features[1].state_dict())
        self.layer2.load_state_dict(vgg.features[2].state_dict())
        self.layer3.load_state_dict(vgg.features[3].state_dict())
        self.layer4.load_state_dict(vgg.features[4].state_dict())
        self.layer5.load_state_dict(vgg.features[5].state_dict())
        self.layer6.load_state_dict(vgg.features[6].state_dict())
        self.layer7.load_state_dict(vgg.features[7].state_dict())
        self.layer8.load_state_dict(vgg.features[8].state_dict())
        self.layer9.load_state_dict(vgg.features[9].state_dict())
        self.layer10.load_state_dict(vgg.features[10].state_dict())
        self.layer11.load_state_dict(vgg.features[11].state_dict())
        self.layer12.load_state_dict(vgg.features[12].state_dict())
        self.layer13.load_state_dict(vgg.features[13].state_dict())
        self.layer14.load_state_dict(vgg.features[14].state_dict())
        self.layer15.load_state_dict(vgg.features[15].state_dict())
        self.layer16.load_state_dict(vgg.features[16].state_dict())
        self.layer17.load_state_dict(vgg.features[17].state_dict())
        self.layer18.load_state_dict(vgg.features[18].state_dict())
        self.layer19.load_state_dict(vgg.features[19].state_dict())
        self.layer20.load_state_dict(vgg.features[20].state_dict())
        self.layer21.load_state_dict(vgg.features[21].state_dict())
        self.layer22.load_state_dict(vgg.features[22].state_dict())
        self.layer23.load_state_dict(vgg.features[23].state_dict())
        self.layer24.load_state_dict(vgg.features[24].state_dict())
        self.layer25.load_state_dict(vgg.features[25].state_dict())
        self.layer26.load_state_dict(vgg.features[26].state_dict())
        self.layer27.load_state_dict(vgg.features[27].state_dict())
        self.layer28.load_state_dict(vgg.features[28].state_dict())
        return

