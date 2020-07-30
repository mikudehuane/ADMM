import torch
import torchvision
from torch import nn
import os.path as osp

from common import pretrained_models_dir


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class DeFlatten(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, x):
        return x.view(*self.target_shape)


class NetWithMidFeature(nn.Module):
    def __init__(self, label, *modules):
        super().__init__()
        self.label = label
        self.module_list = nn.ModuleList(modules)
        self.device = 'cpu'

    def to_(self, device='cuda'):
        self.to(device=device)
        self.device = device
        return self

    # noinspection PyArgumentList
    def forward(self, inputs):
        inputs = nn.Sequential(*self.module_list)(inputs)
        return inputs

    def get_features(self, inputs, layer, batch_size=5000, verbose=False, to_cuda=False):
        """
        get the output of layer^th layer
        """
        with torch.no_grad():
            modules = nn.Sequential(*self.module_list[:layer + 1])
            outputs = []
            for start in range(0, len(inputs), batch_size):
                stop = start + batch_size
                input_batch = inputs[start:stop]
                input_batch = input_batch.cuda()
                output_batch = modules(input_batch)
                output_batch = output_batch.cpu()
                outputs.append(output_batch)
                if verbose:
                    print(f"{stop} samples processed")
            outputs = torch.cat(outputs)
            if to_cuda:
                outputs = outputs.cuda()
            return outputs

    @staticmethod
    def __get_pretrained_models_dir(fn_no_ext):
        return osp.join(pretrained_models_dir, 'classification', f"{fn_no_ext}.pth")

    def has_been_trained(self, fn_no_ext=None):
        fn_no_ext = self.label if fn_no_ext is None else fn_no_ext
        save_fp = NetWithMidFeature.__get_pretrained_models_dir(fn_no_ext)
        return True if osp.exists(save_fp) else False

    def save(self, fn_no_ext=None):
        fn_no_ext = self.label if fn_no_ext is None else fn_no_ext
        save_fp = NetWithMidFeature.__get_pretrained_models_dir(fn_no_ext)
        # if osp.exists(save_fp):
        #     print(f"The saving path {save_fp} already exists, overwrite it (type yes)?")
        #     command = input()
        #     if command != "yes":
        #         return
        torch.save(self.state_dict(), save_fp)

    def load(self, fn_no_ext=None):
        fn_no_ext = self.label if fn_no_ext is None else fn_no_ext
        save_fp = NetWithMidFeature.__get_pretrained_models_dir(fn_no_ext)
        self.load_state_dict(torch.load(save_fp))


class NetACifar10(NetWithMidFeature):
    def __init__(self):
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(16 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )


class NetAMnist(NetWithMidFeature):
    def __init__(self):
        super(NetAMnist, self).__init__(
            self.__class__.__name__,
            nn.Conv2d(1, 6, 5, padding=2),  # 0
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5, padding=2),  # 3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(16 * 7 * 7, 1024),  # 7
            nn.ReLU(),
            nn.Linear(1024, 100),  # 9
            nn.ReLU(),
            nn.Linear(100, 10),  # 11
        )


class NetBCifar10(NetWithMidFeature):
    def __init__(self):
        c, c1, c2, num_class = 3, 128, 256, 10
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(c, c1, (5, 5), padding=2),  # 0
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, (3, 3), padding=1),  # 3
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),  # 6
            nn.Conv2d(c2, c2, (3, 3), padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),  # 9
            nn.Conv2d(c2, c1, (5, 5), padding=2),
            nn.BatchNorm2d(c1),
            nn.ReLU(),  # 12
            nn.MaxPool2d((2, 2), 2),
            Flatten(),
            nn.Linear(c1 * 8 * 8, num_class),  # 15
            nn.BatchNorm1d(10),
        )


class NetBMnist(NetWithMidFeature):
    def __init__(self):
        c, c1, c2, num_class = 1, 128, 256, 10
        super(NetBMnist, self).__init__(
            self.__class__.__name__,
            nn.Conv2d(c, c1, (5, 5), padding=2),  # 0
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, (3, 3), padding=1),  # 3
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),  # 6
            nn.Conv2d(c2, c2, (3, 3), padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),  # 9
            nn.Conv2d(c2, c1, (5, 5), padding=2),
            nn.BatchNorm2d(c1),
            nn.ReLU(),  # 12
            nn.MaxPool2d((2, 2), 2),
            Flatten(),
            nn.Linear(c1 * 7 * 7, num_class),  # 15
            nn.BatchNorm1d(10),
        )


class NetCMnist(NetWithMidFeature):
    def __init__(self):
        c, c1, num_class = 1, 6, 10
        super(NetCMnist, self).__init__(
            self.__class__.__name__,
            nn.Conv2d(c, c1, (5, 5), padding=2),  # 0
            nn.ReLU(),
            Flatten(),
            nn.Linear(c1 * 28 * 28, 1024),  # 3
            nn.ReLU(),
            nn.Linear(1024, 512),  # 5
            nn.ReLU(),
            nn.Linear(512, 256),  # 7
            nn.ReLU(),
            nn.Linear(256, 100),  # 9
            nn.ReLU(),
            nn.Linear(100, 10),  # 11
        )


class NetAllConvMnist(NetWithMidFeature):
    def __init__(self):
        c, c1, c2, num_class = 1, 2, 4, 10
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(c, c1, (5, 5), padding=2),  # 0
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),    # 2 output: 14 x 14 x 2 = 392
            nn.Conv2d(c1, c2, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),    # 5 output: 7 x 7 x 4 = 192
            nn.Conv2d(c2, num_class, (7, 7), padding=0),
            Flatten(),
        )


class NetAllFcMnist(NetWithMidFeature):
    def __init__(self):
        c, d1, d2, num_class = 28*28, 392, 196, 10
        super().__init__(
            self.__class__.__name__,
            Flatten(),
            nn.Linear(c, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, num_class)
        )


class NetAllConvCifar10(NetWithMidFeature):
    def __init__(self):
        c, c1, c2, num_class = 3, 6, 12, 10
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(c, c1, (3, 3), padding=1),        # *** 0 32 x 32 x 6
            nn.ReLU(),
            nn.Conv2d(c1, c1, (3, 3), padding=1),       # *** 2 32 x 32 x 6
            nn.MaxPool2d((2, 2), 2),
            nn.BatchNorm2d(c1),                         # 4 16 x 16 x 6
            nn.ReLU(),
            nn.Conv2d(c1, c2, (3, 3), padding=1),       # *** 6 16 x 16 x 12
            nn.ReLU(),
            nn.Conv2d(c2, c2, (3, 3), padding=1),       # 8 16 x 16 x 12
            nn.MaxPool2d((2, 2), 2),
            nn.BatchNorm2d(c2),                         # *** 10 8 x 8 x 12
            nn.ReLU(),
            nn.Conv2d(c2, num_class, (8, 8), padding=0),       # *** 12 10
            Flatten(),
        )


class NetMostFcCifar10(NetWithMidFeature):
    def __init__(self):
        c, c1, num_class = 3, 6, 10
        d1, d2, d3 = 32*32*6, 16*16*12, 8*8*12
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(c, c1, (3, 3), padding=1),        # *** 0 32 x 32 x 6
            nn.ReLU(),
            nn.Conv2d(c1, c1, (3, 3), padding=1),       # *** 2 32 x 32 x 6
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(d1, d2),                          # *** 6 3072
            nn.ReLU(),
            nn.Linear(d2, d3),                          # *** 8 768
            nn.BatchNorm1d(d3),
            nn.ReLU(),
            nn.Linear(d3, num_class),                   # *** 11 10
        )


class VGG16Cifar10(NetWithMidFeature):
    def __init__(self, pretrained=True,):
        num_classes = 10
        hidden_layers = 4096
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(3, 64, kernel_size=3, padding=1),         # 0
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 30
            Flatten(),
            nn.Linear(512 * 1 * 1, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, num_classes)
        )
        if pretrained:
            self.features = self.module_list[:31]
            pretrained_vgg = torchvision.models.vgg16(pretrained=pretrained)
            loaded_state_dict = pretrained_vgg.state_dict()
            self.load_state_dict(loaded_state_dict, strict=False)
            # for param in self.features.parameters():
            #     param.requires_grad = False
            del self.features


class VGG16BNCifar10(NetWithMidFeature):
    def __init__(self):
        num_classes = 10
        hidden_layers = 4096
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(3, 64, kernel_size=3, padding=1),         # 0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 30
            Flatten(),
            nn.Linear(512 * 1 * 1, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, num_classes)
        )


class VGG19BNCifar10(NetWithMidFeature):
    def __init__(self):
        num_classes = 10
        hidden_layers = 4096
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(3, 64, kernel_size=3, padding=1),         # 0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Flatten(),
            nn.Linear(512 * 1 * 1, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, num_classes)
        )


# padding the first layer to match 32x32
class VGG16BNMnist(NetWithMidFeature):
    def __init__(self):
        num_classes = 10
        hidden_layers = 100
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(1, 64, kernel_size=3, padding=3),         # 0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Flatten(),
            nn.Linear(512 * 1 * 1, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, num_classes)
        )


class VGG16Mnist(NetWithMidFeature):
    def __init__(self):
        num_classes = 10
        hidden_layers = 100
        super().__init__(
            self.__class__.__name__,
            nn.Conv2d(1, 64, kernel_size=3, padding=1),         # 0
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(512 * 1 * 1, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_layers, num_classes)
        )
