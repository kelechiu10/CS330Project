# for all code below, credit to huyvnphan on GitHub
import torch
import torch.nn as nn
#import torchvision.models as models
import os
import requests
import zipfile
from tqdm import tqdm


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
]


class Sequential(nn.Module):
    def __init__(self, args):
        super(Sequential, self).__init__()
        self.layers = args
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input, weights=None):
        if weights is not None:
            i = 0
            for module in self.layers:
                if isinstance(module, BasicBlock) or isinstance(module, Bottleneck):
                    input = module(input, weights[i])
                    i += 1
                elif isinstance(module, nn.Conv2d):
                    print(weights[i])
                    print(module.weight, module.bias)
                    input = module._conv_forward(input, weights[i], None)
                    i += 1
                else:
                    input = module(input)
            return input
        else:
            for module in self.layers:
                input = module(input)
            return input


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, weights=None):
        if weights is not None:
            identity = x

            out = self.conv1._conv_forward(x, weights[0], None)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2._conv_forward(out, weights[1], None)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x, weights[2])

            out += identity
            out = self.relu(out)

            return out
        else:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, weights=None):
        if weights is not None:
            identity = x

            out = self.conv1._conv_forward(x, weights[0], None)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2._conv_forward(out, weights[1], None)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3._conv_forward(out, weights[2], None)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x, weights[3])

            out += identity
            out = self.relu(out)

            return out
        else:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                [conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)]
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return Sequential(layers)

    def forward(self, x, weights=None):
        if weights is not None:
            x = self.conv1._conv_forward(x, weights['conv1.weight'], None)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            layer = 'layer1'
            x = self.layer1(x, [
                [weights[f'{layer}.0.conv1.weight'], weights[f'{layer}.0.conv2.weight'], weights[f'{layer}.0.conv3.weight'],
                 weights[f'{layer}.0.downsample.0.weight']],
                [weights[f'{layer}.1.conv1.weight'], weights[f'{layer}.1.conv2.weight'], weights[f'{layer}.1.conv3.weight']],
                [weights[f'{layer}.2.conv1.weight'], weights[f'{layer}.2.conv2.weight'], weights[f'{layer}.2.conv3.weight']]
                 ])
            layer = 'layer2'
            x = self.layer2(x, [
                [weights[f'{layer}.0.conv1.weight'], weights[f'{layer}.0.conv2.weight'], weights[f'{layer}.0.conv3.weight'],
                 weights[f'{layer}.0.downsample.0.weight']],
                [weights[f'{layer}.1.conv1.weight'], weights[f'{layer}.1.conv2.weight'], weights[f'{layer}.1.conv3.weight']],
                [weights[f'{layer}.2.conv1.weight'], weights[f'{layer}.2.conv2.weight'], weights[f'{layer}.2.conv3.weight']],
                [weights[f'{layer}.3.conv1.weight'], weights[f'{layer}.3.conv2.weight'], weights[f'{layer}.3.conv3.weight']]
                 ])
            layer = 'layer3'
            x = self.layer3(x, [
                [weights[f'{layer}.0.conv1.weight'], weights[f'{layer}.0.conv2.weight'], weights[f'{layer}.0.conv3.weight'],
                 weights[f'{layer}.0.downsample.0.weight']],
                [weights[f'{layer}.1.conv1.weight'], weights[f'{layer}.1.conv2.weight'], weights[f'{layer}.1.conv3.weight']],
                [weights[f'{layer}.2.conv1.weight'], weights[f'{layer}.2.conv2.weight'], weights[f'{layer}.2.conv3.weight']],
                [weights[f'{layer}.3.conv1.weight'], weights[f'{layer}.3.conv2.weight'], weights[f'{layer}.3.conv3.weight']],
                [weights[f'{layer}.4.conv1.weight'], weights[f'{layer}.4.conv2.weight'], weights[f'{layer}.4.conv3.weight']],
                [weights[f'{layer}.5.conv1.weight'], weights[f'{layer}.5.conv2.weight'], weights[f'{layer}.5.conv3.weight']]
                 ])
            layer = 'layer4'
            x = self.layer4(x, [
                [weights[f'{layer}.0.conv1.weight'], weights[f'{layer}.0.conv2.weight'], weights[f'{layer}.0.conv3.weight'],
                 weights[f'{layer}.0.downsample.0.weight']],
                [weights[f'{layer}.1.conv1.weight'], weights[f'{layer}.1.conv2.weight'], weights[f'{layer}.1.conv3.weight']],
                [weights[f'{layer}.2.conv1.weight'], weights[f'{layer}.2.conv2.weight'], weights[f'{layer}.2.conv3.weight']]
                 ])

            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x, [weights['fc.weight'], weights['fc.bias']])
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)

        return x


def download_weights(dir):
    url = (
        "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
    )

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2 ** 20  # Mebibyte
    t = tqdm(total=total_size, unit="MiB", unit_scale=True)

    path_to_zip_file = os.path.join(dir, "state_dicts.zip")
    directory_to_extract_to = os.path.join(dir, "cifar10_models")

    with open(path_to_zip_file, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise Exception("Error, something went wrong")

    print("Download successful. Unzipping file...")
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        print("Unzip file successful!")


def _resnet(arch, block, layers, pretrained_dir, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained_dir is not None:
        print('Loading model weights')
        if not os.path.exists(pretrained_dir):
            os.makedirs(pretrained_dir)
            download_weights(pretrained_dir)

        state_dict = torch.load(
            os.path.join(pretrained_dir, "cifar10_models/state_dicts/", arch + ".pt"), map_location=device
        )
        model.load_state_dict(state_dict)
    layers = [nn.Sequential(model.conv1, model.layer1), model.layer2, model.layer3, model.layer4, model.fc]
    return model, layers


def resnet18(pretrained_dir=None, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained_dir, device, **kwargs
    )


def resnet34(pretrained_dir=None, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained_dir, device, **kwargs
    )


def resnet50(pretrained_dir=None, device="cpu", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Loading pretrained resnet50')
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained_dir, device, **kwargs
    )