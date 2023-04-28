import torch
import torch.nn.functional as F

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


# binary classifier
class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=1, kernel_size=3):
        super().__init__()
        self.input_mean = torch.Tensor([0.3235, 0.3310, 0.3445])
        self.input_std = torch.Tensor([0.2533, 0.2224, 0.2483])

        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
        return self.classifier(z.mean(dim=[2, 3]))

# this binary classifier was used for testing purposes only (not used anymore)
class ActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 16, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 5, stride=2),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(32, 1)
    
    def forward(self, x):
        f = self.network(x)
        return self.classifier(f.mean(dim=(2,3)))

# puck center detection class
class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 32, 32]):
        super().__init__()
        conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 5, 2, 2), torch.nn.ReLU(True)]
        h, _conv = 3, []
        for c in channels:
            _conv += conv_block(c, h)
            h = c
        self._conv = torch.nn.Sequential(*_conv, torch.nn.Conv2d(h, 1, 1))

    def forward(self, img):
        """
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        return spatial_argmax(x[:, 0])


model_factory = {
    'cnn1': CNNClassifier,
    'cnn2': ActionNet,
    'planner' : Planner,
}

def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
