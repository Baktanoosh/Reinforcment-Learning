import torch
import torch.nn.functional as F

        
class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[64,128,256,512], input_channels=3, kernel_size=3):
        super().__init__()
        L = []  
        c = input_channels  
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride=1, bias=False))
            L.append(torch.nn.BatchNorm2d(l))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            c = l
        L.append(torch.nn.Conv2d(c, 6, kernel_size=1, stride=1, bias=False))
        self.network = torch.nn.Sequential(*L)

    def forward(self, x):
        z = self.network(x).mean(dim=[2,3])
        return z


class FCN(torch.nn.Module):
    def __init__(self, input_channels=3, output_channel=5, kernel_size=3, stride = 2):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        L = []
        c = input_channels
        l = output_channel
        stride = 2
        padding = (kernel_size-1)//2
        layer_up = [64,128,256,512]
        layer_down = [512,256,128,32]
        L.append(torch.nn.Conv2d(input_channels, 32, 3, stride, padding, bias=False))
        L.append(torch.nn.BatchNorm2d(32))
        L.append(torch.nn.Dropout(p=0.25))
        L.append(torch.nn.ReLU())
        L.append(torch.nn.Conv2d(32, 64, 3, stride, padding, bias=False))
        L.append(torch.nn.BatchNorm2d(64))
        L.append(torch.nn.Dropout(p=0.25))
        L.append(torch.nn.ReLU())

        for l in layer_up:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride, padding, bias=False))
            L.append(torch.nn.BatchNorm2d(l))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(3,2,1))
            c = l
        for l in layer_down:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride, padding, bias=False))
            L.append(torch.nn.BatchNorm2d(l))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(3,2,1))
            c = l
        L.append(torch.nn.Conv2d(32, 1, kernel_size=1, stride= 1))
        self.network = torch.nn.Sequential(*L)

        if stride != 1 or l != c:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(c, l, 1),torch.nn.BatchNorm2d(l))

        
    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        z = self.net(x)
        z = z[:,:,:x.shape[2],:x.shape[3]]
        tag_scores = F.log_softmax(z, dim=1)
        return tag_scores 


        
model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
