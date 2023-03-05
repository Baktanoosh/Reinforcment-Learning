import torch
import torch.nn.functional as F
from . import dense_transforms
        
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
        c = input_channels
        l = output_channel
        padding = (kernel_size-1)//2
        stride = 1
        kernel_size = 3
        
        self.L1 = torch.nn.Sequential(torch.nn.Conv2d(c, 32, kernel_size=5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm2d(32), torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1), torch.nn.ReLU())
        self.skip1 = torch.nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1 ,padding=1)

        self.L2 =  torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU())
        self.skip2 = torch.nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1 ,padding=1)

        self.L3 =  torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(128), torch.nn.ReLU())
        self.skip3 = torch.nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1 ,padding=1)

        self.L4 =  torch.nn.Sequential(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU())
        self.skip4 = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1 ,padding=1)

        self.L5 = torch.nn.Sequential(torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),      
            torch.nn.BatchNorm2d(128), torch.nn.ReLU())
        self.skip5 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1 ,padding=1)

        self.L6 = torch.nn.Sequential(torch.nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU())
        self.skip6 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1 ,padding=1)

        self.L7 = torch.nn.Sequential(torch.nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32), torch.nn.ReLU())
        self.skip7 = torch.nn.ConvTranspose2d(32, 5, kernel_size=3, stride=1 ,padding=1)

        self.L8 = torch.nn.Sequential(torch.nn.ConvTranspose2d(64, 5, kernel_size=3, stride=1 ,padding=1, bias=False),
            torch.nn.BatchNorm2d(5), torch.nn.ReLU())
        
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
        layer1 = self.L1(x)
        #print('layer1 ', layer1.shape)  
        layer2 = self.L2(layer1)
        #print('layer2 ', layer2.shape)  
        layer3 = self.L3(layer2)
        #print('layer3 ', layer3.shape)  
        layer4 = self.L4(layer3)
        #print('layer4 ', layer4.shape)  
        layer5 = self.L5(layer4)
        #print('layer5 ', layer5.shape)  
        skip_1 = torch.cat([layer5, layer3], dim=1)
        #print('skip1 ', skip_1.shape)  
        layer6 = self.L6(skip_1)
        #print('layer6 ', layer6.shape)  
        skip_2 = torch.cat([layer6, layer2], dim=1)
        #print('skip2 ', skip_2.shape)  
        layer7 = self.L7(skip_2)
        #print('layer7 ', layer7.shape) 
        skip_3 = torch.cat([layer7, layer1], dim=1)
        #print('skip3 ', skip_3.shape) 
        layer8 = self.L8(skip_3)
        layer2 = layer2 + self.skip1(layer1)
        layer3 = layer3 + self.skip2(layer2)
        layer4 = layer4 + self.skip3(layer3)
        layer5 = layer5 + self.skip4(layer4)
        layer6 = layer6 + self.skip5(layer5)
        layer7 = layer7 + self.skip6(layer6)
        z = layer8 + self.skip7(layer7)
        z = z[:,:,:x.shape[2],:x.shape[3]]
        return z 

        
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
