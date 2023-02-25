import torch
import torch.nn.functional as F

        
class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[64,128,256,512,1024], n_input_channels=3, kernel_size=3):
        super().__init__()

        L = []  
        c = n_input_channels  
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size))
            L.append(torch.nn.BatchNorm2d(l))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(3,2,1))
            c = l
        L.append(torch.nn.Conv2d(c, 6, kernel_size=1)) 
        self.network = torch.nn.Sequential(*L)
        #transforms = torch.nn.Sequential(transforms.CenterCrop(10), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)
    def forward(self, x):
        return self.network(x).mean(dim=[2,3])



class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        layers=[32,64,128,256,512,1024,512,256,128,64,32,3]
        L = []  
        c = 32
        kernel_size = 3
        stride_coff = 1
        padding_coff = 1 
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride_coff, padding_coff))
            L.append(torch.nn.BatchNorm2d(l))
            L.append(torch.nn.ReLU())
            if c > l:
                L.append(torch.nn.UpsamplingBilinear2d(scale_factor = 2))
            L.append(torch.nn.MaxPool2d(3,2,1))
            c = l
        L.append(torch.nn.Conv2d(32, 3, 96, 128)) 
        self.network = torch.nn.Sequential(*L)
        self.downsample = None
        if stride_coff != 1 or l != c:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(l, l, 1),torch.nn.BatchNorm2d(l))
        
        
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
