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
        c = 3
        l = 5
        stride_coff = 1
        self.net = torch.nn.Sequential(
          torch.nn.Conv2d(3, 32, 5, 1, 3),
          torch.nn.BatchNorm2d(32),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
          torch.nn.Conv2d(32, 64, 3, 1, 3),
          torch.nn.BatchNorm2d(64),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
          torch.nn.Conv2d(64, 128, 3, 1, 1),
          torch.nn.BatchNorm2d(128),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
          torch.nn.Conv2d(128, 256, 3, 1, 1),
          torch.nn.BatchNorm2d(256),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
          torch.nn.Conv2d(256, 512, 3, 1, 1),
          torch.nn.BatchNorm2d(512),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
          torch.nn.Conv2d(512, 256, 3, 1, 1),
          torch.nn.BatchNorm2d(256),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
          torch.nn.Conv2d(256, 128, 3, 1, 1),
          torch.nn.BatchNorm2d(128),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
          torch.nn.Conv2d(128, 64, 3, 1, 1),
          torch.nn.BatchNorm2d(64),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
          torch.nn.UpsamplingBilinear2d(scale_factor = 2),
          torch.nn.Conv2d(64, 32, 3, 1, 1),
          torch.nn.BatchNorm2d(32),
          torch.nn.Dropout(p=0.25),
          torch.nn.ReLU(),
           torch.nn.UpsamplingBilinear2d(scale_factor = 2),
          torch.nn.Conv2d(32, l, 3, 1, 1)
        )
        #transforms = torch.nn.Sequential(transforms.CenterCrop(10), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)
        if stride_coff != 1 or l != c:
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
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        z = self.net(x)
        z = z[:,:,:x.shape[2],:x.shape[3]]
        tag_scores = F.log_softmax(z,dim=1)
        return z + identity


        
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
