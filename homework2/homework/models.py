import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        L = [] # a list to contain all layers for the network
        c = n_input_channels #to hold input channel for layer to create next
        #construct a convolution and nonlinearity for each layer
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size))
            L.append(torch.nn.ReLU())
            c = l

        #add final convolution as classification layer
        #kernel_size 3 worked well
        L.append(torch.nn.Conv2d(c, 6, kernel_size=1)) 
        self.network = torch.nn.Sequential(*L)
        
    def forward(self, x):
        return self.network(x).mean(dim=[2,3])


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
