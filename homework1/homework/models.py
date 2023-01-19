import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        loss = torch.nn.functional.cross_entropy(input, target)
        return loss


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input = 3 * 64 * 64
        output = 6
        self.linear = torch.nn.Linear(input,output)


    def forward(self, x):
        flat_linear = self.linear(torch.flatten(x, start_dim=1))      
        return flat_linear
        

class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input = 3 * 64 * 64
        output = 6
        self.linear1 = torch.nn.Linear(input,output)
        
    

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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
