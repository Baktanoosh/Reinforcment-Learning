from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch


def train(args):
    train_path = "/content/cs342/homework1/data/train"
    valid_path = "/content/cs342/homework1/data/valid"
    model = model_factory[args.model]()
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    model.to(device)
 
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)
    loss = ClassificationLoss()    
    
    global_step = 0
    for epoch in range(num_epochs):
        train_accuracy = []
        train_accuracy_value = []
        train_loss = []
        train_loss_value = []
        valid_accuracy = []
        valid_accuracy_value = []

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate , momentum=0.9, weight_decay=1e-4)
        for i, (data, labels) in enumerate(train_data):
            o = model(data.to(device))
            train_loss = loss(o, labels.to(device))
            train_loss_value.append(train_loss.float().detach().cpu().numpy())
            train_accuracy = accuracy(o,labels)
            train_accuracy_value.append(train_accuracy.cpu().detach().numpy())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            global_step += 1

            print (f'Epoch - {global_step+1}, Loss - {round(train_loss.item(),3)}')
        print("------------------------------------------------------------")
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
