from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import numpy as np



def train(args):
    train_path = "/content/cs342/homework2/data/train"
    valid_path = "/content/cs342/homework2/data/valid"
    
    num_epochs = 100
    learning_rate = 0.0001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    model = CNNClassifier()
    model.to(device)
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)
    loss = torch.nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0
    for epoch in range(num_epochs):
        train_accuracy = []
        train_accuracy_value = []
        train_loss = []
        train_loss_value = []
        valid_accuracy = []
        valid_accuracy_value = []

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
            
            print (f'Global Step - {global_step+1}, Loss - {round(train_loss.item(),3)}')
        print("------------------------------------------------------------")
        print (f'EPOCH', epoch)
        save_model(model)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
