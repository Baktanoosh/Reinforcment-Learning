from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import numpy as np
from os import path

def train(args):
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
  
    train_path = "/content/cs342/homework3/data/train"
    valid_path = "/content/cs342/homework3/data/valid"
    
    num_epochs = 100
    learning_rate = 0.0001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    model.to(device)
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)
    loss = torch.nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
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
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            global_step += 1
            
        model.eval()
        total_step = 0
        accuracy = 0
        for image, label in valid_data:
          image = image.to(device)
          label = label.to(device)
          pred = model(image)
          accuracy = accuracy + (pred.argmax(1) == label).float().mean().item()
          total_step += 1
        print("------------------------------------------------------------")
        print("Epoch: " + str(epoch+1))
        print("Accuracy: " + str(accuracy/total_step))  

    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

    
