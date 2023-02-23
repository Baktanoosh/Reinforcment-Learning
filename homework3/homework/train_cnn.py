from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
from os import path

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNNClassifier().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')
    
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
        model.eval()
        train_accuracy_value = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            train_accuracy_value.append(accuracy(model(img), label).detach().cpu().numpy())
        print("------------------------------------------------------------")
        print(f'EPOCH', epoch+2)
        print('Accuracy: ', sum(train_accuracy_value) / len(train_accuracy_value))
    save_model(model)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
