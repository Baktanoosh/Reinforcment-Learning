from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
from data import load


def train(args):
    train_path = "../../../cs342/homework1/data/train"
    valid_path = "../../../cs342/homework1/data/valid"
    model = model_factory[args.model]()
    num_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    model.to(device)
    to_image = load.to_image_transform()
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)
    loss = ClassificationLoss()    
    
    global_step = 0
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_data.size(0))
        train_accuracy = []
        train_accuracy_value = []
        train_loss = []
        train_loss_value = []
        valid_accuracy = []
        valid_accuracy_value = []
        
        
        for it in range(0, len(permutation)-batch_size+1, batch_size):
            train_data, train_label = train_data.to(device), train_label.to(device)
            valid_data, valid_label = valid_data.to(device), valid_label.to(device)

            batch_samples = permutation[it:it+batch_size]
            batch_data, batch_label = train_data[batch_samples], train_label[batch_samples]
            o = model(batch_data)
            
            train_loss = loss(o, batch_label.float()).detach().cpu().numpy()
            train_loss_value.append(train_loss)
            train_accuracy.extend(((o > 0).long() == batch_label).cpu().detach().numpy())
            train_accuracy_value.append(train_accuracy)
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

            valid_pred = net2(valid_data) > 0
            valid_accuracy = float((valid_pred.long() == valid_label).float().mean())
            valid_accuracy_value.append(valid_accuracy)
            
        print("Epoch: ", epoch)
        print("Accercy: ", train_accuracy_value.mean())
        print("Valid Accercy: ", valid_accuracy_value.mean())
        print("Loss: ", train_loss_value.mean())
        print("------------------------------------------------------------")
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
