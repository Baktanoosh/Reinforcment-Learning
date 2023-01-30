from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
        
    train_path = "/content/cs342/homework2/data/train"
    valid_path = "/content/cs342/homework2/data/valid"
    n_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.learning_rate)
    if torch.cuda.is_available(): device=torch.device('cuda')
    else: device=torch.device('cpu')
    print('device = ', device)
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    loss = torch.nn.CrossEntropyLoss()
    train_dataloader = load_data(train_path, num_workers=0, batch_size=batch_size)
    valid_dataloader = load_data(valid_path, num_workers=0, batch_size=batch_size)

    for epoch in range(n_epochs):
        train_accuracy = []
        for train_features, train_labels in train_dataloader:
            train_features,train_labels = train_features.to(device),train_labels.to(device)
            o = model(train_features)
            loss_val = loss(o, train_labels)
            train_accuracy.append((accuracy(o,train_labels).cpu().detach().numpy()))
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        valid_features, valid_labels = next(iter(valid_dataloader))
        valid_features, valid_labels = valid_features.to(device), valid_labels.to(device)
        valid_pred = model(valid_features)
        valid_accuracy = accuracy(valid_pred, valid_labels)

        print("-----------------------------------------------------------------------------")
        print("EPOCH: %s"%epoch)
        print("Train Accuracy: %s" % np.mean(train_accuracy))
        print("Validation Accuracy: %s" % valid_accuracy)
        print("-----------------------------------------------------------------------------")

    print("saving model to disk")
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
