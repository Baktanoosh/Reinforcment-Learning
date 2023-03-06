import torch
import numpy as np
from os import path
from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
schedule_lr=False

def train(args):
    model = FCN()
    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    num_epochs = 25
    learning_rate = 0.001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    model.to(device)
    transform = dense_transforms.Compose((dense_transforms.ColorJitter(0.4701, 0.4308, 0.3839), 
                  dense_transforms.RandomHorizontalFlip(), dense_transforms.RandomCrop(96), dense_transforms.ToTensor()))
    train_data = load_dense_data('dense_data/train')
    train_data_transformed = load_dense_data('dense_data/train', transform=transform)
    valid_data = load_dense_data('dense_data/valid')
    loss = torch.nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    global_step = 0    
    for epoch in range(num_epochs):
        model.train()
        confusion_matrix = ConfusionMatrix()
        for image, label in train_data:
            image = image.to(device)
            label = torch.tensor(label, dtype=torch.long, device=device)
            pred = model(image)
            loss_val = loss(pred, label)
            confusion_matrix.add(pred.argmax(1), label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        for image, label in train_data_transformed:
            image = image.to(device)
            label = torch.tensor(label, dtype=torch.long, device=device)
            pred = model(image)
            loss_val = loss(pred, label)
            confusion_matrix.add(pred.argmax(1), label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        print("------------------------------------------------------------")
        print('Epoch: ', epoch+1)
        print('Accuracy = ',confusion_matrix.average_accuracy)
        model.eval()
        confusion_matrix = ConfusionMatrix()
        for i, (image, label) in enumerate(valid_data):
            image, label = image.to(device), label.to(device)
            pred = model(image)
            confusion_matrix.add(pred.argmax(1), label)
            
            print("------------------------------------------------------------")
            print('Average_Accuracy = ',confusion_matrix.average_accuracy)
            print('Intersection over Union  = ',confusion_matrix.iou)
        scheduler.step(confusion_matrix.iou)
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

    
