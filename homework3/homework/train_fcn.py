import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    num_epochs = 25
    learning_rate = 0.0001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    model.to(device)
    train_data = load_dense_data('dense_data/train')
    valid_data = load_dense_data('dense_data/valid')
    loss = torch.nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0    
    for epoch in range(num_epochs):
        model.train()
        confusion_matrix = ConfusionMatrix()
        for image, label in data_train:
            image = image.to(device)
            label = torch.tensor(label, dtype=torch.long, device=device)
            pred = model(image)
            loss_val = loss(pred, label)
            confu.add(pred.argmax(1), label)
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            train_logger.add_scalar('train/loss', float(loss_val), global_step=global_step)
            global_step += 1
        
        print("------------------------------------------------------------")
        print('Epoch: ', epoch)
        print('Accuracy = ',confusion_matrix.average_accuracy)
        model.eval()
        confusion_matrix = ConfusionMatrix()
        for i, (image, label) in enumerate(data_valid):
            image, label = image.to(device), label.to(device)
            pred = model(image)
            confusion_matrix.add(pred.argmax(1), label)
            
        valid_logger.add_scalar('Eval/accuracy', float(confusion_matrix.average_accuracy), global_step=global_step)
        valid_logger.add_scalar('Eval/iou', float(confusion_matrix.iou), global_step=global_step)
        if valid_logger is None or train_logger is None:
            print("------------------------------------------------------------")
            print('Epoch: ', epoch)
            print('Accuracy = ',confusion_matrix.average_accuracy)
            print('Accuracy = ',confusion_matrix.iou)
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
