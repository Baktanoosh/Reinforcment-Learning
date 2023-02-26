import torch
import numpy as np
from os import path
from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
schedule_lr=False
learning_rate = 0.001
num_epochs = 50

def train(args):
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
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    model.to(device)
    train_data = load_dense_data('dense_data/train')
    valid_data = load_dense_data('dense_data/valid')
    loss = torch.nn.CrossEntropyLoss()   
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)
    global_step = 0    
    for epoch in range(num_epochs):
        model.train()
        confusion_matrix = ConfusionMatrix()
        val_loss = []
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

        print("------------------------------------------------------------")
        print('Epoch: ', epoch+1)
        print('Accuracy = ',confusion_matrix.average_accuracy)
        model.eval()
        confusion_matrix = ConfusionMatrix()
        for i, (image, label) in enumerate(valid_data):
            image, label = image.to(device), label.to(device)
            pred = model(image)
            confusion_matrix.add(pred.argmax(1), label)
            val_loss.append(confusion_matrix.iou.detach().cpu().numpy())

        if valid_logger is None or train_logger is None:
            print("------------------------------------------------------------")
            print('Average_Accuracy = ',confusion_matrix.average_accuracy)
            print('Intersection over Union = ',confusion_matrix.iou)
        scheduler.step(np.mean(val_loss))
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
