import torch
import argparse
import numpy as np
import time
import torch.utils.tensorboard as tb
from os import path
from torchvision import transforms

from .models import Detector, save_model
from .utils import load_detection_data


def train(args):
    # Initialize the detector model
    detector = Detector()

    # Set up tensorboard loggers for train and validation runs
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train' + '/{}'.format(time.strftime('%H-%M-%S'))), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid' + '/{}'.format(time.strftime('%H-%M-%S'))), flush_secs=1)

    # Set up hyperparameters and input data
    learning_rate = args.learning_rate
    train_dir_path = args.train
    valid_dir_path = args.valid
    num_epochs = args.epochs
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print device being used (GPU or CPU)
    print(device)

    # Send model to the specified device
    detector = detector.to(device)

    # Load model state dictionary if continue_training flag is set to True
    if args.continue_training:
        detector.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'image_agent.pt')))

    # Define the optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Define the loss function and send it to the specified device
    loss_function = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(2.0)).to(device)

    # Set up data augmentations for training and validation
    transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()]
    )

    # Load train and validation data using the custom loader function
    train_data_loader = load_detection_data(train_dir_path, num_workers=0, batch_size=batch_size, transform=transform)
    valid_data_loader = load_detection_data(valid_dir_path, num_workers=0, batch_size=batch_size, transform=transform)

    global_step = 0

    # Start training loop for the specified number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        # Set the model to training mode
        detector.train()

        # Iterate over the train data
        for batch_idx, (images, labels) in enumerate(train_data_loader):
            # Send data and labels to the specified device
            images, labels = images.to(device), labels.to(device)

            # Forward pass and compute loss
            logits = detector(images).view(-1, 1, 128, 128)
            loss = loss_function(logits, labels)

            # Log to tensorboard and print to console every 100 steps
            if train_logger is not None and batch_idx % 100 == 0:
                log(train_logger, images, labels, logits, global_step)
                train_logger.add_scalar('train/loss_heat', loss, global_step)

            # Zero out the gradients, backpropagate and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        # Set the model to evaluation mode
        detector.eval()
        
        # Compute validation loss
        validation_loss = 0
        with torch.no_grad():
            for images, labels in valid_data_loader:
                images, labels = images.to(device), labels.to(device)
                logits = detector(images).view(-1, 1, 128, 128)
                validation_loss += loss_function(logits, labels).item()
        
        if valid_logger is not None:
            valid_logger.add_scalar('valid/loss', validation_loss/len(valid_data_loader), global_step)
        
        # Save the damn model
        save_model(model)

def log(logger, imgs, gt_det, det, global_step):
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


def main(log_dir='runs', epochs=20, train='data/train', valid='data/valid',
         learning_rate=1e-4, momentum=0.9, decay=0.01, batch=50, continue_training=False):
    train_args = {
        'log_dir': log_dir,
        'epochs': epochs,
        'train': train,
        'valid': valid,
        'learning_rate': learning_rate,
        'momentum': momentum,
        'decay': decay,
        'batch': batch,
        'continue_training': continue_training
    }
    train(**train_args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-log', '--log_dir', type=str, default='runs')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-t', '--train', type=str, default='data/train')
    parser.add_argument('-v', '--valid', type=str, default='data/valid')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-mo', '--momentum', type=float, default=0.9)
    parser.add_argument('-d', '--decay', type=float, default=0.01)
    parser.add_argument('-b', '--batch', type=int, default=50)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()

    main(log_dir=args.log_dir, epochs=args.epochs, train=args.train, valid=args.valid,
         learning_rate=args.learning_rate, momentum=args.momentum, decay=args.decay,
         batch=args.batch, continue_training=args.continue_training)
