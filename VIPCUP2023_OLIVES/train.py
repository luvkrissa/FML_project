import torch
from utils import AverageMeter,save_model
import sys
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from config import parse_option
import os
from utils import set_loader, set_model, set_optimizer, adjust_learning_rate

def train_supervised(train_loader, model,criterion, optimizer, epoch, opt):
    """one epoch training"""
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()

    for idx, (image, bio_tensor) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = image.to(device)

        labels = bio_tensor.float()

        labels = labels.to(device)
        bsz = labels.shape[0]

        # compute loss
        output = model(images)
        # print()
        # print('labels:',labels, labels.shape)
        # print('output:',output, output.shape)
        loss = criterion(output, labels)
        # print('loss：',loss)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'.format(
                epoch, idx + 1, len(train_loader)))

            sys.stdout.flush()

    return losses.avg

def submission_generate(val_loader, model, opt):
    """validation"""
    model.eval()

    device = opt.device
    out_list = []
    with torch.no_grad():
        for idx, (image) in (enumerate(val_loader)):

            images = image.float().to(device)
            print(images)
            # forward
            output = model(images)
            print(output)
            output = torch.round(torch.sigmoid(output))
            out_list.append(output.squeeze().detach().cpu().numpy())


    out_submisison = np.array(out_list)
    np.save('output',out_submisison)


def sample_evaluation(val_loader, model, opt):
    """validation"""
    model.eval()

    device = opt.device
    out_list = []
    label_list = []
    with torch.no_grad():
        for idx, (image,bio_tensor) in (enumerate(val_loader)):

            images = image.float().to(device)
            labels = bio_tensor.float()

            labels = labels.float()

            label_list.append(labels.squeeze().detach().cpu().numpy())
            # forward
            output = model(images)
            output = torch.round(torch.sigmoid(output))
            out_list.append(output.squeeze().detach().cpu().numpy())
    
    label_array = np.array(label_list)
    out_array = np.array(out_list)
    # print(label_array.shape)
    # print(out_array.shape)
    # print("Labels contain NaN:", np.isnan(out_array).any())
    f = f1_score(label_array,out_array,average='macro')
    print(f"F1 Score (macro): {f:.4f}")
    # Compute accuracy
    accuracy = accuracy_score(label_array, out_array)
    print(f"Accuracy: {accuracy:.4f}")

    # # Compute confusion matrix
    # cm = confusion_matrix(label_array, out_array)
    # print("Confusion Matrix:")
    # print(cm)

    # # Save the arrays to disk for later use
    # np.save(f"label_array_{opt.model}.npy", label_array)
    # np.save(f"out_array_{opt.model}.npy", out_array)
    # print("Arrays saved as 'label_array.npy' and 'out_array.npy'")
    if opt.mode == 'test':
        return out_list, label_list


def main():
    opt = parse_option()

    # build data loader
    train_loader,test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)


    # training routine
    for epoch in range(1, opt.epochs + 1):
        train_supervised(train_loader, model, criterion, optimizer, epoch, opt)

    # submission_generate(test_loader, model, opt)
    sample_evaluation(test_loader, model, opt)

    save_file = os.path.join(
        opt.save_folder, f'last_{opt.model}.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


'''def train_supervised(train_loader, model,criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()

    for idx, (image, bio_tensor, clinical_tensor) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = image.to(device)
        clinical = clinical_tensor.float()
        clinical = clinical.to(device)

        labels = bio_tensor.float()

        labels = labels.to(device)
        bsz = labels.shape[0]

        # compute loss
        output = model(images, clinical)
        # print(clinical)
        # print('labels:',labels, labels.shape)
        loss = criterion(output, labels)
        # print('loss：',loss)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'.format(
                epoch, idx + 1, len(train_loader)))
            #print('output:',output, output.shape)

            sys.stdout.flush()

    return losses.avg


def sample_evaluation(val_loader, model, opt):
    """validation"""
    model.eval()

    device = opt.device
    out_list = []
    label_list = []
    with torch.no_grad():
        for idx, (image,bio_tensor, clinical_tensor) in (enumerate(val_loader)):

            images = image.float().to(device)
            clinical = clinical_tensor.to(device)

            labels = bio_tensor.float()

            labels = labels.float()

            label_list.append(labels.squeeze().detach().cpu().numpy())
            # forward
            output = model(images,clinical)
            output = torch.round(torch.sigmoid(output))
            out_list.append(output.squeeze().detach().cpu().numpy())

    label_array = np.array(label_list)
    out_array = np.array(out_list)
    f = f1_score(label_array,out_array,average='macro')
    print(f)'''
