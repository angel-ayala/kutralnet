import os
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
# from utils.images import normalize

from datasets import FireImagesDataset, CustomNormalize
from models.octave_resnet import OctFiResNet
from utils.nadam_optim import Nadam

from sklearn.metrics import classification_report

# Seed
seed_val = 666
torch.manual_seed(seed_val)
np.random.seed(seed_val)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

must_train = True
must_test = True

def show_samples(data):
    fig = plt.figure()

    for i in range(len(data)):
        sample = train_data[i]

        print(i, sample[0].shape, sample[1].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        label = 'Fire' if sample[1] == 1 else 'Nofire'
        ax.set_title('Sample {}'.format(label))
        ax.axis('off')
        img = sample[0].transpose(2, 0)
        plt.imshow(img.transpose(0, 1))

        if i == 3:
            plt.show()
            break

# img_dims = (68, 68)
img_dims = (64, 64)
model_name = 'model_octfiresnet.pth'

# common preprocess
transform_compose = transforms.Compose([
           transforms.Resize(img_dims), #redimension
           transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # values [-1, 1]
           # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # values ~[-1, 1]
           CustomNormalize((0, 1))
        ])

#### Training
if must_train:
    # dataset read
    data_path = os.path.join('.', 'datasets', 'FireNetDataset')
    train_data = FireImagesDataset(name='FireNet', root_path=data_path,
                transform=transform_compose)
    val_data = FireImagesDataset(name='FireNet', root_path=data_path,
                purpose='test', transform=transform_compose)

    # train config
    batch_size = 32
    shuffle_dataset = True
    num_classes = len(train_data.labels)
    epochs = 100

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                shuffle=shuffle_dataset, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                shuffle=shuffle_dataset, num_workers=2)

    # model parameters
    net = OctFiResNet(classes=num_classes)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = Nadam(net.parameters())#, lr=0.0001)#, eps=1e-7)#, eps=None)

    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(train_loader) * batch_size, "val": len(validation_loader) * batch_size}
    print('data_lengths', data_lengths)

    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {:03d}/{:03d}'.format(epoch +1, epochs), end=": ")

        # Each epoch has a training and validation phase
        epoch_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                # optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0

            # Iterate over data.
            for i, data in enumerate(data_loaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # statistics
                running_loss += loss.item()#data[0]
                # print('loss.item()', loss.item())
                # Accuracy
                # outputs = (outputs > 0.5).float()
                correct = (outputs.argmax(dim=1) == labels).float().sum()
                # correct = (outputs == labels).float().sum()
                running_acc += correct

            epoch_loss = running_loss / data_lengths[phase]
            epoch_acc = running_acc / data_lengths[phase]
            print('{} Loss: {:.4f}'.format(phase.capitalize(), epoch_loss), 'Acc: {:.4f}'.format(epoch_acc), end=" | ")
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                torch.save(best_model_wts, 'models/saved/' + model_name)

        epoch_elapsed = time.time() - epoch_time
        print('time elapsed: {:.0f}m {:.0f}s'.format(
        epoch_elapsed // 60, epoch_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))




### Test
if must_test:
    # dataset read
    data_path = os.path.join('.', 'datasets', 'FireNetDataset')
    dataset = FireImagesDataset(name='FireNet', root_path=data_path, csv_file='test_dataset.csv',
                transform=transform_compose)

    # test config
    batch_size = 32
    num_classes = len(dataset.labels)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    print('Evaluating model')
    net = OctFiResNet(classes=num_classes)
    net.load_state_dict(torch.load('models/saved/' + model_name))

    net.eval()  # Set model to evaluate mode

    total = 0
    correct = 0
    Y_test = []
    y_pred = []
    since = time.time()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Y_test.extend(labels)
            y_pred.extend(predicted)

    time_elapsed = time.time() - since
    print('Completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    target_names = [ dataset.labels[label]['name'] for label in dataset.labels ]
    print('target_names', target_names)

    class_report = classification_report(Y_test, y_pred,
                            target_names=target_names)#, output_dict=True)

    print('Accuracy of the network on the test images: {:.2f}%'.format(
        100 * correct / total))

    # print('Confusion Matrix', confusion)
    print('Classification Report')
    print(class_report)
