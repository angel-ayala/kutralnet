import time
import copy
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data.sampler import SubsetRandomSampler

def train_model(model, criterion, optimizer, train_data, val_data, epochs=100, batch_size=32,
                shuffle_dataset=True, use_cuda=True):
    # prepare dataset
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                shuffle=shuffle_dataset, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                shuffle=shuffle_dataset, num_workers=2)

    if use_cuda:
        model.cuda()

    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(train_data), "val": len(val_data)}
    print('data_lengths', data_lengths)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {}
    history['loss'] = []
    history['acc'] = []
    history['val_loss'] = []
    history['val_acc'] = []

    for epoch in range(epochs):
        print('Epoch {:03d}/{:03d}'.format(epoch +1, epochs), end=": ")

        # Each epoch has a training and validation phase
        epoch_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                # optimizer = scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0

            # Iterate over data.
            for i, data in enumerate(data_loaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                if use_cuda:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                # Accuracy
                correct = (outputs.argmax(dim=1) == labels).float().sum()
                running_acc += correct

            epoch_loss = running_loss / data_lengths[phase]
            epoch_acc = running_acc / data_lengths[phase]

            loss_key = 'loss' if phase == 'train' else 'val_loss'
            acc_key = 'acc' if phase == 'train' else 'val_acc'
            history[loss_key].append(epoch_loss)
            history[acc_key].append(epoch_acc)

            print('{} Loss: {:.4f}'.format(phase.capitalize(), epoch_loss), 'Acc: {:.4f}'.format(epoch_acc), end=" | ")
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_elapsed = time.time() - epoch_time
        print('time elapsed: {:.0f}m {:.0f}s'.format(
        epoch_elapsed // 60, epoch_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return history, best_model_wts
# end train_model

def test_model(model, dataset, batch_size=32, use_cuda=True):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    print('Evaluating model')
    if use_cuda:
        model.cuda()

    model.eval()  # Set model to evaluate mode

    total = 0
    correct = 0
    Y_test = []
    y_pred = []
    since = time.time()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            if use_cuda:
                images = images.to('cuda')
                labels = labels.to('cuda')

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Y_test.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

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

# end test_model

def show_samples(data):
    fig = plt.figure()

    for i in range(len(data)):
        sample = data[i]

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
# end show_samples

def plot_history(history):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
# end plot_history
