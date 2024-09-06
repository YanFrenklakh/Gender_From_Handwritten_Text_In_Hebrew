"""
Author: Yan Fren
Date: 2024-09-06
"""
import secrets
import csv
import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam, SGD, Adadelta
from torch.autograd import Variable
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from skimage import io
import cv2
import sys
from torcheval.metrics import BinaryAccuracy
np.set_printoptions(threshold=sys.maxsize)
torch.cuda.empty_cache()

def preprossess(csv_file, root_dir):
    def pic_best_slice(img1, img2, img3, img4):
        images = (img1, img2, img3, img4)
        arr = []
        for element in enumerate(images):
            try:
                arr.append(np.count_nonzero(element==255) / np.count_nonzero(element==0) * 100)
            except ZeroDivisionError:
                arr.append(0)
        
        max_val = max(arr)
        index = arr.index(max_val)
        return images[index]

    df = pd.read_excel(csv_file)
    idx = df.id.tolist()
    train = re.search(r"^[\/\-a-zA-Z0-9]*train[\-\.a-zA-Z0-9]*", csv_file)
    val = re.search(r"^[\/\-a-zA-Z0-9]*val[\-\.a-zA-Z0-9]*",csv_file)
    
    f = ""
    body = "data/"
    if train:
        body = body + 'train-C3-new/' 
        f = body + 'train.csv'
    elif val :
        body = body + 'val-C3-new/'
        f = body + 'val.csv'
    else:
        body = body + 'test-C3-new/'
        f = body + 'test.csv'

    with open(f, 'w', newline='') as csvfile:
        if not val and not train:
            fieldnames = ['oldId', 'id']
        else:
            fieldnames = ['oldId', 'id', 'label']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        
        for idx in idx:
            img_name = os.path.join(root_dir, str(idx) + ".tiff")
            image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            # remove big black borders
            _,thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            i = 0;
            if len(contours) > 200:
                secure_random = secrets.SystemRandom()
                contours = secure_random.sample(contours, 200)
            for contour_line in contours:
                data = []
                x, y, w, h = cv2.boundingRect(contour_line)
                image2 = image[x:x+400,y:y+400]
                image3 = image[x:x+400,y:y-400]
                image4 = image[x:x-400,y:y+400]
                image5 = image[x:x-400,y:y-400]
                image2 = pic_best_slice(cv2.bitwise_not(image2),
                        cv2.bitwise_not(image3),
                        cv2.bitwise_not(image4),
                        cv2.bitwise_not(image5))
                
                if not cv2.countNonZero(image2) == 0:
                    if np.count_nonzero(image2==255) / np.count_nonzero(image2==0) * 100 > 1.5:
                        if not cv2.imwrite(body + str(idx) + "_" + str(i) + ".tiff", image2):
                            raise Exception("Could not write image")
                        data.append(str(idx) + "_" + str(i))
                        data.append(idx)
                        if val or train:
                            data.append(df.loc[df.id == idx].label.values[0])
                        writer.writerow(data)
                i += 1

# Catout on the images
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

# costum dataset class and DataLoader
class TextImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train', prossesed=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv = re.search(r"\w+\.csv", csv_file)
        self.file = csv_file
        if csv:
            self.data_frame = pd.read_csv(csv_file)
        else: 
            self.data_frame = pd.read_excel(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.prossesed = prossesed

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not self.prossesed:
            img_name = os.path.join(self.root_dir,
                                str(self.data_frame.iloc[idx, 0]) + ".tiff")
            image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            image = cv2.medianBlur(image,5)
            image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
            #remove big black borders
            _,thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
            contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            x, y = [], []

            for contour_line in contours:
                for contour in contour_line:
                    x.append(contour[0][0])
                    y.append(contour[0][1])

            x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
            image = image[y1:y2, x1:x2]
            image = cv2.bitwise_not(image)

            if self.transform:
                image = self.transform(image)


            if self.mode in ['train', 'valid']:
                labels = self.data_frame.iloc[idx, 1]

                return image, labels
        
            return image
        else:
            img_name = os.path.join(self.root_dir, str(self.data_frame.iloc[idx, 0]) + ".tiff")
            image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            if self.transform:
                image = self.transform(image)

            if self.mode in ['train', 'val']:
                labels = self.data_frame.iloc[idx, 2]
                
                return image, labels
            return image, self.data_frame.iloc[idx, 1], self.data_frame.iloc[idx, 0] 

 
batch_size = 16
number_of_labels = 2
classes = (0, 1)

# Convolution neural network
class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            #nn.Dropout(0.4),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size = 2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            #nn.Dropout(0.6),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size = 2),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(24*24*128, num_classes),
            nn.Dropout(0.5),
            nn.Softmax(dim=1))
    def forward(self, input):
        output = self.layer1(input)      
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output

model = CNN(1, number_of_labels)

# Loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss() #nn.BCELoss()
optimizer = Adadelta(model.parameters(), lr=0.001, weight_decay=0.001)

def postprocessing(y, threshold=0.5):
    '''
    set input y with values larger than threshold to 1 and lower than threshold to 0
    input: y - [N,1] numpy array or pytorch Tensor
    output: int array [N,1] the same class type as input
    '''
    assert type(y) == np.ndarray or torch.is_tensor(
        y), f'input should be numpy array or torch tensor. Received input is: {type(y)}'
    assert len(y.shape) == 2, f'input shape should be [N,classes]. Received input shape is: {y.shape}'
    if torch.is_tensor(y):
        return (y >= threshold).int()
    else:
        return (y >= threshold).astype(int)

# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric = BinaryAccuracy()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            predicted = torch.argmax(outputs, dim=1)
            total += labels.numel()
            accuracy += (predicted == labels.to(device)).long().sum().item()
            #metric.update(predicted, labels.to(device))

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)
    #return(metric.compute().item())

# Function to test what classes performed well
def testClassess():
    metric = BinaryAccuracy()
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            predicted = postprocessing(outputs)
            metric.update(predicted.squeeze(), labels.to(device))
            
            #c = (predicted == labels.size(-1)).long().squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += metric.compute()[i].item()
                class_total[label] += 1

    for i in range(number_of_labels):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

# Training function.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train(True)
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


def classify():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    output_file = "./resolts.csv"
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['id', 'sempel_id', 'label']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images, orig_id, cur_id = data
                outputs = model(images.to(device))
                predicted = torch.argmax(outputs, dim=1)
                writer.writerows(zip(orig_id.detach().cpu().numpy(), cur_id,
                    predicted.squeeze().detach().cpu().numpy()))
    

def data_loader(batch_size,
                shuffle=True,
                test=False):
  
    normalize = transforms.Normalize(
        mean=[0.5, ],
        std=[0.5, ],
    )

    # define transforms
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((100, 100)),
                                      transforms.ToTensor(),
                                      #normalize,
    ])
    test_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((100, 100)),
            #normalize,
    ])

    val_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((100,100)),
            #normalize
    ])

    if test:
        train_dataset = TextImagesDataset(
            csv_file='data/test-C3-new/test.csv',
            root_dir='data/test-C3-new/',
            transform=test_transform,
            mode='test',
            prossesed=True
        )

        data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = TextImagesDataset(
        csv_file='data/train-C3-new/train.csv',
        root_dir='data/train-C3-new/',
        transform=train_transform,
        mode='train',
        prossesed=True
    )

    valid_dataset = TextImagesDataset(
        csv_file='data/val-C3-new/val.csv', 
        root_dir='data/val-C3-new/',
        transform=val_transform, 
        mode='val',
        prossesed=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size)

    return (train_loader, valid_loader)


if __name__ == "__main__":
    #preprossess(csv_file='data/train-C3/train-c3-labels.xlsx', root_dir='data/train-C3/')
    #preprossess(csv_file='data/val-C3/val-c3-labels.xlsx', root_dir='data/val-C3/')
    #preprossess(csv_file='data/test-C3/test-c3-IDs.xlsx', root_dir='data/test-C3/')
    
    # lode the data in to train loders
    
    train_loader, val_loader = data_loader(batch_size=200)
    
    #model = CNN(1, number_of_labels)
    #path = "./myFirstModel.pth"
    #model.load_state_dict(torch.load(path))
    train(20)
    print('Finished Training')
    #saveModel()
    # Test which classes performed well
    testAccuracy()
    
    test_loader = data_loader(batch_size=200, test=True)

    classify()
    x = os.path.abspath('./')
    x = x + "/resolts.csv"
    df = pd.read_csv(x)
    df2 = df.groupby('id')['label'].apply(lambda x: x.value_counts().index[0]).reset_index()
    df2 = df2[['id','label']]
    s = pd.read_excel('data/test-C3/test-c3-IDs.xlsx')['id']
    df2 = df2.set_index('id').reindex(s).rename_axis('id').reset_index('id')
    
    df2.label = df2.label.fillna(0)
    df2.label = df2.label.astype('int64')
    df2.to_csv(x, encoding='utf-8', index=False)
