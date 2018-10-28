import torch
from PIL import Image
from skimage import io, transform
import torchvision
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
import pandas as pd
import os
import time
import copy
import argparse
import matplotlib.pyplot as plt




class cxDataset(torch.utils.data.Dataset):
    def __init__(self, imagedir, labelfile, transform):

        self.imagedir = imagedir
        self.labels = pd.read_csv(labelfile)
        self.transform = transform
        self.diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
         'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
        self.mlb = MultiLabelBinarizer(classes=self.diseases)
        # label disease or not (0 - healthy, 1 - not healthy)
        # self.labels['disease'] = np.where(self.labels['label'] == 10,0,1)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        img = os.path.join(self.imagedir, self.labels.iloc[index,0])
        label = self.mlb.fit_transform([self.labels.iloc[index,1].split('|')]).reshape(14)
        out = { 'image': Image.open(img),'label': torch.from_numpy(label).float()}
        if self.transform:
            out['image'] = self.transform(out['image'])
        return out

    def __len__(self):
        return len(self.labels)


def train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, weight_decay, lr):
    since = time.time()
    best_model_ets = copy.deepcopy(model.state_dict())
    best_loss = 10000
    best_epoch = 0

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('='*60)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for i, sample in enumerate(dataloaders[phase]):
                inputs = sample['image'].to(device)
                labels = sample['label'].to(device)
                batch_size = inputs.size(0)
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) #criteria should be binary cross entropy

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # outputs_binary = torch.Tensor.detach().map_(outputs, threshold, func)
                # running_corrects += min(torch.sum(torch.eq(outputs_binary, labels)))
                running_loss += loss.item() * batch_size
                print("Batch: {}/{}, Loss: {}".format( i , dataset_sizes[phase]/batch_size, running_loss/((i+1) * batch_size)))
            epoch_loss = running_loss / dataset_sizes[phase]
            print("Phase: {}, Epoch: {}, loss: {:4f}".format(phase,epoch, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_ets = copy.deepcopy(model.state_dict())

            if phase == 'val' and epoch_loss > best_loss:
                lr = lr/10

                optimizer = optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay
                    )
                print("New optimizer created with LR: {}".format(lr))

        if epoch - best_epoch > 3:
            print("Early termination of epoch - no improvements in the last 3 epochs")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    print('Best Model: {}'.format(best_model_ets))
    model.load_state_dict(best_model_ets)

    return model, best_epoch

def test_model(model, testloader, testData, name):
    preddf = pd.DataFrame(columns=['Image'])
    truedf = pd.DataFrame(columns=['Image'])

    for i, sample in enumerate(testloader):
        inputs = sample['image'].to(device)
        labels = sample['label'].to(device)
        batch_size = inputs.size(0)

        true_labels = labels.cpu().data.numpy()
        outputs = model(inputs)
        outputs_prob = outputs.cpu().data.numpy()

        for m in range(0, batch_size):
            predrow = {}
            truerow = {}
            predrow['Image'] = testData.labels.iloc[batch_size * i + m, 0]
            truerow['Image'] = testData.labels.iloc[batch_size * i + m, 0]

            for n in range(len(testData.diseases)):
                predrow['prob_' + testData.diseases[n]] = outputs_prob[m,n]
                truerow[testData.diseases[n]] = true_labels[m, n]

            preddf = preddf.append(predrow, ignore_index=True)
            truedf = truedf.append(truerow, ignore_index=True)




    aucdf = pd.DataFrame(columns=["label", "auc"])

    # AUC scores
    for column in truedf.columns[1:]:
        actual = truedf[column]
        pred = preddf["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = roc_auc_score(
                actual.values.astype(int), pred.values)
        except BaseException:
            print("can't calculate auc for " + str(column))
        aucdf = aucdf.append(thisrow, ignore_index=True)

        print("Label: {}, AUC: {}".format(column, thisrow['auc']))
    preddf.to_csv("results/pred_{}.csv".format(name), index=False)
    aucdf.to_csv("results/aucs_{}.csv".format(name), index=False)
    truedf.to_csv("results/true_{}.csv".format(name), index=False)


    return preddf, truedf

def generate_PR_AUC(preddf, truedf, name):

    precision, recall, average_precision = {}, {}, {}
    for column in truedf.columns[1:]:
        actual = truedf[column]
        pred = preddf["prob_" + column]
        precision[column], recall[column], _ = precision_recall_curve(actual.values, pred.values)
        average_precision[column] = average_precision_score(actual.values, pred.values)


    precision["mico"], recall["mico"], _ = precision_recall_curve(truedf.iloc[:,1:].values.ravel(), preddf.iloc[:,1:].values.ravel())
    average_precision["micro"] = average_precision_score(truedf.iloc[:,1:].values, preddf.iloc[:,1:].values)
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    #generate plots
    cm = plt.get_cmap("tab20")
    n_classes = len(truedf.columns[1:])
    colors = [cm(1.*i/n_classes) for i in range(n_classes)]
    lines = []
    labels = []
    plt.figure()
    plt.style.use('ggplot')
    for i, color in zip(truedf.columns[1:], colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append("{}:{:0.2f}".format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(lines, labels, title="PR-AUC", loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.savefig("results/PRAUC_{}.png".format(name), bbox_inches='tight')


def generate_ROC_AUC(preddf, truedf, name):

    fpr, tpr, auc_val = {}, {}, {}
    for column in truedf.columns[1:]:
        actual = truedf[column]
        pred = preddf["prob_" + column]
        fpr[column], tpr[column], thresholds = roc_curve(actual.values, pred.values)
        auc_val[column] = auc(fpr[column], tpr[column])

    cm = plt.get_cmap("tab20")
    n_classes = len(truedf.columns[1:])
    colors = [cm(1. * i / n_classes) for i in range(n_classes)]
    lines = []
    labels = []
    plt.figure()
    plt.style.use('ggplot')
    for i, color in zip(truedf.columns[1:], colors):
        l, = plt.plot(fpr[i], tpr[i], color=color, lw=1)
        lines.append(l)
        labels.append("{}:{:0.2f}".format(i, auc_val[i]))


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(lines, labels, title="ROC-AUC", loc='center right',bbox_to_anchor=(1.5,0.5))
    plt.savefig("results/ROCcurve_{}.png".format(name), bbox_inches='tight')



# def test_model(model, testData, threshold):
#     model.eval()
#     ncorrect = 0
#     with torch.no_grad():
#         for _, sample in enumerate(testData):
#             inputs = sample['image'].to(device)
#             labels = sample['label'].to(device)
#
#             y_pred = model.forward(inputs)
#             y_pred = [1.0 if i > threshold else 0.0 for i in y_pred[0]]
#
#             if np.array_equal(np.array(y_pred), labels[0].cpu().numpy()):
#                 ncorrect+=1.0
#     return ncorrect/len(testData)



class modifiedModel(nn.Module):

    def __init__(self, pretrained):
        super(modifiedModel, self).__init__()
        self.pretrained = pretrained
        self.fc1 = nn.Linear(4096, 14)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        #model = self.relu(self.pretrainedVGG(x))
        model = self.fc1(self.pretrained(x))
        return self.sigmoid(model)

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Supply image directory and label filename")
    parser.add_argument('--i', help="File path of original image directory")
    parser.add_argument('--tr', help="df for train data")
    parser.add_argument('--te', help="df for test name")
    parser.add_argument('--w',type=int, help="number of workers")
    parser.add_argument('--e', type=int, help="number of epochs")
    parser.add_argument('--n', help="Filename of results")

    args = parser.parse_args()


    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epochs = args.e
    filename = args.n
    num_classes = 14
    batch_size = 64
    validation_split = 0.2
    random_seed = 42
    weight_decay = 1e-4
    lr = 0.01


    trainData = cxDataset(imagedir = args.i,
                         labelfile = args.tr,
                          transform=data_transforms['train'])

    testData = cxDataset(imagedir = args.i,
                         labelfile = args.te,
                         transform=data_transforms['test'])


    trainDatasize = trainData.__len__()
    indices = list(range(int(trainDatasize)))
    split = int(np.floor(validation_split*trainDatasize))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)


    train_loader = torch.utils.data.DataLoader(trainData,
                                               batch_size=batch_size,
                                               num_workers=args.w,
                                               sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(trainData,
                                               batch_size=batch_size,
                                               num_workers=args.w,
                                               sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(testData,
                                              batch_size=batch_size,
                                              num_workers=args.w,
                                              shuffle=True)

    train_val_loader = {'train': train_loader, 'val': valid_loader}

    dataset_sizes = {'train': len(train_indices), 'val': len(valid_indices)}

    model = torchvision.models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) #nn.Sequential(*[model.classifier[i] for i in range(4)])

    model = freeze(model)
    model = modifiedModel(model).to(device)
    for name, child in model.named_children():
        print(name,child)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay=weight_decay, momentum=0.9)

    model, best_epoch = train_model(model, criterion, optimizer, num_epochs, train_val_loader, dataset_sizes, weight_decay, lr)
    preddf, truedf = test_model(model, test_loader, testData, filename)
    generate_PR_AUC(preddf, truedf, filename)
    generate_ROC_AUC(preddf, truedf, filename)