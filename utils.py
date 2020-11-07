import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pylab
import params
from datasets import load_dataset

def visulize_loss(train_hist):
    x = range(len(train_hist['Total_loss']))
    x = [i * params.plot_iter for i in x]

    total_loss = train_hist['Total_loss']
    class_loss = train_hist['Class_loss']
    mmd_loss = train_hist['MMD_loss']

    plt.plot(x, total_loss, label = 'total loss')
    plt.plot(x, class_loss, label = 'class loss')
    plt.plot(x, mmd_loss, label = 'mmd loss')

    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.grid(True)
    pylab.show()

def visualize_accuracy(test_hist):
    x = range(len(test_hist['Source Accuracy']))

    source_accuracy = test_hist['Source Accuracy']
    target_accuracy = test_hist['Target Accuracy']

    plt.plot(x, source_accuracy, label = 'source accuracy')
    plt.plot(x, target_accuracy, label = 'target accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.grid(True)
    pylab.show()

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    pylab.show()

def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def mmd_loss(source_features, target_features):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    if params.use_gpu:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        )
    else:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value



class SentimentDataset(Dataset):
    def __init__(self, tokenizer, text, target, max_len=512):
        self.tokenizer = tokenizer
        self.text = text
        self.target = target
        self.max_len =  max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text  = self.text[idx]
        target = self.target[idx]
        
        # encode the text and target into tensors return the attention masks as well
        encoding = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
          'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

def get_train_loader(dataset,  data_file="./data/twitter/train.csv", tokenizer, ):
    """
    Get train dataloader of source domain or target domain
    :return: dataloader
    """
    train = load_dataset("csv", data_files=data_file, split='train[20%:]')
    text, target = train['review_text'], train['sentiment']
    dataset = SentimentDataset(tokenizer=tokenizer, text=text, target=target)
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    return loader

    
def get_test_loader(dataset, data_file="./data/twitter/train.csv", tokenizer):
    """
    Get test dataloader of source domain or target domain
    :return: dataloader
    """

    # first 30% data reserved for validation
    val = load_dataset("csv", data_files=data_file, split='train[:20%]')
    text, target = val['review_text'], val['sentiment']
    dataset = SentimentDataset(tokenizer=self.tokenizer, text=text, target=target)
    loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
    return loader