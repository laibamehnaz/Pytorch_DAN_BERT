import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pylab
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import utils
import models
import params
import train, test
from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup


###BERT model instead of the Extractor
# create the BERTConfig, BERTTokenizer, and BERTModel 
model_name = "bert-base-uncased"
config = BertConfig.from_pretrained(model_name, output_hidden_states=True,  return_dict=True)
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
bert = BertModel.from_pretrained(model_name, config=config)

src_train_dataloader = utils.get_train_loader('/content/drive/My Drive/Data_summarization/pytorch_DAN/data/books.csv',tokenizer)
src_test_dataloader =utils.get_test_loader('/content/drive/My Drive/Data_summarization/pytorch_DAN/data/books.csv',tokenizer)
tgt_train_dataloader = utils.get_train_loader('/content/drive/My Drive/Data_summarization/pytorch_DAN/data/dvd.csv',tokenizer)
tgt_test_dataloader = utils.get_test_loader('/content/drive/My Drive/Data_summarization/pytorch_DAN/data/dvd.csv',tokenizer)



common_net = bert
src_net = models.BertForSequenceClassification(config, common_net)
tgt_net = models.BertForSequenceClassification(config, common_net)



src_dataiter = iter(src_train_dataloader)
tgt_dataiter = iter(tgt_train_dataloader)

src_text, src_input_ids, src_attention_mask, src_targets = next(src_dataiter)
tgt_text, tgt_input_ids, tgt_attention_mask, tg_targets = next(tgt_dataiter)
#tgt_imgs, tgt_labels = next(tgt_dataiter)

#print(src_input_ids)
#print(type(next(src_dataiter)))
#print(len(next(src_dataiter)))
#print(next(src_dataiter)[src_input_ids])

#print(next(src_dataiter))
#print(next(tgt_dataiter))

src_input_tensor=next(src_dataiter)[src_input_ids]
tgt_input_tensor=next(tgt_dataiter)[tgt_input_ids]

src_imgs_show = src_input_tensor
tgt_imgs_show = tgt_input_tensor

utils.imshow(vutils.make_grid(src_imgs_show))
utils.imshow(vutils.make_grid(tgt_imgs_show))



train_hist = {}
train_hist['Total_loss'] = []
train_hist['Class_loss'] = []
train_hist['MMD_loss'] = []

test_hist = {}
test_hist['Source Accuracy'] = []
test_hist['Target Accuracy'] = []

if params.use_gpu:
    common_net.cuda()
    src_net.cuda()
    tgt_net.cuda()

src_features = common_net(src_input_tensor.cuda())
tgt_features = common_net(tgt_input_tensor.cuda())


print("MODEL OUTPUTS:")
print(src_features['last_hidden_state'][0].size())
##Need to look into what the model returns to be able to plot

src_features = src_features['last_hidden_state'][0].cpu().data.numpy()
tgt_features = tgt_features['last_hidden_state'][0].cpu().data.numpy()

src_features = TSNE(n_components= 2).fit_transform(src_features)
tgt_features = TSNE(n_components= 2).fit_transform(tgt_features)

plt.scatter(src_features[:, 0], src_features[:, 1], color = 'r')
plt.scatter(tgt_features[:, 0], tgt_features[:, 1], color = 'b')
plt.title('Non-adapted')
pylab.show()

###Trying to remove the optimizer 
optimizer = optim.SGD([{'params': common_net.parameters()},
                       {'params': src_net.parameters()},
                       {'params': tgt_net.parameters()}], lr= params.lr, momentum= params.momentum)
"""
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
                {
                "params": [p for n, p in common_net.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in common_net.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in src_net.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in tgt_net.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=3e-4,
                betas=(0.9, 0.99),
                eps=1e-8,
            )
lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=10, num_training_steps=100
            )


"""

criterion = nn.CrossEntropyLoss()

for epoch in range(params.epochs):
    t0 = time.time()
    print('Epoch: {}'.format(epoch))
    train.train(common_net, src_net, tgt_net, optimizer, criterion,
                epoch, src_train_dataloader, tgt_train_dataloader, train_hist)
    t1 = time.time() - t0
    print('Time: {:.4f}s'.format(t1))
    test.test(common_net, src_net, src_test_dataloader, tgt_test_dataloader, epoch, test_hist)

src_features = common_net(Variable(src_imgs.expand(src_imgs.shape[0], 3, 28, 28).cuda()))
tgt_features = common_net(Variable(tgt_imgs.expand(tgt_imgs.shape[0], 3, 28, 28).cuda()))
src_features = src_features.cpu().data.numpy()
tgt_features = tgt_features.cpu().data.numpy()
src_features = TSNE(n_components= 2).fit_transform(src_features)
tgt_features = TSNE(n_components= 2).fit_transform(tgt_features)


utils.visulize_loss(train_hist)
utils.visualize_accuracy(test_hist)
plt.scatter(src_features[:, 0], src_features[:, 1], color = 'r')
plt.scatter(tgt_features[:, 0], tgt_features[:, 1], color = 'b')
plt.title('Adapted')
pylab.show()


