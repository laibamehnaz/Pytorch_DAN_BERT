import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size= 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 50, kernel_size= 5)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(50 * 4 * 4, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn4 = nn.BatchNorm1d(100)

    def forward(self, input):
        x = F.max_pool2d(F.relu((self.bn1(self.conv1(input)))), 2)
        x = F.max_pool2d(F.relu((self.conv2_drop(self.bn2(self.conv2(x))))), 2)
        x = x.view(-1, 50 * 4 * 4)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.bn4(x)

        return x

class BertForSequenceClassification(nn.Module):
    def __init__(self, config, model):
        
        super(BertForSequenceClassification, self).__init__()
    
        self.num_labels = 2
        print("Printing self")
        print(self.num_labels)
        #self.bert = model
        self.config=config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

   
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #pooled_output = self.dropout(pooled_output)
        pooled_output = self.dropout(input_ids)
        logits = self.classifier(pooled_output)

        return logits
