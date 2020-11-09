import torch
from torch.autograd import Variable
import numpy as np

import params
import utils

def test(common_net, src_net, source_dataloader, target_dataloader, epoch, test_hist):

    common_net.eval()
    src_net.eval()

    source_correct = 0
    target_correct = 0

    for batch_idx, sdata in enumerate(source_dataloader):
        #input1, label1 = sdata
        
        src_text, src_input_ids, src_attention_mask, src_targets = sdata
        input1=sdata[src_input_ids]
        attention_mask1=sdata[ src_attention_mask]
        label1=sdata[src_targets]



        """
        if params.use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)
        """
        input1, attention_mask1, label1 = input1.cuda(), attention_mask1.cuda(), label1.cuda()

        ##Why are we expanding this?
        #input1 = input1.expand(input1.shape[0], 3, 28, 28)
        
        output1 = src_net(common_net(input1, attention_mask1)['pooler_output'])
        #pred1 = output1.data.max(1, keepdim = True)[1]
        pred1=torch.argmax(output1, dim=1)
        #pred1=output1
        source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()

    for batch_idx, tdata in enumerate(target_dataloader):
        #input2, label2 = tdata

        tgt_text, tgt_input_ids, tgt_attention_mask, tgt_targets = tdata
        input2= tdata[tgt_input_ids]
        attention_mask2=tdata[tgt_attention_mask]
        label2=tdata[tgt_targets]        

        input2, attention_mask2, label2 = input2.cuda(), attention_mask2.cuda(), label2.cuda()
        """
        if params.use_gpu:
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
        else:
            input2, label2 = Variable(input2), Variable(label2)
        """



        output2 = src_net(common_net(input2, attention_mask2)['pooler_output'])
        
        pred2 = torch.argmax(output2, dim=1)
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()

    source_accuracy = 100. * source_correct / len(source_dataloader.dataset)
    target_accuracy = 100. * target_correct / len(target_dataloader.dataset)

    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'.format(
        source_correct, len(source_dataloader.dataset), source_accuracy,
        target_correct, len(target_dataloader.dataset), target_accuracy,
    ))
    test_hist['Source Accuracy'].append(source_accuracy)
    test_hist['Target Accuracy'].append(target_accuracy)
