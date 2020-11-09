import torch
import torchvision
import numpy as np
from torch.autograd import Variable

import utils
import params

def train(common_net, src_net, tgt_net, optimizer, criterion, epoch,
          source_dataloader, target_dataloader, train_hist):

    common_net.train()
    src_net.train()
    tgt_net.train()

    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)

    print("Start steps:")
    print(start_steps)
    print("Total Steps:")
    print(total_steps)


    
    source_iter = iter(source_dataloader)
    target_iter = iter(target_dataloader)
    
    for batch_idx in range(min(len(source_dataloader), len(target_dataloader))):
        # get data
        
        #src_text, src_input_ids, src_attention_mask, src_targets = next(source_iter)
        #tgt_text, tgt_input_ids, tgt_attention_mask, tgt_targets = next(target_iter)

        """Batch iterator getting used up, hence trying this:
        """
        try:
            ##images, targets = next(batch_iterator)
            src_text, src_input_ids, src_attention_mask, src_targets = next(source_iter)
            tgt_text, tgt_input_ids, tgt_attention_mask, tgt_targets = next(target_iter)

            input1=next(source_iter)[src_input_ids]
            attention_mask1=next(source_iter)[ src_attention_mask]
            label1=next(source_iter)[src_targets]

            input2= next(target_iter)[tgt_input_ids]
            attention_mask2=next(target_iter)[tgt_attention_mask]
            label2=next(target_iter)[tgt_targets]            

        except StopIteration:
            source_iter = iter(source_dataloader)
            target_iter = iter(target_dataloader)
            src_text, src_input_ids, src_attention_mask, src_targets = next(source_iter)
            tgt_text, tgt_input_ids, tgt_attention_mask, tgt_targets = next(target_iter)
            #batch_iterator = iter(data_loader)
            #mages, targets = next(batch_iterator)
            input1=next(source_iter)[src_input_ids]
            attention_mask1=next(source_iter)[ src_attention_mask]
            label1=next(source_iter)[src_targets]

            input2= next(target_iter)[tgt_input_ids]
            attention_mask2=next(target_iter)[tgt_attention_mask]
            label2=next(target_iter)[tgt_targets]

        # prepare the data
        
        #input1, label1 = sdata
        #input2, label2 = tdata
        
        """
        if params.use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)
            input2, label2 = Variable(input2), Variable(label2)

        """


        input1, attention_mask1, label1 = input1.cuda(), attention_mask1.cuda(), label1.cuda()
        input2, attention_mask2, label2 = input2.cuda(), attention_mask2.cuda(), label2.cuda()

        optimizer.zero_grad()

        #input1 = input1.expand(input1.shape[0], 3, 28, 28)
        
        input = torch.cat((input1, input2), 0)
        attention_mask = torch.cat((attention_mask1, attention_mask2), 0)
        common_feature = common_net(input, attention_mask)

        src_feature, tgt_feature = torch.split(common_feature['pooler_output'], int(2))

        #print(len(torch.split(common_feature['pooler_output'], int(2))))

        src_output = src_net(src_feature)
        tgt_output = tgt_net(tgt_feature)

        class_loss = criterion(src_output, label1)

        mmd_loss = utils.mmd_loss(src_feature, tgt_feature) * params.theta1 + \
                   utils.mmd_loss(src_output, tgt_output) * params.theta2

        loss = class_loss + mmd_loss
        loss.backward()
        optimizer.step()
        step = epoch * len(target_dataloader) + batch_idx

        #print('Loss.data:')
        #print(loss.size())
        #print(mmd_loss.data)


        if (batch_idx + 1) % params.plot_iter == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tMMD Loss: {:.6f}'.format(
                batch_idx * len(input2), len(target_dataloader.dataset),
                100. * batch_idx / len(target_dataloader), loss.data, class_loss.data,
                mmd_loss.data
            ))
            train_hist['Total_loss'].append(loss.cpu().data)
            train_hist['Class_loss'].append(class_loss.cpu().data)
            train_hist['MMD_loss'].append(mmd_loss.cpu().data)
