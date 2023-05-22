'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-17 01:14:39
'''
import math
from tqdm import tqdm

def find_lr(model,optimizer,loss_fn,dataloader,init_value=1e-8,final_value=1,beta=0.98,
            inputs_transform=lambda x: x,outputs_transform=lambda x: x,targets_transform=lambda x: x,
            device=None):
    '''
    Description: Function for finding a good init-learning-rate
    Args (type): 
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        dataload  er :(Iterable): Collection of batches allowing repeated iteration 
            (e.g., list or `DataLoader`)
        init_value  (float): init value of learning-rate
        final_value (float): final value of learning rate
        input_transform (callable): a callable that is used to transform the input.
            This can be useful if, for example, you have a multi-input model and
            you want to compute the metric with respect to one of the inputs.
            The input is is expected to be a tuple (prediction, target) or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments.
        output_transform (callable):a callable that is used to transform the output.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is is expected to be a tuple (prediction, target) or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments.
        beta (float) : beta is a parameter we pick between 0 and 1. it will to average losses and
            reduce the noise
    Return : 
        log_lrs (list): list of lrs
        log_losses(list): list of losses
    Reference:
        https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    '''
    def set_lr(optimizer,lr):
        optimizer.param_groups[0]['lr'] = lr
        
    assert(final_value>init_value)
    factor = (final_value/init_value)**(1/len(dataloader))

    if device is not None:
        model.to(device)
    lr = init_value
    set_lr(optimizer,lr)

    batch_count = 0
    avg_loss = 0
    best_loss = 0
    log_losses = []
    log_lrs = []
    for (inputs,targets) in tqdm(dataloader):
        # import pdb; pdb.set_trace()
        batch_count +=1
        inputs,targets = inputs_transform(inputs),targets_transform(targets)
        inputs,targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        loss = model(inputs,targets)
        # outputs = model(inputs,targets)
        # outputs = outputs_transform(outputs)
        # loss = loss_fn(outputs,targets)
        
        # smooth loss
        avg_loss = beta*avg_loss + (1-beta)*loss
        smooth_loss = avg_loss/(1-beta**(batch_count))

        if (batch_count > 1) and (smooth_loss >4*best_loss):
            return log_lrs,log_losses
        if smooth_loss < best_loss or batch_count==1:
            best_loss = smooth_loss
        log_losses.append(smooth_loss.cpu().data.numpy())
        log_lrs.append(lr)

        loss.backward()
        optimizer.step()

        lr = lr * factor
        set_lr(optimizer,lr)
    return log_lrs,log_losses