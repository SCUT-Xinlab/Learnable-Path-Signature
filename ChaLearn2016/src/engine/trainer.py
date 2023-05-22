import logging
from ignite.engine import Events,create_supervised_trainer,create_supervised_evaluator
from ignite.handlers import Timer,TerminateOnNan,ModelCheckpoint
from ignite.metrics import Loss,RunningAverage,Accuracy
import torch
import os
from tqdm import tqdm
import numpy as np
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torch.utils.tensorboard import SummaryWriter



class Save_Best_Checkpoint(object):
    def __init__(self,save_dir="",n_saved=2,mode="high",):
        self.n_saved = n_saved
        self.mode = mode
        self.score_list = []
        self.save_dir = save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def update_score_list(self,score):
        if len(self.score_list) < self.n_saved:
            self.score_list.append(score)
            self.score_list.sort()
            return True
        if self.mode == "low":
            for value in self.score_list:
                if score < value:
                    self.score_list.remove(value)
                    self.score_list.append(score)
                    self.score_list.sort(reverse=True)
                    return value
                else:
                    return False
        else:
            for value in self.score_list:
                if score > value:
                    self.score_list.remove(value)
                    self.score_list.append(score)
                    self.score_list.sort(reverse=False)
                    return value
                else:
                    return False                

    
    def save_checkpoint(self,cfg,model,score):
        model_name = os.path.join(self.save_dir,cfg['tag']+"_temp.pth")
        if cfg['multi_gpu']:
            save_pth = {'model':model.module.state_dict(),'cfg':cfg}
            torch.save(save_pth,model_name)
        else:
            save_pth = {'model':model.state_dict(),'cfg':cfg}
            torch.save(save_pth,model_name)

        pop_value = self.update_score_list(score)

        if pop_value is not False:
            model_name = os.path.join(self.save_dir,cfg['tag']+"_"+str(score)+".pth")
            if cfg['multi_gpu']:
                save_pth = {'model':model.module.state_dict(),'cfg':cfg}
                torch.save(save_pth,model_name)
            else:
                save_pth = {'model':model.state_dict(),'cfg':cfg}
                torch.save(save_pth,model_name)
                
            # import pdb; pdb.set_trace()
            if os.path.isfile(os.path.join(self.save_dir,cfg['tag']+"_"+str(pop_value)+".pth")):
                print(os.path.join(self.save_dir,cfg['tag']+"_"+str(pop_value)+".pth"))
                os.remove(os.path.join(self.save_dir,cfg['tag']+"_"+str(pop_value)+".pth"))



def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,metrics,device):

    def _prepare_batch(batch, device=None, non_blocking=False):
        """Prepare batch for training: pass to a device with options.

        """
        device = "cuda:" + str(device)
        x, y = batch
        x = convert_tensor(x,device=device,non_blocking=non_blocking)
        y = convert_tensor(y,device=device,non_blocking=non_blocking)
        return x,y


    def create_supervised_dp_trainer(model, optimizer,
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=lambda x, y, y_pred, loss: loss.item()):
        """
        Factory function for creating a trainer for supervised models.

        Args:
            model (`torch.nn.Module`): the model to train.
            optimizer (`torch.optim.Optimizer`): the optimizer to use.
            loss_fn (torch.nn loss function): the loss function to use.
            device (str, optional): device type specification (default: None).
                Applies to both model and batches.
            non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
                with respect to the host. For other cases, this argument has no effect.
            prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
                tuple of tensors `(batch_x, batch_y)`.
            output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
                to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

        Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
            of the processed batch by default.

        Returns:
            Engine: a trainer engine with supervised update function.
        """
        if device:
            model.to(device)

        def _update(engine, batch):
            model.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            # import pdb; pdb.set_trace()
            total_loss = model(x,y)
            total_loss = total_loss.mean() # model 里求均值
            total_loss.backward()
            optimizer.step()
            return output_transform(x, y, None, total_loss)

        return Engine(_update)



    master_device = device[0] #默认设置第一块为主卡
    trainer = create_supervised_dp_trainer(model,optimizer,device=master_device)
    trainer.add_event_handler(Events.ITERATION_COMPLETED,TerminateOnNan())
    RunningAverage(output_transform=lambda x:x).attach(trainer,'avg_loss')
    
    log_dir = cfg['log_dir']
    writer = SummaryWriter(log_dir=log_dir)

    # create pbar
    len_train_loader = len(train_loader)
    pbar = tqdm(total=len_train_loader)

    save_checkpoint = Save_Best_Checkpoint(save_dir=cfg['save_dir'],n_saved=cfg['n_save'])
    # froze_num_layers = cfg['warm_up']['froze_num_lyers']
    # if cfg['multi_gpu']:
    #     freeze_layers(model.module.encoder,froze_num_layers)
    # else:
    #     freeze_layers(model.encoder,froze_num_layers)


            
    ##########################################################################################
    ###########                    Events.ITERATION_COMPLETED                    #############
    ##########################################################################################

    # 每 log_period 轮迭代结束输出train_loss
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        log_period = cfg['log_period']
        log_per_iter = int(log_period*len_train_loader) if int(log_period*len_train_loader) >=1 else 1   # 计算打印周期
        current_iter = (engine.state.iteration-1)%len_train_loader + 1 + (engine.state.epoch-1)*len_train_loader # 计算当前 iter

        lr = optimizer.state_dict()['param_groups'][0]['lr']

        if current_iter % log_per_iter == 0:
            pbar.write("Epoch[{}] Iteration[{}] lr {:.7f} Loss {:.7f}".format(engine.state.epoch,current_iter,lr,engine.state.metrics['avg_loss']))
            pbar.update(log_per_iter)
            # writer.add_scalar('loss',engine.state.metrics['avg_loss'],current_iter)
    

    
    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_scheduler_epoch(engine):
        scheduler.EPOCH_COMPLETED()

    @trainer.on(Events.ITERATION_COMPLETED)
    def lr_scheduler_epoch(engine):
        scheduler.ITERATION_COMPLETED()

    ##########################################################################################
    ##################               Events.EPOCH_COMPLETED                    ###############
    ##########################################################################################
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_temp_epoch(engine):
        save_dir = cfg['save_dir']
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        epoch = engine.state.epoch
        if epoch%1==0:
            model_name=os.path.join(save_dir,cfg['tag'] +"_temp.pth")

            if cfg['multi_gpu']:
                save_pth = {'model':model.module.state_dict(),'cfg':cfg}
                torch.save(save_pth,model_name)
            else:
                save_pth = {'model':model.state_dict(),'cfg':cfg}
                torch.save(save_pth,model_name)
            
        # if epoch%cfg['save_period']==0:
        #     model_name=os.path.join(save_dir,cfg['tag'] +"_"+str(epoch)+".pth")
        #     if cfg['multi_gpu']:
        #         save_pth = {'model':model.module.state_dict(),'cfg':cfg}
        #         torch.save(save_pth,model_name)
        #     else:
        #         save_pth = {'model':model.state_dict(),'cfg':cfg}
        #         torch.save(save_pth,model_name)



    @trainer.on(Events.EPOCH_COMPLETED)
    def call_acc(engine):
        epoch = engine.state.epoch
        if epoch%cfg['save_period']==0:
            model.eval()
            num_correct = 0
            num_example = 0

            with torch.no_grad():
                for inputs,target in tqdm(val_loader):
                    inputs,target = inputs.to(master_device),target.to(master_device)
                    pred_logit = model(inputs,target)
                    indices = torch.max(pred_logit, dim=1)[1]
                    correct = torch.eq(indices, target).view(-1)
                    num_correct += torch.sum(correct).item()
                    num_example += correct.shape[0]

            acc = (num_correct/num_example)
            pbar.write("Acc: {}".format(acc))
            writer.add_scalar("Acc",acc,epoch)
            save_checkpoint.save_checkpoint(cfg,model,acc)
            model.train()

    
    @trainer.on(Events.EPOCH_COMPLETED)
    def reset_pbar(engine):
        pbar.reset()
    

    max_epochs = cfg['max_epochs']
    trainer.run(train_loader,max_epochs=max_epochs)
    pbar.close()   


def do_test(cfg,model,val_loader,metrics,device, top_k=5):
	


    master_device = device[0] #默认设置第一块为主卡
    
    if master_device:
        model.to(master_device)

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    num_params = count_params(model)

    model.eval()
    num_correct = 0
    num_example = 0

    num_correct_10 = 0

    with torch.no_grad():
        for inputs,target in tqdm(val_loader):
            inputs,target = inputs.to(master_device),target.to(master_device)
            pred_logit = model(inputs,target)
            indices = torch.max(pred_logit, dim=1)[1]
            indices_10 = torch.sort(pred_logit, dim=1)[1][:, -top_k:]
            correct = torch.eq(indices, target).view(-1)

            if len(target.shape) == 1:
                target = target.unsqueeze(dim=1).repeat((1, top_k))
            correct_10  = torch.eq(indices_10, target).view(-1)
            num_correct += torch.sum(correct).item()
            num_correct_10 += torch.sum(correct_10).item()
            num_example += correct.shape[0]

    acc = (num_correct/num_example)
    acc_10 = (num_correct_10/num_example)

    print("Top1 acc: %f" % acc)
    print("Top5 acc: %f" % acc_10)
    print("Params: %f" % num_params)


