'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-17 05:59:18
'''
import torch

# 自定义 batch_data 预处理函数
def my_collate_fn(batch_list):
    image_list = list()
    label_list = list()
    for i in range(len(batch_list)):
        image,labels = batch_list[i]
        image_list.append(image.squeeze(0))
        label_list.append(labels.squeeze(0))
    return torch.cat(image_list,dim=0),torch.cat(label_list,dim=0)

# def concat(batch_list):
#     batch_images,batch_K = list(),list()
#     batch_masks,batch_mano,batch_K,batch_scale = list(),list(),list(),list()
#     batch_verts,batch_xyz,batch_xy,batch_idx = list(),list(),list(),list()
#     for i in range(len(batch_list)):
#         inputs,targets = batch_list[i]
#         batch_images.append(inputs[0])

#         batch_masks.append(targets[0])
#         batch_mano.append(targets[1])
#         batch_K.append(targets[2])
#         batch_scale.append(targets[3])
#         batch_verts.append(targets[4])
#         batch_xyz.append(targets[5])
#         batch_xy.append(targets[6])
#         batch_idx.append(targets[7])
#     batch_images = torch.cat(batch_images,dim=0)
#     batch_masks = torch.cat(batch_masks,dim=0)
#     batch_mano = torch.cat(batch_mano,dim=0)
#     batch_K = torch.cat(batch_K,dim=0)
#     batch_scale = torch.cat(batch_scale,dim=0)
#     batch_verts = torch.cat(batch_verts,dim=0)
#     batch_xyz = torch.cat(batch_xyz,dim=0)
#     batch_xy = torch.cat(batch_xy,dim=0)
#     batch_idx = torch.cat(batch_idx,dim=0)
#     return [batch_images,batch_K],[batch_masks,batch_mano,batch_K,batch_scale,batch_verts,batch_xyz,batch_xy,batch_idx]