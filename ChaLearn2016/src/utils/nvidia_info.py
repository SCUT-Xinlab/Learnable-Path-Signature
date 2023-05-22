'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-11 22:02:40
'''
import pynvml 

def get_free_device_ids(threshold=0.3):
    pynvml.nvmlInit()
    num_device = pynvml.nvmlDeviceGetCount()
    free_device_id = []
    for i in range(num_device):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        men_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(men_info.total,men_info.free)
        # import pdb; pdb.set_trace()
        if men_info.free >= men_info.total*threshold:
            free_device_id.append(i)
    return free_device_id


if __name__ == "__main__":
    print(get_free_device_ids())
    import pdb; pdb.set_trace()