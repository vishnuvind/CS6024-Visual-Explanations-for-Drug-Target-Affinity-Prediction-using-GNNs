import os
import numpy as np
import torch

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

def load_checkpoint(model_path):
    return torch.load(model_path)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    # print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def load_drug_target_models(model_d, model_t, saved_model):
    try:
        load_model_dict(model_t, saved_model)
        load_model_dict(model_d, saved_model)
    except:
        model_dict = torch.load(saved_model)
        for key, val in model_dict.copy().items():
            if 'lin_l' in key:
                new_key = key.replace('lin_l', 'lin_rel')
                model_dict[new_key] = model_dict.pop(key)
            elif 'lin_r' in key:
                new_key = key.replace('lin_r', 'lin_root')
                model_dict[new_key] = model_dict.pop(key)
        model_t.load_state_dict(model_dict)
        model_d.load_state_dict(model_dict)
    return model_d, model_t

def load_module(model, module_list):
    if module_list is None:
        raise Exception('Module List must not be empty!')
    elif len(module_list) == 0:
        raise Exception('Module List must not be empty!')
    
    module = model._modules[module_list[0]]
    for i in range(1, len(module_list)):
        module = module._modules[module_list[i]]
    return module

def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x

def mask_drug(attention, threshold, device, soft_thresh, keep_imp = True):
    num_mols = len(attention)
    att_mask = torch.Tensor(attention).reshape((num_mols, 1)).broadcast_to((num_mols, 18))
    if soft_thresh:
        if keep_imp:
            return (att_mask).to(device)
        else:
            return (1.0 - att_mask).to(device)
    else:
        if keep_imp:
            return (att_mask >= threshold).to(device)
        else:
            return (att_mask < threshold).to(device)
    
def mask_targ(attention, threshold, device, soft_thresh, keep_imp = True):
    num_mols = len(attention)
    att_mask = torch.Tensor(attention).reshape((1, num_mols))
    if soft_thresh:
        if keep_imp:
            return (att_mask).to(device)
        else:
            return (1.0 - att_mask).to(device)
    else:
        if keep_imp:
            return (att_mask >= threshold).to(device)
        else:
            return (att_mask < threshold).to(device)