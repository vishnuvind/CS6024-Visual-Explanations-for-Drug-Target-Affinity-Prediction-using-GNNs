import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import torch
import cairosvg
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from IPython.display import SVG
from matplotlib.colors import ListedColormap

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)

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
    
def mask_targ(attention, threshold, device, soft_thresh, keep_imp):
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
        
def clourMol(mol,highlightAtoms_p=None,highlightAtomColors_p=None,highlightBonds_p=None,highlightBondColors_p=None,sz=[400,400], radii=None):
    d2d = rdMolDraw2D.MolDraw2DSVG(sz[0], sz[1])
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 40
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p,highlightAtomColors=highlightAtomColors_p, highlightBonds= highlightBonds_p,highlightBondColors=highlightBondColors_p, highlightAtomRadii=radii)
    d2d.FinishDrawing()
    svg = SVG(d2d.GetDrawingText())
    res = cairosvg.svg2png(svg.data, dpi = 600, output_width=2400, output_height=2400)
    nparr = np.frombuffer(res, dtype=np.uint8)
    segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return segment_data

def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def best_pos(att, seqlen, window=9):
    seqlen = len(att)
    max = -np.inf
    pos = None
    for i in range(seqlen-window):
        poss = i + window//2
        summ = 0
        for j in range(i, i+window+1):
            gauss = gaussian(j, poss, window/6)
            summ += att[j]*gauss
        if summ>max:
            pos = poss
            max = summ
    return pos

def get_colormap(bottom, top):
    newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), top(np.linspace(0.15, 0.65, 128))])
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    return newcmp

def save_atom_activation_map(idx, smile, attention, cmap, results_path, model_name):
    mol = Chem.MolFromSmiles(smile)
    atom_color = dict([(idx, cmap(attention[idx])[:3]) for idx in range(len(attention))])
    radii = dict([(idx, 0.2) for idx in range(len(attention))])
    img = clourMol(mol,highlightAtoms_p=range(len(attention)), highlightAtomColors_p=atom_color, radii=radii)
    results_path = os.path.join(results_path, os.path.join(model_name, 'drug'))
    cv2.imwrite(os.path.join(results_path, f'{idx}.png'), img)

def save_protein_activation_map(idx, sequence, attention, cmap, results_path, model_name):
    seqlen=len(attention)

    m = 6
    p = seqlen/m
    fig, ax = plt.subplots()
    Z = np.zeros((200, seqlen))
    for i in range(len(attention)):
        Z[:50, i] = np.ones(50)*attention[i]
    extent = (0, m, 0, 2)
    ax.imshow(Z, extent=extent, origin="lower")

    # # inset axes....
    pos = best_pos(attention, seqlen)/p
    left, right = max(0, pos-25/p), min(m, pos+25/p)
    if left==0.0: right = 50/p
    if right==m: left = m-50/p
    x1, x2, y1, y2 = left, right, 0.2375, 0.2625  # subregion of the original image
    axins = ax.inset_axes(
        [0, 1, m, 1],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], xticks=[], yticks=[], yticklabels=[], transform=ax.transData)
    axins.imshow(Z, extent=extent, origin="lower")
    ax.axis('off')
    ax.indicate_inset_zoom(axins, edgecolor="black")
    for i, s in enumerate(sequence[int(p*left):int(p*right)]):
        axins.text(left+i/p, 0.25, s, fontsize = 4)

    results_path = os.path.join(results_path, os.path.join(model_name, 'target'))
    plt.savefig(os.path.join(results_path, f'{idx}.png'), dpi=360, bbox_inches='tight')