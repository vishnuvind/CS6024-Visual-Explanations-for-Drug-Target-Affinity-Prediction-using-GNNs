import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import cairosvg
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('tkagg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import SVG
from matplotlib.colors import ListedColormap

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)

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
    # print(seqlen)

    m = 6
    p = seqlen/m
    fig, ax = plt.subplots()
    Z = np.zeros((200, seqlen))
    for i in range(len(attention)):
        Z[:50, i] = np.ones(50)*attention[i]
    Z2 = Z[:, seqlen//2 - 25: seqlen//2 + 25]
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
    plt.savefig(os.path.join(results_path, f'{idx}.png'), dpi=360)