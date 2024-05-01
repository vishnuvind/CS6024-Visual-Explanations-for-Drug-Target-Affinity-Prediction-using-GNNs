import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import matplotlib
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import summary
from torch_geometric.data import Batch

from utils import *
from dataset import *
from metrics import *
from explainers import GradAAM, GradTAM
from log.explain_logger import ExplainLogger
from model import  GraphDTA_GCN, GraphDTA_GIN, GraphDTA_GAT

_MODEL_NAME = {
    'mgraphdta': 'MGraphDTA',
    'gcn': 'GraphDTA + GCN',
    'gin': 'GraphDTA + GIN',
    'gat': 'GraphDTA + GAT',
}

_MODELS = {
    # 'mgraphdta': MGraphDTA(1, 25 + 1, embedding_size=128, filter_num=32, out_dim=1),
    'gcn': GraphDTA_GCN(1, 32, embed_dim=128, num_features_xd=78, num_features_xt=25),
    'gin': GraphDTA_GIN(1, 32, embed_dim=128, num_features_xd=78, num_features_xt=25),
    'gat': GraphDTA_GAT(1, 32, embed_dim=128, num_features_xd=78, num_features_xt=25),
}

_TARGET_LAYERS = {
    'mgraphdta': None,
    'gcn': {
        'drug': ['features_drug', 'Encoder1'],
        'targ': ['features_target', 'Encoder2'],
    },
    'gin': {
        'drug': ['features_drug', 'Encoder1'],
        'targ': ['features_target', 'Encoder2'],
    },
    'gat': {
        'drug': ['features_drug', 'Encoder1'],
        'targ': ['features_target', 'Encoder2'],
    }
}

def main():
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='davis', help='davis or kiba')
    parser.add_argument('--model', type=str, default='gcn', help='DTA prediction model')
    parser.add_argument('--saved_model', type=str, help='load saved model')
    parser.add_argument('--thresh', type=float, default=None, help='attention threshold used to create masks for evaluating explanations')
    parser.add_argument('--plot_stuff', action='store_true', help='whether or not to plot drug target activation maps')
    parser.add_argument('--mask_drug', action='store_true', help='whether or not to mask drug')
    parser.add_argument('--mask_targ', action='store_true', help='whether or not to mask target')
    args = parser.parse_args()

    # get results path
    results_path = os.path.join('./results/', args.saved_model.split('/')[2])
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    # Setup logger
    params = dict(
        save_dir=results_path,
        saved_model=args.saved_model.split('/')[4],
    )
    logger = ExplainLogger(params)
    logger.info(__file__)

    # load test dataset
    device = torch.device('cuda:0')
    fpath = os.path.join('data', args.dataset)
    test_df = pd.read_csv(os.path.join(fpath, 'raw', 'data_test.csv'))
    test_set = GNNDataset(fpath, train=False)

    # load DTA models
    if args.model in _MODELS:
        model_d = _MODELS[args.model].to(device)
        model_t = _MODELS[args.model].to(device)
    else:
        raise Exception("Please use a supported model")

    # load saved checkpoints
    model_d, model_t = load_drug_target_models(model_d, model_t, args.saved_model)
    logger.info(f'{_MODEL_NAME[args.model]} model loaded successfully!')

    # Load Explainer (GradAAM & GradTAM)
    if args.model not in _TARGET_LAYERS:
        raise Exception("Please use a supported model")
    else:
        explain_drug = GradAAM(model_d, load_module(model_d, _TARGET_LAYERS[args.model]['drug']))
        explain_targ = GradTAM(model_t, load_module(model_t, _TARGET_LAYERS[args.model]['targ'])) 
    
    # setup smile list, colormap and threshold for activation map evaluation
    # newcmp = get_colormap(bottom = cm.get_cmap('Blues_r', 256), top = cm.get_cmap('Oranges', 256))
    newcmp = get_colormap(bottom = matplotlib.colormaps['Blues_r'], top = matplotlib.colormaps['Oranges'])
    # newcmp = matplotlib.colormaps['viridis']
    drug_list = list(test_df['Drug'].unique())
    targ_list = list(test_df['Target'].unique())

    # collect evaluation metrics
    criterion = nn.MSELoss()
    running_loss_imp = AverageMeter()
    running_loss_res = AverageMeter()
    att_list = {'drug': [], 'targ': []}
    pred_list = {'y': [], 'imp': [], 'res': []}
    sparsity_list = {'drug': [], 'targ': []}
    threshlist = {'drug': {'mean': [], 'quant': []}, 'targ': {'mean': [], 'quant': []}}

    # iterate over test set
    for idx in tqdm(range(len(test_set))):
        model_d.train()
        model_t.train()
        targ = test_df.iloc[idx]['Target']
        data1 = Batch.from_data_list([test_set[idx]]).to(device)
        data2 = Batch.from_data_list([test_set[idx]]).to(device)

        # get drug and target explanations
        _, drug_att = explain_drug(data1)
        _, targ_att = explain_targ(data2)

        att_list['drug'].append(drug_att)
        att_list['targ'].append(targ_att)
        threshlist['drug']['mean'].append(np.mean(drug_att))
        threshlist['targ']['mean'].append(np.mean(targ_att[:len(targ)+1]))
        if args.thresh is not None and args.thresh>=0.0:
            threshlist['drug']['quant'].append(np.quantile(drug_att, args.thresh))
            threshlist['targ']['quant'].append(np.quantile(targ_att[:len(targ)+1], args.thresh))
        # if idx==100: break

    if args.thresh is None:
        soft_thresh = True
        drug_thresh = -1.0
        targ_thresh = -1.0
        logger.info('No thresholding! Soft masking of input!')
    elif args.thresh >= 0.0:
        soft_thresh = False
        drug_thresh = np.mean(threshlist['drug']['quant'])
        targ_thresh = np.mean(threshlist['targ']['quant'])
        logger.info('Thresholding at the %d-th quantile!' % (args.thresh*100))
        logger.info('Saliency threshold (Drug): %.3f' % (drug_thresh))
        logger.info('Saliency threshold (Targ): %.3f' % (targ_thresh))
    else: 
        soft_thresh = False
        drug_thresh = np.mean(threshlist['drug']['mean'])
        targ_thresh = np.mean(threshlist['targ']['mean'])
        logger.info('Thresholding at the mean activation!')
        logger.info('Saliency threshold (Drug): %.3f' % (drug_thresh))
        logger.info('Saliency threshold (Targ): %.3f' % (targ_thresh))

    # iterate over test set again
    for idx in tqdm(range(len(test_set))):
        drug = test_df.iloc[idx]['Drug']
        targ = test_df.iloc[idx]['Target']

        # get activation map and calculate sparsity
        drug_att = np.array(att_list['drug'][idx])
        targ_att = np.array(att_list['targ'][idx])
        sparsity_list['drug'].append(sparsity(drug_att, drug_thresh))
        sparsity_list['targ'].append(sparsity(targ_att[:len(targ)+1], targ_thresh))

        if args.plot_stuff:
            # if drug activation map hasn't been considered yet, output it
            if drug in drug_list:
                save_atom_activation_map(idx, drug, drug_att, newcmp, results_path, args.saved_model.split('/')[4])
                drug_list.remove(drug)

            # if target activation map hasn't been considered yet, output it
            if targ in targ_list and idx<100:
                save_protein_activation_map(idx, targ, targ_att[:len(targ)+1], newcmp, results_path, args.saved_model.split('/')[4])
                targ_list.remove(targ)
        
        # create masked drug and target data
        # data_imp masks all non salient features in input
        # data_res masks all salient features in input
        data_imp = Batch.from_data_list([test_set[idx]]).to(device)
        data_res = Batch.from_data_list([test_set[idx]]).to(device)
        if args.mask_drug:
            if idx == 0: logger.info('Masking drugs...')
            drug_mask_imp = mask_drug(drug_att, drug_thresh, device, soft_thresh, True)
            drug_mask_res = mask_drug(drug_att, drug_thresh, device, soft_thresh, False)
            data_imp.x = torch.mul(data_imp.x, drug_mask_imp)
            data_res.x = torch.mul(data_res.x, drug_mask_res)

        if args.mask_targ:
            if idx == 0: logger.info('Masking targets...')
            targ_mask_imp = mask_targ(targ_att, targ_thresh, device, soft_thresh, True)
            targ_mask_res = mask_targ(targ_att, targ_thresh, device, soft_thresh, False)
            data_imp.target = torch.mul(data_imp.target, targ_mask_imp)
            data_res.target = torch.mul(data_res.target, targ_mask_res)
        
        # calculate for data_imp
        model_d.eval()
        model_t.eval()
        with torch.no_grad():
            pred_imp = model_d(data_imp)
            pred_res = model_d(data_res)
            pred_list['y'].append(data_imp.y.detach().cpu().numpy().reshape(-1))
            pred_list['imp'].append(pred_imp.detach().cpu().numpy().reshape(-1))
            pred_list['res'].append(pred_res.detach().cpu().numpy().reshape(-1))
            loss_imp = criterion(pred_imp.view(-1), data_imp.y.view(-1))
            loss_res = criterion(pred_res.view(-1), data_res.y.view(-1))
            running_loss_imp.update(loss_imp.item())
            running_loss_res.update(loss_res.item())
        # if idx == 100: break  

    r2_imp = get_r2_score(pred_list['y'], pred_list['imp'])
    r2_res = get_r2_score(pred_list['y'], pred_list['res'])
    cindex_imp = get_cindex(np.array(pred_list['y']), np.array(pred_list['imp']))
    cindex_res = get_cindex(np.array(pred_list['y']), np.array(pred_list['res']))

    logger.info('C-Index with only Salient regions: %.5f' % (cindex_imp))
    logger.info('C-Index excluding Salient regions: %.5f' % (cindex_res))
    logger.info('R2 Score with only Salient regions: %.5f' % (r2_imp))
    logger.info('R2 Score excluding Salient regions: %.5f' % (r2_res))
    logger.info('MSE Loss with only Salient regions: %.5f' % (running_loss_imp.get_average()))
    logger.info('MSE Loss excluding Salient regions: %.5f' % (running_loss_res.get_average()))
    logger.info('Drug Explanation Sparsity: Mean = %.5f Std = %.5f' % (np.mean(sparsity_list['drug']), np.std(sparsity_list['drug'])))
    logger.info('Targ Explanation Sparsity: Mean = %.5f Std = %.5f' % (np.mean(sparsity_list['targ']), np.std(sparsity_list['targ'])))
    logger.info('COMPLETE!\n')

if __name__ == '__main__':
    main()

