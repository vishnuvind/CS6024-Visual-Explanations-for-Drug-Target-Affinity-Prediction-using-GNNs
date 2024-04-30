# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
import torch.nn.functional as F
import argparse

from metrics import get_cindex
from dataset import *
from model import GraphDTA_GCN, GraphDTA_GIN, GraphDTA_GAT
from utils import *
from log.train_logger import TrainLogger

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

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()
    model.train()

    return epoch_loss

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', default='davis', help='davis or kiba')
    parser.add_argument('--model', type=str, default='mgraphdta', help='graphdta vs mgraphdta')
    parser.add_argument('--save_model', type=bool, default=True, help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        model = args.model,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    model_name = params.get("model")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    batch_size = params.get("batch_size")
    lr = params.get('lr')
    fpath = os.path.join(data_root, DATASET)

    train_set = GNNDataset(fpath, train=True)
    test_set = GNNDataset(fpath, train=False)
    logger.info('%s Dataset loaded successfully!' % ('DAVIS' if args.dataset=='davis' else 'KIBA'))

    logger.info('Train Length: %d' % (len(train_set)))
    logger.info('Test Length: %d' % (len(test_set)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    # load DTA models
    if args.model in _MODELS:
        model = _MODELS[args.model].to(device)
        logger.info(f'{_MODEL_NAME[args.model]} model loaded successfully!')
    else:
        raise Exception("Please use a supported model")

    epochs = 500
    steps_per_epoch = 50 if args.dataset=='davis' else 250
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 400

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()
    logger.info('Training for %d epochs (%d steps) begins...' % (epochs, num_iter))

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
                              
            global_step += 1       
            data = data.to(device)
            pred = model(data)


            loss = criterion(pred.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0)) 
            running_cindex.update(cindex, data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                test_loss = val(model, criterion, test_loader, device)

                msg = "model-%s, dataset-%s, epoch-%d, loss-%.4f, cindex-%.4f, test_loss-%.4f" % (model_name, DATASET, global_epoch, epoch_loss, epoch_cindex, test_loss)
                logger.info(msg)

                if test_loss < running_best_mse.get_best():
                    running_best_mse.update(test_loss)
                    if save_model:
                        logger.info('Saving model...')
                        save_model_dict(model, logger.get_model_dir(), msg)
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break

if __name__ == "__main__":
    main()
