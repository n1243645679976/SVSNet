# -*- coding: utf-8 -*-
import os
import torch
import random

import numpy as np
import torch.backends.cudnn as cudnn

from argparse import ArgumentParser

from logger import Logger
from dataset import Dataset
from torch.utils.data import DataLoader

print('#################')
seed = 1827
batch_size = 5

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
          
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--continue-from', default=None, help='model to continue training, if not specified, then continue from the last checkpoint in exp_dir') 
    parser.add_argument('--model-type', default='regression', choices=['regression', 'classification'], help="model type to train, 'regression' or 'classification'")
    parser.add_argument('--testing', action='store_true', help='list for training')
    parser.add_argument('--train-list', default='../../feature/sim_net/sim_list_train.txt', help='list for training', required=True)
    parser.add_argument('--test-list', default='../../feature/sim_net/sim_list_test.txt', help='list for testing', required=True)
    parser.add_argument('--feat-path', default='../../feature/sim_net/wav', help='the path where reposit features', required=True)
    parser.add_argument('--exp-dir', default='result/SVSNet', help='result folder, used for saving models and results')
    parser.add_argument('--epoch', default=30, type=int, help='max epoch for training')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'], help="training device, 'cuda' or 'cpu'")
    parser.add_argument('--batch-size', default=5, type=int, help="batch size")
    args = parser.parse_args()

    if args.model_type == 'regression':
        from model.lib_resnet import regression_model as model_class
    elif args.model_type == 'classification':
        from model.lib_resnet import classification_model as model_class

    model = model_class(device=args.device)
    print(model)
    print(f'experiment folder: {args.exp_dir}')
    start_epoch = 0
    dataset = Dataset(testing=args.testing, feat_path=args.feat_path, test_list=args.test_list, train_list=args.train_list, device=args.device)
    cofn = lambda x: ([x_ for x_, _, _ in x], torch.cat([y_ for x_, y_, _ in x]), [z_ for x_, y_, z_ in x])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=cofn, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=0, verbose=False, 
        threshold=1e-3, threshold_mode="rel", cooldown=5, min_lr=1e-4, eps=1e-08,
    )

    logger = Logger(args.model, args.comment)
    if args.continue_from == None:
        if os.path.isdir(os.path.join(args.exp_dir, 'models')):
            models = os.listdir(os.path.join(args.exp_dir, 'models'))
            if len(models) > 0:
                args.continue_from = os.path.join(args.exp_dir, 'models', sorted(models)[-1])

    if args.continue_from is not None:
        print(f'resume from {args.continue_from}')
        model_ = torch.load(args.continue_from)
        model.load_state_dict(model_['model'])
        optimizer.load_state_dict(model_['optimizer'])
        lr_scheduler.load_state_dict(model_['scheduler'])
        start_epoch = model_['epoch']
        logger = model_['logger']
        del model_
    
    if args.testing:
        args.epoch = start_epoch + 1

    for i in range(start_epoch, args.epoch):
        if not args.testing:
            model.train()
            logger.change_mode('train')
            dataset.change_mode('train')
            for t, (x, label, z) in enumerate(dataloader):
                loss, y =  model(x, label)
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                logger.append(i, t, loss, y, label, z)

            logger.log(i, lr_scheduler.optimizer.param_groups[0]['lr'])
            logger.reset_log()

        # evaluation
        model.eval()
        logger.change_mode('test')
        dataset.change_mode('test')
        for t, (x, label, z) in enumerate(dataloader):
            loss, y = model.predict(x, label)
            logger.append(i, t, loss, y, label, z)
        logger.log(i, lr_scheduler.optimizer.param_groups[0]['lr'])
        logger.reset_log()
        
        if args.testing:
            break
        
        lr_scheduler.step(logger.log['loss'][-1])
        # back to train
        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler':lr_scheduler.state_dict(), 'logger':logger, 'epoch':i},
                    os.path.join(os.path.join(args.exp_dir, f'models/SVSNet.ep.{i:0>5d}.pt').format(i)))

    if not args.testing:
        logger.get_best_epoch()
        logger.show_result()
    else:
        logger.show_result()
