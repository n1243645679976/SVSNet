import os
import shutil 

import numpy as np
import scipy.stats as stats

from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

class Logger(object):
    def __init__(self, exp_dir, show_result_only=False):
        self.log_file = os.path.join(exp_dir, 'log.txt')
        self.output_dir = os.path.join(exp_dir, 'output')
        self.models_dir = os.path.join(exp_dir, 'models')
        self.mode = 'train'
        self.exp_dir = exp_dir
        self.log = defaultdict(list)
        self.best_epoch = -1
        self.best_criterion = float('-inf')

        if not show_result_only:
            if os.path.isfile(self.log_file):
                print(f'{self.log_file} exists! Backup to {self.log_file}.backup')
                os.rename(f'{self.log_file}', f'{self.log_file}.backup')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def reset_log(self):
        for log in ['name', 'loss', 'label', 'output']:
            self.log[log] = []

    def change_mode(self, mode):
        self.mode = mode

    def append(self, t, it, loss, output, label, name):
        if it % 100 == 0:
            print('t:{} it: {}, loss: {:.5f}                  \r'.format(t, it, loss))
        _loss = loss.cpu().detach().numpy()
        _output = output.reshape(-1).cpu().detach().numpy()
        _label = label.reshape(-1).long().cpu().detach().numpy()
        self.log['loss'].append(np.mean(_loss))
        for i in range(len(_output)):
            self.log['label'].append(_label[i])
            self.log['output'].append(_output[i])
            self.log['name'].append(name[i])
    
    def metric(self, label, output):
        acc = np.mean(label == output)
        lcc = np.corrcoef(label, output)[0][1]
        srcc = stats.spearmanr(label, output)[0]
        mse = np.mean((label - output) ** 2)
        return acc, lcc, srcc, mse

    def log(self, ep, lr):
        output_file = os.path.join(self.output_dir, f'out.{ep}.txt')
        with open(output_file, 'w+') as out, open(self.log_file, 'a+') as log:
            for name, label, output in zip(self.log['name'], self.log['label'], self.log['output']):
                out.write(f'{name},{label},{output}\n')

            loss = np.mean(self.log['loss'])
            time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            acc, lcc, srcc, mse = self.metric(np.array(self.log['label']), np.array(self.log['output']))
            log.write(f'{ep=}, {lr=:.5f}, {self.mode=}, {acc=:.5f}, {loss=}, {time=}, {lcc=:.5f}, {srcc=:.5f}, {mse=:.5f}\n')

        self.log['criterion'].append((self.criterion((acc, lcc, srcc, mse)), ep))
    
    def get_best_epoch(self):
        best_criterion, best_epoch = max(self.log['criterion'])
        best_output_file = os.path.join(self.exp_dir, f'out.{best_epoch}.txt')
        best_model_file = os.path.join(self.exp_dir, f'SVSNet.ep.{best_epoch:0>5d}.pt')
        save_best_output = os.path.join(self.exp_dir, f'best_output.txt')
        save_best_model = os.path.join(self.exp_dir, 'best_model.pt')
        shutil.copy(best_output_file, save_best_output)
        shutil.copy(best_model_file, save_best_model)

    def show_result(self):
        save_best_output = os.path.join(self.exp_dir, f'best_output.txt')
        pair_dict = {'label':defaultdict(list), 'output':defaultdict(list)}
        sys_dict = {'label':defaultdict(list), 'output':defaultdict(list)}
        with open(save_best_output) as f:
            for line in f.read().splitlines():
                pair, label, output = line.split(',')
                utt_a, utt_b = pair.split()
                if utt_a[0] + utt_b[0] in ['SS', 'ST', 'TS', 'SS']:
                    pass
                elif 'S00' in utt_a or 'T00' in utt_a:
                    utt_a, utt_b = utt_b, utt_a
                pair_key = utt_a + utt_b
                pair_dict['label'][pair_key].append(float(label))
                pair_dict['output'][pair_key].append(float(output))

                sys_key = utt_a[:3]
                sys_dict['label'][sys_key].append(float(label))
                sys_dict['output'][sys_key].append(float(output))
        
        pair_keys = pair_dict['label'].keys()
        pair_label = [np.mean(pair_dict['label'][key]) for key in pair_keys]
        pair_output = [np.mean(pair_dict['output'][key]) for key in pair_keys]
        acc, lcc, srcc, mse = self.metric(pair_label, pair_output)
        print(f'best result:')
        print(f'  pair:\t{acc=},\t{lcc=},\t{srcc=},\t{mse=}')

        sys_keys = sys_dict['label'].keys()
        sys_label = [np.mean(pair_dict['label'][key]) for key in sys_keys]
        sys_output = [np.mean(pair_dict['output'][key]) for key in sys_keys]
        acc, lcc, srcc, mse = self.metric(sys_label, sys_output)
        print(f'  system:\t{acc=},\t{lcc=},\t{srcc=},\t{mse=}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp-dir', default='result/SVSNet', help='result folder, used for saving models and results', required=True)
    args = parser.parse_args() 
    Logger(exp_dir=args.exp_dir, show_result_only=True)
        

                

