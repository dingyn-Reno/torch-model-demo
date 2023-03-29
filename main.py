#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
from torchlight import DictAction
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
from send_mail import send
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=16,
        help='the number of worker for data loader')

    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--momentum',type=float,default=0)
    parser.add_argument('--clip',type=float,default=1000)
    parser.add_argument('--grad_norm', type=str2bool, default=False)
    parser.add_argument('--compile', type=str2bool, default=False)
    parser.add_argument('--compile_mode',  default='default')
    parser.add_argument('--AMP', type=str2bool, default=True)
    parser.add_argument('--AMP_scaler', type=str2bool, default=False)
    parser.add_argument('--to_onnx',type=str2bool, default=False)
    return parser


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train' or arg.phase =='train_only':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

# 1 数据读取部分，编写feeder文件夹中的数据处理代码，并修改dataset参数设置
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train' or self.arg.phase=='train_only':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

# 2 模型读取部分，编写model文件夹中的模型代码，修改Model函数参数
    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        # 模型参数配置位置
        self.model = Model(**self.arg.model_args)
        if self.arg.compile==True:
            self.model=torch.compile(self.model,mode=self.arg.compile_mode)
        print(self.model)

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

# 3 优化器配置，如果使用需要更新参数的loss，需要把loss置入损失函数
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=self.arg.momentum,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    # 设置损失函数
    def set_loss(self):
        self.loss = nn.CrossEntropyLoss().cuda(self.output_device)

    # 计算准确率
    def get_acc(self,output,label):
        value, predict_label = torch.max(output.data, 1)
        acc = torch.mean((predict_label == label.data).float())
        return acc

    def train(self, epoch, save_model=False,grad_norm=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)
        scaler = GradScaler()
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            if self.arg.AMP==True:
                with autocast():
                    output = self.model(data)
                    loss = self.loss(output, label)
            else:
                output = self.model(data)
                loss = self.loss(output, label)
            self.optimizer.zero_grad()
            if self.arg.AMP_scaler==True:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if grad_norm==True:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.arg.clip)
            if self.arg.AMP_scaler==True:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()
            acc=self.get_acc(output,label)
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')


    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            acc_value=[]
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    score_frag.append(output.data.cpu().numpy())
                    loss = self.loss(output, label)
                    acc = self.get_acc(output, label)
                    acc_value.append(acc.data.item())
                    loss_value.append(loss.data.item())
                    step += 1

            loss = np.mean(loss_value)
            accuracy=np.mean(acc_value)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            self.print_log('\tAccuracy: {}'.format(accuracy))
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            # if save_score:
            #     score_pth=np.concatenate(score_frag,axis=0)
            #     print(score_pth.shape)
            #     torch.save(score_pth, 'test.pth')
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))

    # 设置数据保存
    def save_eval(self):
        pass
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            self.set_loss()
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model, grad_norm=self.arg.grad_norm)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'-*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')
            body = "你的模型已经跑完，精度为{}".format(self.best_acc)
            send(body)
        elif self.arg.phase == 'test':
            self.set_loss()
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.\n')
        elif self.arg.phase == 'train_only':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            self.set_loss()
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model, grad_norm=self.arg.grad_norm)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
