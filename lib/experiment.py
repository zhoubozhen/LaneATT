import os
import re
import json
import logging
import subprocess

import torch
from torch.utils.tensorboard import SummaryWriter


class Experiment:
    def __init__(self, exp_name, args=None, mode='train', exps_basedir='experiments', tensorboard_dir='tensorboard'):
        self.name = exp_name
        self.exp_dirpath = os.path.join(exps_basedir, exp_name)
        self.models_dirpath = os.path.join(self.exp_dirpath, 'models')
        self.results_dirpath = os.path.join(self.exp_dirpath, 'results')
        self.cfg_path = os.path.join(self.exp_dirpath, 'config.yaml')
        self.code_state_path = os.path.join(self.exp_dirpath, 'code_state.txt')
        self.log_path = os.path.join(self.exp_dirpath, 'log_{}.txt'.format(mode))
        self.tensorboard_writer = SummaryWriter(os.path.join(tensorboard_dir, exp_name))
        self.cfg = None
        self.setup_exp_dir()
        self.setup_logging()

        if args is not None:
            self.log_args(args)

    def save_model(self, model, optimizer, scheduler, epoch, tag="best"):
        """Save model checkpoint with tag (e.g. 'best' or 'latest')."""
        import os
        import torch

        # 构造保存路径
        models_dir = os.path.join("experiments", getattr(self, "name", None) or getattr(self, "exp_name", "default"), "models")
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"model_{tag}.pt")

        # 保存状态
        torch.save({
            "epoch": epoch,
            "best_acc": getattr(self, "best_acc", 0.0),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, path)

        if hasattr(self, "logger"):
            self.logger.info(f"💾 Saved checkpoint: {path}")
        else:
            print(f"💾 Saved checkpoint: {path}")

        return path


    def get_output_dir(self):
        """保持兼容性: 旧版本用 get_output_basedir，新版本用 get_output_dir"""
        import os

        # 优先兼容旧函数
        if hasattr(self, "get_output_basedir"):
            return self.get_output_basedir()
        elif hasattr(self, "output_basedir"):
            return self.output_basedir

        # 尝试自动识别 exp_name / name
        exp_name = getattr(self, "exp_name", None) or getattr(self, "name", None)
        if exp_name is None:
            raise AttributeError("Experiment has neither exp_name nor name field.")
        return os.path.join("experiments", exp_name)



    def setup_exp_dir(self):
        if not os.path.exists(self.exp_dirpath):
            os.makedirs(self.exp_dirpath)
            os.makedirs(self.models_dirpath)
            os.makedirs(self.results_dirpath)
            self.save_code_state()

    def save_code_state(self):
        state = "Git hash: {}".format(
            subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, check=False).stdout.decode('utf-8'))
        state += '\n*************\nGit diff:\n*************\n'
        state += subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, check=False).stdout.decode('utf-8')
        with open(self.code_state_path, 'w') as code_state_file:
            code_state_file.write(state)

    def setup_logging(self):
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])
        self.logger = logging.getLogger(__name__)

    def log_args(self, args):
        self.logger.debug('CLI Args:\n %s', str(args))

    def set_cfg(self, cfg, override=False):
        assert 'model_checkpoint_interval' in cfg
        self.cfg = cfg
        if not os.path.exists(self.cfg_path) or override:
            with open(self.cfg_path, 'w') as cfg_file:
                cfg_file.write(str(cfg))

    def get_last_checkpoint_epoch(self):
        pattern = re.compile('model_(\\d+).pt')
        last_epoch = -1
        for ckpt_file in os.listdir(self.models_dirpath):
            result = pattern.match(ckpt_file)
            if result is not None:
                epoch = int(result.groups()[0])
                if epoch > last_epoch:
                    last_epoch = epoch

        return last_epoch

    def get_checkpoint_path(self, epoch):
        return os.path.join(self.models_dirpath, 'model_{:04d}.pt'.format(epoch))

    def get_epoch_model(self, epoch):
        return torch.load(self.get_checkpoint_path(epoch))['model']

    def load_last_train_state(self, model, optimizer, scheduler):
        epoch = self.get_last_checkpoint_epoch()
        train_state_path = self.get_checkpoint_path(epoch)
        train_state = torch.load(train_state_path)
        model.load_state_dict(train_state['model'])
        optimizer.load_state_dict(train_state['optimizer'])
        scheduler.load_state_dict(train_state['scheduler'])

        return epoch, model, optimizer, scheduler

    def save_train_state(self, epoch, model, optimizer, scheduler):
        train_state_path = self.get_checkpoint_path(epoch)
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, train_state_path)

    def iter_end_callback(self, epoch, max_epochs, iter_nb, max_iter, loss, loss_components):
        line = 'Epoch [{}/{}] - Iter [{}/{}] - Loss: {:.5f} - '.format(epoch, max_epochs, iter_nb, max_iter, loss)
        line += ' - '.join(
            ['{}: {:.5f}'.format(component, loss_components[component]) for component in loss_components])
        self.logger.debug(line)
        overall_iter = (epoch * max_iter) + iter_nb
        self.tensorboard_writer.add_scalar('loss/total_loss', loss, overall_iter)
        for key in loss_components:
            self.tensorboard_writer.add_scalar('loss/{}'.format(key), loss_components[key], overall_iter)

    def epoch_start_callback(self, epoch, max_epochs):
        self.logger.debug('Epoch [%d/%d] starting.', epoch, max_epochs)

    def epoch_end_callback(self, epoch, max_epochs, model, optimizer, scheduler):
        self.logger.debug('Epoch [%d/%d] finished.', epoch, max_epochs)
        if epoch % self.cfg['model_checkpoint_interval'] == 0:
            self.save_train_state(epoch, model, optimizer, scheduler)

    def train_start_callback(self, cfg):
        self.logger.debug('Beginning training session. CFG used:\n%s', str(cfg))

    def train_end_callback(self):
        self.logger.debug('Training session finished.')

    def eval_start_callback(self, cfg):
        self.logger.debug('Beginning testing session. CFG used:\n%s', str(cfg))

    def eval_end_callback(self, dataset, predictions, epoch_evaluated):
        metrics = self.save_epoch_results(dataset, predictions, epoch_evaluated)
        self.logger.debug('Testing session finished on model after epoch %d.', epoch_evaluated)
        self.logger.info('Results:\n %s', str(metrics))
        return metrics


    def save_epoch_results(self, dataset, predictions, epoch):
        # 计算标准 TuSimple 指标
        metrics = dataset.eval_predictions(predictions, self.get_output_dir(), None)

        # ------------------- 新增：验证集平均 loss 计算 -------------------
        try:
            import torch
            import numpy as np
            model = self.cfg.get_model().to("cuda")
            model.load_state_dict(self.get_epoch_model(epoch))
            model.eval()

            val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=8,
                shuffle=False,
                num_workers=4
            )
            total_loss, total_cls, total_reg, count = 0, 0, 0, 0
            loss_parameters = self.cfg.get_loss_parameters() if hasattr(self, "cfg") else {}

            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = model(images)
                    loss, loss_dict = model.loss(outputs, labels, **loss_parameters)
                    total_loss += loss.item()
                    total_cls += loss_dict["cls_loss"]
                    total_reg += loss_dict["reg_loss"]
                    count += 1

            metrics.update({
                "val/loss": total_loss / count,
                "val/cls_loss": total_cls / count,
                "val/reg_loss": total_reg / count,
            })
        except Exception as e:
            self.logger.warning(f"⚠️ Could not compute val losses automatically: {e}")
        # ---------------------------------------------------------------

        # 保存结果并返回
        self.logger.info('Results:\n %s', str(metrics))
        return metrics

