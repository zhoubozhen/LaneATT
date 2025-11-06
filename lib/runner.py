import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange


class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        from lib.logger_wandb import init_wandb
        import wandb
        import torch

        # ✅ 初始化 wandb
        if not wandb.run:
            init_wandb(self.exp.name, self.cfg.__dict__)

        # ✅ 定义 metric
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("epoch", hidden=True)

        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model().to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)

        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1

        max_epochs = self.cfg["epochs"]
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()

        best_acc = 0.0
        best_epoch = 0
        global_step = 0

        for epoch in range(starting_epoch, max_epochs + 1):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)

            total_loss = total_cls = total_reg = 0.0
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                total_cls += loss_dict_i["cls_loss"]
                total_reg += loss_dict_i["reg_loss"]

                global_step += 1  # ✅ global counter

                # ✅ batch log 用递增 step
                wandb.log({
                    "train/loss": loss.item(),
                    "train/cls_loss": loss_dict_i["cls_loss"],
                    "train/reg_loss": loss_dict_i["reg_loss"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "step": global_step
                })

                postfix_dict = {
                    "loss": float(loss.item()),
                    "cls_loss": float(loss_dict_i["cls_loss"]),
                    "reg_loss": float(loss_dict_i["reg_loss"]),
                    "lr": optimizer.param_groups[0]["lr"]
                }
                pbar.set_postfix(ordered_dict=postfix_dict)
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)

            # ✅ epoch 平均 loss
            n_batches = len(train_loader)
            avg_loss = total_loss / n_batches
            avg_cls = total_cls / n_batches
            avg_reg = total_reg / n_batches

            wandb.log({
                "epoch/train_loss": avg_loss,
                "epoch/train_cls_loss": avg_cls,
                "epoch/train_reg_loss": avg_reg,
                "epoch": epoch,
                "step": global_step
            })

            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # ✅ 每 val_every 验证一次
            if (epoch + 1) % self.cfg["val_every"] == 0:
                self.logger.info(f"🔍 Running validation at epoch {epoch} ...")
                self.exp.save_train_state(epoch, model, optimizer, scheduler)
                try:
                    result = self.eval(epoch, on_val=True)
                except Exception as e:
                    self.logger.warning(f"⚠️ Validation failed at epoch {epoch}: {e}")
                    result = None

                # ✅ validation log —— 不指定 step，沿用 global_step！
                wandb.log({
                    "val/Accuracy": 0.0 if not result else result.get("Accuracy", 0),
                    "val/FP": 0.0 if not result else result.get("FP", 0),
                    "val/FN": 0.0 if not result else result.get("FN", 0),
                    "val/loss": 0.0 if not result else result.get("val/loss", 0),
                    "val/cls_loss": 0.0 if not result else result.get("val/cls_loss", 0),
                    "val/reg_loss": 0.0 if not result else result.get("val/reg_loss", 0),
                    "epoch": epoch
                })

                # ✅ 最优模型
                acc = 0.0 if not result else result.get("Accuracy", 0)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    best_path = self.exp.save_model(model, optimizer, scheduler, epoch, tag="best")
                    self.logger.info(f"🔥 New best model (Acc={best_acc:.4f}) saved at {best_path}")

        self.logger.info(f"✅ Training finished. Best accuracy: {best_acc:.4f} (epoch {best_epoch})")
        self.exp.train_end_callback()







    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model.eval()
        model = model.to(self.device)
        

        dataloader = self.get_val_dataloader() if on_val else self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []

        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)

        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # ✅ 调用 evaluation 计算指标
        result = self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

        # ✅ 返回结果以便上层 log_metrics 使用
        return result


    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
