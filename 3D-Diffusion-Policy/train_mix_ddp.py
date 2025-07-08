if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import accelerate
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, ConcatDataset
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
import sys

## Accelerate
from accelerate import Accelerator


sys.path.append("/mnt/hwfile/liumingyu/code/3D-Diffusion-Policy/experiments")
from debug_utils import setup_debug
OmegaConf.register_new_resolver("eval", eval, replace=True)
import os
# 在初始化分布式训练前设置
os.environ['NCCL_TIMEOUT'] = '600'  # 30分钟
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
# os.environ['NCCL_DEBUG'] = 'INFO'  # 输出详细日志
# os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'  # 检查所有子系统

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # 初始化 accelerator
        self.accelerator = Accelerator()

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        # 只在主进程输出模型信息
        if self.accelerator.is_main_process:
            print(f"Model created with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        cfg.training.resume = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        
        dataset = hydra.utils.instantiate(cfg.task.dataset) 
        assert isinstance(dataset, BaseDataset) or (
            isinstance(dataset, ConcatDataset) and 
            all(isinstance(d, BaseDataset) for d in dataset.datasets)
        ), f"dataset must be BaseDataset or ConcatDataset of BaseDataset, got {type(dataset)}"
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        
        datas = [d.get_data() for d in dataset.datasets]
        
        normalizer = LinearNormalizer()
        combined_data = {
            key: np.concatenate([d[key] for d in datas], axis=0)
            for key in datas[0].keys()  # 使用第一个字典的键作为参考
        }
        
        normalizer.fit(data=combined_data, last_n_dims=1, mode='limits')
        # import pdb; pdb.set_trace()
        # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )


        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)

        cfg.logging.name = str(cfg.logging.name)
        # configure logging
        if self.accelerator.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )
        else:
            wandb_run = None
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # 使用 accelerator 准备模型、优化器和数据加载器
        self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader
        )
        if cfg.training.use_ema:
            self.ema_model = self.accelerator.prepare(self.ema_model)
        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # 确保所有进程同步
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    # import pdb; pdb.set_trace()
                    raw_loss, loss_dict = self.model.module.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every

                    self.accelerator.backward(loss)
                    # loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        ema.step(unwrapped_model)
                        self.accelerator.wait_for_everyone()
                    
                    t1_4 = time.time()
                    # logging
                    if self.accelerator.is_main_process:
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                        t2 = time.time()
                        
                        if verbose:
                            print(f"total one step time: {t2-t1:.3f}")
                            print(f" compute loss time: {t1_2-t1_1:.3f}")
                            print(f" step optimizer time: {t1_3-t1_2:.3f}")
                            print(f" update ema time: {t1_4-t1_3:.3f}")
                            print(f" logging time: {t1_5-t1_4:.3f}")

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                    self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            if self.accelerator.is_main_process:
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model

            # 同步所有进程，确保模型状态一致
            self.accelerator.wait_for_everyone()
            # policy.eval()

            # # run rollout
            # if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
            #     t3 = time.time()
            #     # runner_log = env_runner.run(policy, dataset=dataset)
            #     runner_log = env_runner.run(policy)
            #     t4 = time.time()
            #     # print(f"rollout time: {t4-t3:.3f}")
            #     # log all
            #     step_log.update(runner_log)

            
            
            # run validation
            # import pdb; pdb.set_trace()
            # if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
            #     with torch.no_grad():
            #         val_losses = list()
            #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
            #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            #             for batch_idx, batch in enumerate(tepoch):
            #                 import pdb; pdb.set_trace()
            #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            #                 loss, loss_dict = self.model.compute_loss(batch)
            #                 val_losses.append(loss)
            #                 if (cfg.training.max_val_steps is not None) \
            #                     and batch_idx >= (cfg.training.max_val_steps-1):
            #                     break
            #         if len(val_losses) > 0:
            #             val_loss = torch.mean(torch.tensor(val_losses)).item()
            #             # log epoch average validation loss
            #             step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            # import pdb; pdb.set_trace()
            if self.accelerator.is_main_process:
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        # batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
        
                        obs_dict = dict()
                        obs_dict['obs'] = batch['obs']
                        obs_dict['actions'] = batch['action']
                        gt_action = batch['action']
                        # import pdb; pdb.set_trace()
                        result = policy.predict_action(obs_dict)
                        # import pdb; pdb.set_trace()
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if wandb_run is not None:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
    
    def eval_mine(self):
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        
        # self.modelself.model
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        dataset: BaseDataset
        # import pdb; pdb.set_trace()
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset) or (
            isinstance(dataset, ConcatDataset) and 
            all(isinstance(d, BaseDataset) for d in dataset.datasets)
        ), f"dataset must be BaseDataset or ConcatDataset of BaseDataset, got {type(dataset)}"
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        
        
        datas = [d.get_data() for d in dataset.datasets]
        # import pdb; pdb.set_trace()
        normalizer = LinearNormalizer()
        combined_data = {
            key: np.concatenate([d[key] for d in datas], axis=0)
            for key in datas[0].keys()  # 使用第一个字典的键作为参考
        }
        # import pdb; pdb.set_trace()
        normalizer.fit(data=combined_data, last_n_dims=1, mode='limits')

        # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)


    
        # device transfer
        # import pdb; pdb.set_trace()
        device = torch.device('cpu')
        # self.model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    if train_sampling_batch is None:
                        train_sampling_batch = batch
            
            with torch.no_grad():
                # sample trajectory from training set, and evaluate difference
                
                # batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = dict()
                obs_dict['obs'] = batch['obs']
                obs_dict['actions'] = batch['action']
                # import pdb; pdb.set_trace()
                gt_action = batch['action']
                # import pdb; pdb.set_trace()
                
                result = policy.predict_action(obs_dict)
                pred_action = result['action_pred']
                import pdb; pdb.set_trace()
                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                step_log['train_action_mse_error'] = mse.item()
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        # 只在主进程中保存检查点
        if not self.accelerator.is_main_process:
            return None
            
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    # 对于模型，获取未包装的原始模型
                    if key == 'model':
                        unwrapped_model = self.accelerator.unwrap_model(value)
                        if use_thread:
                            payload['state_dicts'][key] = _copy_to_cpu(unwrapped_model.state_dict())
                        else:
                            payload['state_dicts'][key] = unwrapped_model.state_dict()
                    elif key == 'ema_model' and hasattr(self, 'ema_model'):
                        unwrapped_ema = self.accelerator.unwrap_model(value)
                        if use_thread:
                            payload['state_dicts'][key] = _copy_to_cpu(unwrapped_ema.state_dict())
                        else:
                            payload['state_dicts'][key] = unwrapped_ema.state_dict()
                    else:
                        # 对于优化器等其他组件
                        if use_thread:
                            payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                        else:
                            payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    

    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None,
            include_keys=None):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        # 确保所有进程都等待检查点文件准备好
        self.accelerator.wait_for_everyone()
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # 加载检查点
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        
        # 加载state_dicts
        for key, value in payload['state_dicts'].items():
            if key in self.__dict__:
                if key == 'model':
                    # 对于模型，直接加载到accelerator包装的模型中
                    self.model.load_state_dict(value, strict=False)
                elif key == 'ema_model' and hasattr(self, 'ema_model'):
                    # 对于EMA模型
                    self.ema_model.load_state_dict(value, strict=False)
                elif hasattr(self.__dict__[key], 'load_state_dict'):
                    # 对于优化器等其他组件
                    self.__dict__[key].load_state_dict(value)
        
        # 加载pickles
        for key, value in payload['pickles'].items():
            if key in include_keys:
                self.__dict__[key] = dill.loads(value)
        
        # 同步所有进程
        self.accelerator.wait_for_everyone()
        
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    
# import pdb; pdb.set_trace()
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config')) # 'diffusion_policy_3d/config'
)


def main(cfg):
    # import pdb; pdb.set_trace()

    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    # setup_debug(True)

    main()
