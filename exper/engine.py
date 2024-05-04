import logging
import os
import sys
from typing import Any, List

import torch
from torch import distributed as dist
from torch import nn
from torch.cuda import amp as amp
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

import exper.comms as comms
import exper.core as core
import exper.cuda_utils as utils
import exper.pretty as pretty
from exper.tasks import Task


module = sys.modules[__name__]
logger_outside = logging.getLogger(__name__)


class Engine:
    """
    Parameters:
        task (nn.Module): task
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer
        scheduler (lr_scheduler._LRScheduler, optional): scheduler
        gpus (list of int, optional): GPU ids. By default, CPUs will be used.
            For multi-node multiprocess case, repeat the GPU ids for each node.
        batch_size (int, optional): batch size of a single CPU / GPU
        num_worker (int, optional): number of CPU workers per GPU
        logger (str or core.LoggerBase, optional): logger type or logger instance.
            Available types are ``logging`` and ``wandb``.
        half_precision(bool, optional): use the half precision mode

    """

    def __init__(  # noqa
        self,
        task: Task,
        train_set: Dataset | PyGDataset,
        valid_set: Dataset | PyGDataset,
        test_set: Dataset | PyGDataset,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        gpus: List[int] | None = None,
        batch_size: int = 1,
        num_worker: int = 0,
        log_interval: int = 100,
        half_precision: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        self.rank = comms.get_rank()
        self.world_size = comms.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.half_precision = half_precision
        for key, value in kwargs.items():
            setattr(self, key, value)

        if not hasattr(self, "follow_batch"):
            self.follow_batch = None

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += (
                        ". Did you launch with `python -m torch.distributed.launch`?"
                    )
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger_outside.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comms.init_process_group(backend, init_method="env://")

        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)  # type: ignore
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        # `task` is a self-designed model to perform your experiments
        self.model = task

        # Here are datasets you will use during the experiments
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

        # determine the dataset type
        self.dataset_type = train_set.__module__.split(".")[0]

        # This is the optimizer and scheduler to help the model optimize and converge
        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.half_precision:
            self.scaler = amp.GradScaler(enabled=True)

        # Here are functions to record experiments
        self.log = core.LoggingLogger()
        self.meter = core.Meter(
            log_interval=log_interval, silent=self.rank > 0, logger=self.log
        )

        module.logger_outside.warning(
            "Mix Precision Training Enabled: {}".format(self.half_precision)
        )

    def train(self, num_epoch: int = 1):
        """
        Parameters:
            num_epoch (int, optional): number of epochs
        """
        sampler = torch_data.DistributedSampler(
            self.train_set, self.world_size, self.rank
        )

        if self.dataset_type == "torch":
            dataloader = DataLoader(
                self.train_set,
                self.batch_size,
                sampler=sampler,
                num_workers=self.num_worker,
            )

        else:
            assert isinstance(self.train_set, PyGDataset)
            dataloader = PyGDataLoader(
                self.train_set,
                self.batch_size,
                sampler=sampler,
                num_workers=self.num_worker,
                follow_batch=self.follow_batch,
            )

        model = self.model

        # TODO: here we use the DDP to wrapper the model
        # self.model -> the original model
        # model -> the wrappered model

        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.device], find_unused_parameters=True
                )
            else:
                model = nn.parallel.DistributedDataParallel(
                    model, find_unused_parameters=True
                )

        model.train()

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)
            losses = []

            for i, batch in enumerate(dataloader):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                # TODO: cause DDP warp the original model
                # so use the `self.model` to get label
                # target = self.model.target(batch)

                with torch.cuda.amp.autocast(enabled=self.half_precision):
                    # TODO: be sure to use `model` (not `self.model`)
                    loss, pred = model(batch)

                # TODO: this is the mix precision training strategy
                if self.half_precision:
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                # TODO: Here we can add self-designed loss to update the model
                # TODO: key: loss name (support multiple loss record)
                # TODO: You have to implement the loss_name attribute in the self.model
                cur_batch_loss_info = self.model.loss_info()
                losses.append(cur_batch_loss_info)

                # Here is the DDP loss, we need to get the loss from all clusters
                n_loss = utils.stack(losses, dim=0)
                n_loss = utils.mean(n_loss, dim=0)

                if self.world_size > 1:
                    n_loss = comms.reduce(n_loss, op="mean")
                assert isinstance(n_loss, dict)
                self.meter.update(n_loss)

                # clear the loss history
                losses = []

            if self.scheduler:
                self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, split: str = "train", log: bool = True):
        """
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not
        Returns:
            dict: metrics
        """
        if self.rank == 0:
            self.log.logger.warning(pretty.separator)
            self.log.logger.warning("Evaluate on %s" % split)

        test_set = getattr(self, "%s_set" % split)

        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)

        if self.dataset_type == "torch":
            dataloader = DataLoader(
                self.test_set,
                self.batch_size,
                sampler=sampler,
                num_workers=self.num_worker,
            )

        else:
            assert isinstance(self.test_set, PyGDataset)
            dataloader = PyGDataLoader(
                self.test_set,
                self.batch_size,
                sampler=sampler,
                num_workers=self.num_worker,
                follow_batch=self.follow_batch,
            )

        model = self.model
        model.eval()

        preds = []
        targets = []
        for i, batch in enumerate(dataloader):
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            target = model.target(batch)
            loss, pred = model(batch)

            preds.append(pred)
            targets.append(target)

        pred = utils.cat(preds)
        target = utils.cat(targets)

        if self.world_size > 1:
            pred = comms.cat(pred)
            target = comms.cat(target)

        metric = model.evaluate(pred, target)

        # TODO: we only keep the main thread log record
        if log and self.rank == 0:
            self.meter.log(metric, category="%s/epoch" % split)
        return metric

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comms.get_rank() == 0:
            self.log.logger.warning("Load checkpoint from %s" % checkpoint)

        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])
        self.meter.epoch_id = state["epoch_id"]

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comms.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comms.get_rank() == 0:
            self.log.logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch_id": self.meter.epoch_id,
            }
            torch.save(state, checkpoint)
        comms.synchronize()

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id
