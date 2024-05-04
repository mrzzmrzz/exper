import logging
import pprint
import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

import exper.pretty as pretty


logger = logging.getLogger(__name__)


class LoggerBase(ABC):
    """
    Base class for loggers.

    Any custom logger should be derived from this class.
    """

    @abstractmethod
    def log(self, record: dict, step_id: int, category: str = "train/batch") -> None:
        """
        Log a record.

        Parameters:
            record (dict): dict of any metric
            step_id (int): index of this log step
            category (str, optional): log category.
                Available types are ``train/batch``, ``train/epoch``, ``valid/epoch`` and ``test/epoch``.
        """
        raise NotImplementedError

    @abstractmethod
    def log_config(self, config: dict) -> None:
        """
        Log a hyperparameter config.

        Parameters:
            config (dict): hyperparameter config
        """
        raise NotImplementedError


class LoggingLogger(LoggerBase):
    """
    Log outputs with the builtin logging module of Python.

    By default, the logs will be printed to the console. To additionally log outputs to a file,
    add the following lines in the beginning of your code.

    .. code-block: python

        import logging

        format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger = logging.getLogger("")
        logger.addHandler(handler)
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def log(self, record: dict, step_id: int, category: str = "train/batch") -> None:
        if category.endswith("batch"):
            self.logger.warning(pretty.separator)
        elif category.endswith("epoch"):
            self.logger.warning(pretty.line)
        if category == "train/epoch":
            for k in sorted(record.keys()):
                self.logger.warning("average %s: %g" % (k, record[k]))
        else:
            for k in sorted(record.keys()):
                self.logger.warning("%s: %g" % (k, record[k]))

    def log_config(self, config: dict) -> None:
        self.logger.warning(pprint.pformat(config))


class Meter(object):
    """
    Meter for recording metrics and training progress.

    Parameters:
        log_interval (int): log every n updates
        silent (int): surpress all outputs or not
        logger (core.LoggerBase): log handler
    """

    def __init__(
        self,
        logger: LoggerBase,
        log_interval: int = 100,
        silent: bool = False,
    ) -> None:
        
        self.records = defaultdict(list)
        self.log_interval = log_interval
        self.epoch2batch = [0]
        self.time = [time.time()]
        self.epoch_id = 0
        self.batch_id = 0
        self.silent = silent
        self.logger = logger

    def log(self, record: dict, category: str ="train/batch") -> None:
        """
        Log a record.

        Parameters:
            record (dict): dict of any metric
            category (str, optional): log category.
                Available types are ``train/batch``, ``train/epoch``, ``valid/epoch`` and ``test/epoch``.
        """
        if self.silent:
            return

        step_id: int = -1

        if category.endswith("batch"):
            step_id = self.batch_id
        elif category.endswith("epoch"):
            step_id = self.epoch_id

        self.logger.log(record, step_id=step_id, category=category)

    def log_config(self, config: dict):
        """
        Log a hyperparameter config.

        Parameters:
            config (dict): hyperparameter config
        """
        if self.silent:
            return

        self.logger.log_config(config)

    def update(self, record: dict):
        """
        Update the meter with a record.

        Parameters:
            record (dict): dict of any metric
        """
        if self.batch_id % self.log_interval == 0:
            self.log(record, category="train/batch")
        self.batch_id += 1

        for k, v in record.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.records[k].append(v)

    def step(self):
        """
        Step an epoch for this meter.

        Instead of manually invoking :meth:`step()`, it is suggested to use the following line

            >>> for epoch in meter(num_epoch):
            >>> # do something
        """
        self.epoch_id += 1
        self.epoch2batch.append(self.batch_id)
        self.time.append(time.time())
        index = slice(self.epoch2batch[-2], self.epoch2batch[-1])
        duration = self.time[-1] - self.time[-2]
        speed = (self.epoch2batch[-1] - self.epoch2batch[-2]) / duration
        if self.silent:
            return

        logger.warning("duration: %s" % pretty.time(duration))
        logger.warning("speed: %.2f batch / sec" % speed)

        eta = (
            (self.time[-1] - self.time[self.start_epoch])
            / (self.epoch_id - self.start_epoch)
            * (self.end_epoch - self.epoch_id)
        )

        logger.warning("ETA: %s" % pretty.time(eta))
        if torch.cuda.is_available():
            logger.warning(
                "max GPU memory: %.1f MiB" % (torch.cuda.max_memory_allocated() / 1e6)
            )

            torch.cuda.reset_peak_memory_stats()

        record = {}
        for k, v in self.records.items():
            record[k] = np.mean(v[index])
        self.log(record, category="train/epoch")

    def __call__(self, num_epoch):
        self.start_epoch = self.epoch_id
        self.end_epoch = self.start_epoch + num_epoch

        for epoch in range(self.start_epoch, self.end_epoch):
            if not self.silent:
                logger.warning(pretty.separator)
                logger.warning("Epoch %d begin" % epoch)
            yield epoch
            if not self.silent:
                logger.warning(pretty.separator)
                logger.warning("Epoch %d end" % epoch)
            self.step()
