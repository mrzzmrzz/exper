from abc import abstractmethod
from typing import Any

import torch.nn as nn


class Task(nn.Module):
    @abstractmethod
    def forward(self, batch: Any) -> Any:
        """
        input:
            batch: the batch data used for training the model
        output:
            loss: the final loss used to optimize the model
            pred: the pred value used to evaluate the model
        """
        raise NotImplementedError

    @abstractmethod
    def target(self, batch: Any) -> Any:
        """
        input:
            batch: the batch data used to train the model
        output:
            target: the batch's true label
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, pred: Any, target: Any) -> dict:
        """
        input:
            pred: the pred value used to evaluate the model
            target: the batch's true label
        output:
            metric: the evaluation metric for current batch,
                    the format should be in dict format {metric_name: metric_value}
        """
        raise NotImplementedError

    @abstractmethod
    def loss_info(self) -> dict:
        """
        input:
        output:
            loss_info: a batch training loss statics used to
                       trace the whole training process {loss_name: loss_value}
        """
        raise NotImplementedError
