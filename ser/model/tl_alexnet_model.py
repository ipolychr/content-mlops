from torchvision import models

from ser.data_model.criterion import Criterion
from ser.model.base_model import TLModelBase
from ser.trainer.optimizer import Optimizer


class AlexNetModel(TLModelBase):

    def __init__(self, model, criterion, optimizer, features):
        super().__init__(model=model, criterion=criterion, optimizer=optimizer,
                         in_features=features)


def train_alex_net_model(criterion: Criterion,
                         optimizer: Optimizer,
                         model=models.alexnet(pretrained=True), ) -> AlexNetModel:
    return AlexNetModel(model=model, criterion=criterion, optimizer=optimizer,
                        features=model.classifier[-1].in_features)
