from torchvision import models

from ser.data_model.criterion import Criterion
from ser.model.base_model import TLModelBase
from ser.trainer.optimizer import Optimizer


class Resnet18Model(TLModelBase):

    def __init__(self, model, criterion, optimizer, features):
        super().__init__(model=model, criterion=criterion, optimizer=optimizer,
                         in_features=features)


def train_resnet18_model(criterion: Criterion,
                         optimizer: Optimizer,
                         model=models.resnet18(pretrained=True)) -> Resnet18Model:
    return Resnet18Model(model, criterion, optimizer, features=model.fc.in_features)
