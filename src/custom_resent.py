import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=padding,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, stride=stride,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __call__(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=padding,
                      stride=stride, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __call__(self, x):
        x = self.conv_block1(x)
        return x


class Net(LightningModule):

    def __init__(self, learning_rate=0.01):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_classes = 10
        self.train_steps = 0
        self.optimizer = None

        # Initialize instance attributes to store misclassified images, labels, and predictions
        self.misclassified_images = None
        self.misclassified_labels = None
        self.misclassified_predictions = None
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        # Prep Layer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        self.basic_block1 = BasicBlock(in_channels=64, out_channels=128)
        self.R1 = ResBlock(in_channels=128, out_channels=128)

        # Layer 2
        self.basic_block2 = BasicBlock(in_channels=128, out_channels=256, stride=1, padding=2)

        # # Layer 3
        self.basic_block3 = BasicBlock(in_channels=256, out_channels=512)
        self.R2 = ResBlock(in_channels=512, out_channels=512)

        # MaxPooling
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=4)
        )

        # FC layer
        self.fc_layer = nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):
        x = self.prep_layer(x)  # 1. Prep Layer

        x1 = self.basic_block1(x)  # 2. Layer 1
        r1 = self.R1(x1)
        x = x1 + r1

        x = self.basic_block2(x)  # 3. Layer 2

        x2 = self.basic_block3(x)  # 4. Layer 3
        r2 = self.R2(x2)
        x = x2 + r2

        x = self.pool(x)  # 5. MaxPooling

        x = x.view(-1, 512)

        x = self.fc_layer(x)  # 6. FC Layer

        return F.log_softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        self.train_steps += 1
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(y_hat, dim=1)
        self.accuracy(preds, y)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': self.accuracy}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=4.64E-02, steps_per_epoch=98,
                                                             epochs=26),
            'interval': 'step',
            'frequency': 1
        }
        return [self.optimizer], [scheduler]
