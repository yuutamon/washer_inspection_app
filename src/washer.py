# 必要なモジュールをインポート
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# 学習時に使用したものと同じ学習済みモデルをインポート



# 学習済みモデルに合わせた前処理を追加
# データの前処理
transform = transforms.Compose([

    # 画像をテンソルに変換する
    # transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(size=(112, 112)),
    transforms.ToTensor()
])

# エンコーディング層の定義
class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        return h

# デコーディング層の定義
class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.convt2 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(3)

    def forward(self, x):
        h = self.convt1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.convt2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.sigmoid(h)
        return h

class AutoEncoder(pl.LightningModule):

  def __init__(self):
      super().__init__()

      self.encoder = Encoder()
      self.decoder = Decoder()

  # 順伝播
  def forward(self, x):
      h = self.encoder(x)
      h = self.decoder(h)
      return h