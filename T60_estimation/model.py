import torch.nn as nn

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 10), stride=(1, 1), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 10), stride=(1, 1), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(2, 8), stride=(1, 1), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(2, 8), stride=(1, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(1, 6), stride=(1, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(1, 2)),

        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4608, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.cnn_layers(x.float())
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
