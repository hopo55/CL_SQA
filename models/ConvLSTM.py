import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM(nn.Module):
    def __init__(self, dim, win_len, num_classes_1, num_feat_map, dropout_rate, batchnorm=True, dropout=True):
        super(ConvLSTM, self).__init__()

        self.batchnorm = batchnorm
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(1, num_feat_map, kernel_size=(1, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(num_feat_map)
        self.conv2 = nn.Conv2d(num_feat_map, num_feat_map, kernel_size=(1, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(num_feat_map)
        self.fc_size = num_feat_map * dim
        self.lstm = nn.LSTM(input_size=self.fc_size, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, num_classes_1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.batchnorm:
            x = self.bn1(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate)

        x = F.relu(self.conv2(x))
        if self.batchnorm:
            x = self.bn2(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate)

        # Adjusting the dimensions for LSTM input
        x = x.permute(0, 2, 1, 3)  # PyTorch permute order (batch_size, seq_len, num_channels, features)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the dimensions except for batch and seq_len

        x, (hn, cn) = self.lstm(x)  # We only use the output of the last layer of LSTM
        x = self.fc(x[:, -1, :])  # Get the last output for classification
        # x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':
    # Example
    model = ConvLSTM(dim=77, win_len=1, num_classes_1=18, num_feat_map=64, dropout_rate=0.3)
    print(model)
