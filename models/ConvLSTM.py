import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM(nn.Module):
    def __init__(self, dim, win_len, num_classes_1, num_feat_map, dropout_rate, batchnorm=True, dropout=True):
        super(ConvLSTM, self).__init__()

        self.batchnorm = batchnorm
        self.dropout = dropout
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_feat_map, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(num_feat_map)
        self.drop1 = nn.Dropout(dropout_rate)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=num_feat_map, out_channels=num_feat_map, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(num_feat_map)
        self.drop2 = nn.Dropout(dropout_rate)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_feat_map * dim, hidden_size=32, batch_first=True)
        self.drop3 = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(32, num_classes_1)

    def forward(self, x):
        # First convolutional block
        x = F.relu(self.conv1(x))
        if self.batchnorm:
            x = self.bn1(x)
        if self.dropout:
            x = self.drop1(x)

        # Second convolutional block
        x = F.relu(self.conv2(x))
        if self.batchnorm:
            x = self.bn2(x)
        if self.dropout:
            x = self.drop2(x)

        # Prepare for LSTM
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)  # Swap dimensions for LSTM
        x = x.contiguous().view(B, W, -1)

        # LSTM layer
        x, _ = self.lstm(x)
        if self.dropout:
            x = self.drop3(x[:, -1, :])  # Use the last time-step
        else:
            x = x[:, -1, :]

        # Fully connected layer
        x = F.softmax(self.fc(x), dim=1)

        return x


if __name__ == '__main__':
    # Example
    model = ConvLSTM(dim=77, win_len=1, num_classes_1=18, num_feat_map=64, dropout_rate=0.3)
    print(model)
