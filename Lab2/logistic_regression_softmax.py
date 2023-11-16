import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(LogisticRegression, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, 64)
        self.l2 = torch.nn.Linear(64, 32)
        self.l3 = torch.nn.Linear(32, 16)
        self.l4 = torch.nn.Linear(16, 8)
        self.l5 = torch.nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.relu(x)
        x = self.l4(x)
        x = torch.relu(x)
        x = self.l5(x)
        return x
