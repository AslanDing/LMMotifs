import numpy as np
import torch
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric

dataset_name = 'pubmed_gpt-4-1_5parser'
path = f'./data/ours/graphs/{dataset_name}.pth'

graphs = torch.load(path)
train_length = int(len(graphs)*0.8)
test_length = len(graphs)-train_length
label_0_graphs = graphs[:len(graphs)//2]
label_1_graphs = graphs[len(graphs)//2:]

train_graphs = []
train_graphs.extend(label_0_graphs[:train_length//2])
train_graphs.extend(label_1_graphs[:train_length//2])
test_graphs = []
test_graphs.extend(label_0_graphs[train_length//2:])
test_graphs.extend(label_1_graphs[train_length//2:])

train_loader = DataLoader(train_graphs, batch_size=len(train_graphs), shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        try:
            x, edge_index = data.x, data.edge_index
        except:
            x, edge_index = data.x, None

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch_geometric.nn.global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x


model = GCN(in_channels=300, hidden_channels=100, out_channels=2)  # 2 classes
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing loop
def test():
    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        output = model(data)
        _, predicted = output.max(dim=1)
        correct += predicted.eq(data.y).sum().item()
        total += data.num_graphs
    accuracy = correct / total
    return accuracy

# Train and evaluate the model
# best_acc = 0
best_acc = test()
for epoch in range(1, 201):
    loss = train()
    accuracy = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    if accuracy>best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(),path.replace('graphs','model'))
        print("saved model")


