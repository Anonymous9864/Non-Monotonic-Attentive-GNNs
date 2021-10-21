import torch
import argparse
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
from src.backbone.GAT_modified import M_GATLayer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-d', '--dataset', type=str, default='Cora')
parser.add_argument('-mp', '--mapping_function', type=str, default='sin')
parser.add_argument('-bn', '--base_number', type=int, default=1)
parser.add_argument('-dr', '--drop_rate', type=float, default=0.8)
parser.add_argument('-e', '--epoch', type=int, default=500)
parser.add_argument('-hs', '--heads', type=int, default=8)
parser.add_argument('-fyd', '--first_layer_dimension', type=int, default=8)
parser.add_argument('-r', '--rand_seed', type=int, default=1)
args = parser.parse_args()

train_dataset = args.dataset
mapping_function = args.mapping_function
cuda_device = args.cuda
heads = args.heads
first_layer_dimension = args.first_layer_dimension
epoch_num = args.epoch
base_number = args.base_number
drop_rate = args.drop_rate

# set random seed
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="./data", name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)


# Model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gnn1 = M_GATLayer(dataset.num_features, first_layer_dimension, mapping_function=mapping_function,
                               base_number=base_number, heads=8, dropout=drop_rate, normalize=False)
        self.gnn2 = M_GATLayer(first_layer_dimension * heads, dataset.num_classes, mapping_function=mapping_function,
                               base_number=base_number, dropout=drop_rate, normalize=False)

    def forward(self, x: Tensor, edge_index: Adj):
        x = F.dropout(x, p=drop_rate, training=self.training)
        x = self.gnn1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=drop_rate, training=self.training)
        x = self.gnn2(x, edge_index)
        return x

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    _, pred = out.max(dim=1)
    train_correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = train_correct / int(data.train_mask.sum())
    validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    validate_acc = validate_correct / int(data.val_mask.sum())
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    test_acc = test_correct / int(data.test_mask.sum())
    return train_acc, validate_acc, test_acc


best_val_acc = test_acc = 0
for epoch in range(epoch_num):
    train()
    train_acc, val_acc, current_test_acc = test()
    print('For the {} epoch, the train acc is {}, the val acc is {}, the test acc is {}.'.format(epoch, train_acc,
                                                                                                 val_acc,
                                                                                                 current_test_acc))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = current_test_acc

print('Mission completes.')
print('The best val acc is {}, and the test acc is {}.'.format(best_val_acc, test_acc))
