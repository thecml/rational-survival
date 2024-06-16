from sklearn.datasets import make_classification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from rational.generator import Generator
from utility import dotdict
import config as cfg

args = dotdict(cfg.PARAMS_RATIONAL)
args['dropout'] = 0.1
args['selection_lambda'] = 0.1

n_features = 10
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=2, class_sep=2, flip_y=0,
                           weights=[0.5, 0.5], random_state=0)

num_additional_features = 8
additional_features = np.random.normal(loc=0, scale=0.5,
                                       size=(10000, num_additional_features))
X = np.concatenate((X, additional_features), axis=1)


# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

class BinaryClassifier(nn.Module):
    def __init__(self, num_features = 10):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 50)  # Input layer
        self.fc2 = nn.Linear(50, 10) # Hidden layer
        self.fc3 = nn.Linear(10, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
class BinaryClassifierRational(nn.Module):
    def __init__(self, GeneratorArgs, num_features=10):
        super(BinaryClassifierRational, self).__init__()
        self.generator = Generator(num_features, args=GeneratorArgs)
        self.fc1 = nn.Linear(num_features, 50)  # Input layer
        self.fc2 = nn.Linear(50, 10) # Hidden layer
        self.fc3 = nn.Linear(10, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()
        self.mask = None

    def forward(self, x, test=False):
        self.mask, _ = self.generator(x, test)
        x_mask = x * self.mask.unsqueeze(-1)
        x = x.squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
def train_model(model, train_loader, val_loader, num_epochs=50):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs[:, :, None]

            outputs = model(inputs)
            # print (labels)
            # print (outputs)
            outputs = outputs.squeeze(1)
            # print (outputs)
            selection_cost = model.generator.loss(model.mask, inputs)

            loss = criterion(outputs, labels)
            loss += 1 * selection_cost
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, labels in val_loader:
                inputs = inputs[:, :, None]
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                val_loss += criterion(outputs, labels).item()
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Initialize model and train
# model = BinaryClassifier()
model_rat = BinaryClassifierRational(args)

train_model(model_rat, train_loader, val_loader)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs[:, :, None]
            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100*correct / total
    print(f'Accuracy on test data: {accuracy:.2f}%')

# Evaluate the model
evaluate_model(model_rat, test_loader)
X_test = X_test[:, :, None]

model_rat(X_test)
model_rat.mask.sum(axis=0)