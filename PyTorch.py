import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Setup Device
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> Using device: {device}")

# ==========================================
# 2. Load and Preprocess Data
# ==========================================
# Load and clean data to avoid NaN loss
df = pd.read_csv('breast_cancer.csv')
df = df.dropna(axis=1, how='all')
df = df.dropna()

X_raw = df.drop(['id', 'diagnosis'], axis=1).select_dtypes(include=['number']).values
y_raw = (df['diagnosis'] == 'M').astype(int).values 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Convert to writable Tensors
X_tensor = torch.from_numpy(X_scaled.copy()).float()
y_tensor = torch.from_numpy(y_raw.copy()).long()

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train), 
    batch_size=32, 
    shuffle=True,
    pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test), 
    batch_size=32, 
    shuffle=False
)

# ==========================================
# 3. Define the Neural Network
# ==========================================
class DiagnosticNet(nn.Module):
    def __init__(self, input_size):
        super(DiagnosticNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2) 
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DiagnosticNet(input_size=X_train.shape[1]).to(device)

# ==========================================
# 4. Optimizer and Loss Function
# ==========================================
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 5. Training Loop
# ==========================================
epochs = 20

print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            outputs = model(X_test_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_test_batch.size(0)
            correct += (predicted == y_test_batch).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")

# ==========================================
# 6. Save the Model
# ==========================================
torch.save(model.state_dict(), 'breast_cancer_model.pth')
print("\nTraining Complete! Model saved as 'breast_cancer_model.pth'")