import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Prepare data
speed_df['timestamp'] = pd.to_datetime(speed_df['TimeStamp'], format='%H_%M_%S')
speed_df['hour'] = speed_df['timestamp'].dt.hour
speed_df['minute'] = speed_df['timestamp'].dt.minute
speed_df['day_of_week'] = speed_df['timestamp'].dt.dayofweek
speed_df['is_weekend'] = speed_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

features = speed_df[['day_of_week', 'is_weekend', 'hour', 'minute']]
labels = speed_df['TotalDuration']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define the neural network
class DurationNN(nn.Module):
    def __init__(self, input_size):
        super(DurationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = features.shape[1]
model = DurationNN(input_size)

# Training settings
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to PyTorch tensors
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
labels_tensor = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(features_tensor)
    loss = criterion(outputs, labels_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to predict duration
def predict_duration(day, hour, minute):
    model.eval()
    input_features = scaler.transform([[day, hour, minute]])
    input_tensor = torch.tensor(input_features, dtype=torch.float32)
    with torch.no_grad():
        duration = model(input_tensor).item()
    return duration
