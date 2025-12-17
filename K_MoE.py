import torch 
import torch.nn as nn 
import torch.optim as optim

class Expert(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super(Expert, self).__init__() 
        self.layer1 = nn.Linear(input_dim, hidden_dim) 
        self.layer2 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x): 
        x = torch.relu(self.layer1(x)) 
        return torch.softmax(self.layer2(x), dim=1)

# Define the gating model 
class Gating(nn.Module): 
    def __init__(self, input_dim, 
                 num_experts, dropout_rate=0.1): 
        super(Gating, self).__init__() 

    # Layers 
        self.layer1 = nn.Linear(input_dim, 128) 
        self.dropout1 = nn.Dropout(dropout_rate) 

        self.layer2 = nn.Linear(128, 256) 
        self.leaky_relu1 = nn.LeakyReLU() 
        self.dropout2 = nn.Dropout(dropout_rate) 

        self.layer3 = nn.Linear(256, 128) 
        self.leaky_relu2 = nn.LeakyReLU() 
        self.dropout3 = nn.Dropout(dropout_rate) 

        self.layer4 = nn.Linear(128, num_experts) 

    def forward(self, x): 
        x = torch.relu(self.layer1(x)) 
        x = self.dropout1(x) 

        x = self.layer2(x) 
        x = self.leaky_relu1(x) 
        x = self.dropout2(x) 

        x = self.layer3(x) 
        x = self.leaky_relu2(x) 
        x = self.dropout3(x) 

        return torch.softmax(self.layer4(x), dim=1)

class MoE(nn.Module): 
    def __init__(self, trained_experts): 
        super(MoE, self).__init__() 
        self.experts = nn.ModuleList(trained_experts) 
        num_experts = len(trained_experts) 
        # Assuming all experts have the same input dimension 
        input_dim = trained_experts[0].layer1.in_features 
        self.gating = Gating(input_dim, num_experts) 

    def forward(self, x): 
        # Get the weights from the gating network 
        weights = self.gating(x) 

        # Calculate the expert outputs 
        outputs = torch.stack( 
            [expert(x) for expert in self.experts], dim=2) 

        # Adjust the weights tensor shape to match the expert outputs 
        weights = weights.unsqueeze(1).expand_as(outputs) 

        # Multiply the expert outputs with the weights and 
        # sum along the third dimension 
        return torch.sum(outputs * weights, dim=2)

# 生成数据集
num_samples = 3000 
input_dim = 4 
hidden_dim = 32 

# Generate equal numbers of labels 0, 1, and 2 
y_data = torch.cat([ 
    torch.zeros(num_samples // 3), 
    torch.ones(num_samples // 3), 
    torch.full((num_samples - 2 * (num_samples // 3),), 2)  # Filling the remaining to ensure exact num_samples 
]).long() 

# Biasing the data based on the labels 
x_data = torch.randn(num_samples, input_dim) 

for i in range(num_samples): 
    if y_data[i] == 0: 
        x_data[i, 0] += 1  # Making x[0] more positive 
    elif y_data[i] == 1: 
        x_data[i, 1] -= 1  # Making x[1] more negative 
    elif y_data[i] == 2: 
        x_data[i, 0] -= 1  # Making x[0] more negative 

# Shuffle the data to randomize the order 
indices = torch.randperm(num_samples) 
x_data = x_data[indices] 
y_data = y_data[indices] 

# Verify the label distribution 
y_data.bincount() 

# Shuffle the data to ensure x_data and y_data remain aligned 
shuffled_indices = torch.randperm(num_samples) 
x_data = x_data[shuffled_indices] 
y_data = y_data[shuffled_indices] 

# Splitting data for training individual experts 
# Use the first half samples for training individual experts 
x_train_experts = x_data[:int(num_samples/2)] # 准备训练专家的输入数据
y_train_experts = y_data[:int(num_samples/2)] 

mask_expert1 = (y_train_experts == 0) | (y_train_experts == 1) # 布尔数组，0 或 1 的所有位置为 True
mask_expert2 = (y_train_experts == 1) | (y_train_experts == 2) 
mask_expert3 = (y_train_experts == 0) | (y_train_experts == 2) 

# Select an almost equal number of samples for each expert 
num_samples_per_expert = min(mask_expert1.sum(), mask_expert2.sum(), mask_expert3.sum()) 

x_expert1 = x_train_experts[mask_expert1][:num_samples_per_expert] # expert1 最终的训练数据
y_expert1 = y_train_experts[mask_expert1][:num_samples_per_expert] 

x_expert2 = x_train_experts[mask_expert2][:num_samples_per_expert] # expert2 最终的训练数据
y_expert2 = y_train_experts[mask_expert2][:num_samples_per_expert] 

x_expert3 = x_train_experts[mask_expert3][:num_samples_per_expert] # expert3 最终的训练数据
y_expert3 = y_train_experts[mask_expert3][:num_samples_per_expert] 

# 剩下的另一半数据集 
x_remaining = x_data[int(num_samples/2)+1:] 
y_remaining = y_data[int(num_samples/2)+1:] 

split = int(0.8 * len(x_remaining)) # 80% 用于 MoE 模型的训练
x_train_moe = x_remaining[:split] 
y_train_moe = y_remaining[:split] 

x_test = x_remaining[split:] # 20% 用于测试
y_test = y_remaining[split:] 

print(x_train_moe.shape,"\n", x_test.shape,"\n", 
      x_expert1.shape,"\n", 
      x_expert2.shape,"\n", x_expert3.shape)

# Define hidden dimension 
output_dim = 3 
hidden_dim = 32 

epochs = 500 # 训练轮数
learning_rate = 0.001 

# Instantiate the experts 
expert1 = Expert(input_dim, hidden_dim, output_dim) 
expert2 = Expert(input_dim, hidden_dim, output_dim) 
expert3 = Expert(input_dim, hidden_dim, output_dim) 

# Set up loss 损失函数
criterion = nn.CrossEntropyLoss() 

# Optimizers for experts 
optimizer_expert1 = optim.Adam(expert1.parameters(), lr=learning_rate) 
optimizer_expert2 = optim.Adam(expert2.parameters(), lr=learning_rate) 
optimizer_expert3 = optim.Adam(expert3.parameters(), lr=learning_rate)

# ------------------------分开训练各个专家------------------------
# Training loop for expert 1 
for epoch in range(epochs): 
    optimizer_expert1.zero_grad() 
    outputs_expert1 = expert1(x_expert1) 
    loss_expert1 = criterion(outputs_expert1, y_expert1) 
    loss_expert1.backward() 
    optimizer_expert1.step() 

# Training loop for expert 2 
for epoch in range(epochs): 
    optimizer_expert2.zero_grad() 
    outputs_expert2 = expert2(x_expert2) 
    loss_expert2 = criterion(outputs_expert2, y_expert2) 
    loss_expert2.backward() 
    optimizer_expert2.step() 

# Training loop for expert 3 
for epoch in range(epochs): 
    optimizer_expert3.zero_grad() 
    outputs_expert3 = expert3(x_expert3) 
    loss_expert3 = criterion(outputs_expert3, y_expert3) 
    loss_expert3.backward()

# Create the MoE model with the trained experts 
moe_model = MoE([expert1, expert2, expert3]) 

# Train the MoE model 
optimizer_moe = optim.Adam(moe_model.parameters(), lr=learning_rate) 
for epoch in range(epochs): 
    optimizer_moe.zero_grad() 
    outputs_moe = moe_model(x_train_moe) 
    loss_moe = criterion(outputs_moe, y_train_moe) 
    loss_moe.backward() 
    optimizer_moe.step()

# Evaluate all models 
def evaluate(model, x, y): 
    with torch.no_grad(): 
        outputs = model(x) 
        _, predicted = torch.max(outputs, 1) 
        correct = (predicted == y).sum().item() 
        accuracy = correct / len(y) 
    return accuracy

accuracy_expert1 = evaluate(expert1, x_test, y_test) 
accuracy_expert2 = evaluate(expert2, x_test, y_test) 
accuracy_expert3 = evaluate(expert3, x_test, y_test) 
accuracy_moe = evaluate(moe_model, x_test, y_test) 

print("Expert 1 Accuracy:", accuracy_expert1) 
print("Expert 2 Accuracy:", accuracy_expert2) 
print("Expert 3 Accuracy:", accuracy_expert3) 
print("Mixture of Experts Accuracy:", accuracy_moe) 
