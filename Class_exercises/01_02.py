import torch

W_1 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32)
b_1 = torch.tensor([0, 0], dtype=torch.float32)

W_2 = torch.tensor([1, -1], dtype=torch.float32)
b_2 = torch.tensor([0], dtype=torch.float32)

def forward(x):
    Z_1 = torch.matmul(x, W_1) + b_1
    H_1 = torch.relu(Z_1)
    y_pred = torch.matmul(H_1, W_2) + b_2
    return y_pred

x = torch.tensor([5, 5], dtype=torch.float32)

y_pred = forward(x)

print(y_pred)

#Back propagation task

