
print("Hello world")
import torch

x = torch.tensor([[1,2],[4,5],[2,1]])
y = torch.tensor([3,9,3])

W_1 = torch.tensor([[1,-1,0],[1,-1,0]])

b_1 = torch.tensor([0,0,0])

W_2 = torch.tensor([1,-1,0])

b_2 = torch.tensor([0])
print(x,y)

print(W_1,b_1)

print(W_2,b_2)

relu = torch.nn.ReLU()

def forward(x):
    x = torch.matmul(x,W_1) + b_1
    x = relu(x)
    x = torch.matmul(x,W_2.t()) + b_2
    return x

print("Model 1")
result1 = forward(x)
print(result1)

print("Mean squared error")
mse = torch.nn.MSELoss()
loss = mse(result1.to(torch.float),y.to(torch.float))
print(loss)
print("Model 2")

V_1 = torch.tensor([[-4,6,5],[2,-4,7]])

c_1 = torch.tensor([4,-6,-29])

V_2 = torch.tensor([5,10,1])

c_2 = torch.tensor([-17])

def forward2(x):
    x = torch.matmul(x,V_1) + c_1
    x = relu(x)
    x = torch.matmul(x,V_2.t()) + c_2
    return x

model = forward2(x)
print(model)
## mean squared error
loss = mse(result1.to(torch.float),y.to(torch.float))
print(loss)
