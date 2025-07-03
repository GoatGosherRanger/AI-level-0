import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
x = torch.empty(5,3)
print(x)
x = torch.rand(5,3)
print(x)
x = torch.zeros(5,3,dtype = torch.long)
print(x)
x = torch.tensor([5.5,3])
print(x)

x = torch.ones(2,2,requires_grad = True) 
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y*y*3
out = z.mean()
print(z,out)

out.backward()
print(x.grad)
