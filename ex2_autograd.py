import torch


x = torch.ones(2, 2)
# activate gradient calculation
x.requires_grad = True
print(x)

# y will have a grad_fn since it is not created by hand
y = x + 2
print(y)
print(y.grad_fn)

# z and out will also have a gradient
z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
print(a)
a = ((a*3)/(a -1))
print("a requires grad? {}".format(a.requires_grad))
a.requires_grad_(True)
print("a requires grad? {}".format(a.requires_grad))
b = (a*a).sum()
print(b.grad_fn)


