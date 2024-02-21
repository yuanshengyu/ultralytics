from ultralytics.nn.modules.block import FC2fEMA
from ultralytics.nn.modules.conv import Conv
import torch
import torch.nn as nn

x = torch.zeros([2, 8, 80, 80])
print(x.shape)

k = [5, 7, 9]
pool = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
ys = [m(x) for m in pool]
for y in ys:
    print(y.shape)
    
cv = Conv(x.shape[1], )