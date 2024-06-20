import torch
from thop import profile
from nets.pspnet import Resnet

# 定义模型
model = Resnet()
model.eval()
# 创建一个随机的输入张量，以确定模型输入大小
input_tensor = torch.randn(1, 3, 289, 245)

# 计算模型的FLOPs和参数数量
flops, params = profile(model, inputs=(input_tensor,))

print('FLOPs: %.2fG' % (flops / 1e9))
print(f"Number of parameters: {params}")