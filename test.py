import torch
import sys
sys.path.append("/public/home/kscszx_chh/xumingyang/AuxliaryDetection-main")
from conv_cuda_m.conv import Conv

test_data = torch.ones([1, 1, 3, 3])

conv = Conv(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_torch = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
print('conv based on PyTorch extension: \n', conv(test_data))
end.record()
# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))

start1 = torch.cuda.Event(enable_timing=True)
end1 = torch.cuda.Event(enable_timing=True)
start1.record()
print('conv based on PyTorch: \n', conv_torch(test_data))
end1.record()
# Waits for everything to finish running
torch.cuda.synchronize()

print(start1.elapsed_time(end1))
