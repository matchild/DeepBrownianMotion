from DeepBrownianMotion import DeepBrownianMotion
import numpy as np



dpm = DeepBrownianMotion(device='cuda')
print(dpm.device)
x=np.random.rand(3, 102)
print(dpm.inference(x))
