from DeepBrownianMotion import DeepBrownianMotion
from stochastic.processes import FractionalBrownianMotion
import numpy as np

# Choose trajectory parameters
alpha=0.7
trajectory_length=100

# Instantiate simulator object
f = FractionalBrownianMotion(hurst=0.5*alpha, t=1)
trajectory = f.sample(trajectory_length-1).reshape(1, trajectory_length)

# Instantiate deep learning model
dpm = DeepBrownianMotion(device='cpu')
alpha_pred = np.around(dpm.inference(trajectory).item(), 3)

print('Real alpha {0}, predicted alpha {1}'.format(alpha, alpha_pred) )
