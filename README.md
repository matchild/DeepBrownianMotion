# DeepBrownianMotion
*A deep learning classification tool for anomalous diffusion trajectories.*


## Introduction
<p align="center">
  <picture>
    <img src="https://github.com/matchild/DeepBrownianMotion/blob/main/media/diffusion.png">
  </picture>
    <br>
      MSD against time for different fractional Brownian motion realizations ($\alpha$ = 0.5, 1.0, and 1.5).
</p>
Let's consider the example of a particle immersed in a medium. Since its dynamics is complicated and depends on a very large number of parameters, it is not straightforward to identify useful quantities that characterize its motion. One of the most important is the mean squared displacement (MSD),  which connects time to the average of the squared distance travelled.
Given a continuous random variable $\boldsymbol{X}(t)$ describing the position of a particle at time t, MSD is defined as:


   $\mathrm{MSD}\left(\tau\right)$ $=$ $\langle\left|\boldsymbol{X}\left(t+\tau\right)-\boldsymbol{X}\left(t\right)\right|^2\rangle$

where $\left|\right|^2$ refers to the squared norm and $\langle\rangle$ to the mean value over time. When ordinary diffusion takes place, $MSD(\tau)$ is proportional to $\tau$, but there exist cases where $\mathrm{MSD}\left(\tau\right)$ $\sim$ $\tau^\alpha$. 
This naturally leads to a classication of diffusion dynamics into three main classes:

- Subdiffusion when $\alpha < 1$
- Ordinary diffusion when $\alpha = 1$
- Superdiffusion when $\alpha > 1$

Whenever $\alpha \neq 1$ the process is called _anomalous diffusion_.


There have been attemps to use deep learning techniques to overperform traditional methods to infer the $\alpha$ parameter from a trajectory[^1]. This project is meant to extend the previously used _state of the art_ approach -an LSTM neual network- with a Transformer Encoder[^2]


[^1]: Volpe's paper
[^2]: Attention is all you need paper

## Training
Methods based on Fractional Brownian motion are a simple, though mathematically rigorous, way to simulate diffusion trajectories starting from the $\alpha$ parameter. A simulator[^3] was used to generate 200000, 100 step long, trajectories with $\alpha$ sampled from a uniform distribution 0-2.  A Transformer Encoder was then trained to predict $\alpha$ using a pytorch implementation. Preliminary analysis suggests that an embedding size close to 32 is enough to fully take advantage of the information contained in the trajectories.

<p align="center">
  <picture>
    <img src="https://github.com/matchild/DeepBrownianMotion/blob/main/media/training.png">
  </picture>
    <br>
      Mean squared error (MSE) as a function of the number of iterations. Vertical red dashed lines represent epochs.
</p>

[^3]: stochastic

## Requirements

```python
pip install numpy, stochastic, torch
```

## Usage
This script takes as input a vector containing the positions of the particle and returns the $\alpha$ parameter.

```python
from DeepBrownianMotion import DeepBrownianMotion
from stochastic.processes import FractionalBrownianMotion
import numpy as np

# Choose trajectory parameters
alpha = 0.7
trajectory_length = 100

# Instantiate simulator object
f = FractionalBrownianMotion(hurst=0.5*alpha, t=1)
trajectory = f.sample(trajectory_length-1).reshape(1, trajectory_length)

# Instantiate deep learning model
dpm = DeepBrownianMotion(device='cpu')
alpha_pred = np.around(dpm.inference(trajectory).item(), 3)

print('Real alpha {0}, predicted alpha {1}'.format(alpha, alpha_pred) )
```
Output:


_Real alpha 0.7, predicted alpha 0.746_

## Credits
I want to thank Marco Gherardi from the Univeristy of Milan for his support on a previous closely related project. This analysis was inspired by this paper from Volpe and his team and would not have been possible without the [stochastic](https://github.com/crflynn/stochastic) simulation tool. As always, I also want to thank the [pytorch](https://pytorch.org/) team for their amazing work.



