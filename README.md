# DeepBrownianMotion
*A deep learning classification tool for anomalous diffusion trajectories.*

## Introduction

Let's consider the example of a particle immersed in a medium. Since its dynamics is complicated and depends on a very large number of parameters, it is not straightforward to identify useful quantities that characterize its motion. One of the most important is the mean squared displacement (MSD),  which connects time to the average of the squared distance travelled.
Given a continuous random variable $\boldsymbol{X}(t)$ describing the position of a particle at time $t$, MSD is defined as:

   $\mathrm{MSD}\left(\tau\right)$ $=$ $\langle\left|\boldsymbol{X}\left(t+\tau\right)-\boldsymbol{X}\left(t\right)\right|^2\rangle$

where $\left|\right|^2$ refers to the squared norm and $\langle\rangle$ to the mean value over time. When ordinary diffusion takes place, $MSD(\tau)$ is proportional to $\tau$, but there exist cases where

   $\mathrm{MSD}\left(\tau\right)$ $\sim$ $\tau^\alpha$

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

[^3]: stochastic

## Usage
This script takes as input a vector containing the positions of the particle and returns the $\alpha$ parameter.





## Credits
I wanted to thank Marco Gherardi from the Univeristy of Milan for his support on a previous closely related project. This analysis was inspired by this paper from Volpe and his team and would not have been possible without the stochastic simulation tool. As always I also want to thank the Pytorch team for their amazing work.



