# DeepBrownianMotion
*A deep learning classification tool for anomalous diffusion trajectories.*

## Introduction

The dynamics of a particle immersed in a complex medium is extremely complicated and depends on a very large number of parameters. One of the most important observable quantities is the mean squared displacement (MSD),  which connects time to the average of the squared distance travelled.
Given a continuous random variable $\boldsymbol{X}(t)$ describing the position of a particle at time $t$, MSD is defined as:

   $\mathrm{MSD}\left(\tau\right)$ $=$ $\langle\left|\boldsymbol{X}\left(t+\tau\right)-\boldsymbol{X}\left(t\right)\right|^2\rangle$

When ordinary diffusion takes place, $MSD(\tau)$ is proportional to $\tau$.
In general however:

   $\mathrm{MSD}\left(\tau\right)$ $\sim$ $\tau^\alpha$

This leads to a classication of diffusion dynamics into three main classes:

- Subdiffusion when $\alpha < 1$
- Ordinary diffusion when $\alpha = 1$
- Superdiffusion when $\alpha > 1$

Whenever $\alpha \neq 1$ the process is called \emph{anomalous diffusion}. Fractional Brownian motion models are a good compromise between simpicity and flexibility to simulate diffusion trajectories.

## Usage



## Training


## Credits


