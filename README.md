# DeepBrownianMotion
*A deep learning classification tool for Brownian diffusion trajectories.*

## Introduction: diffusion

The dynamics of a particle immersed in a complex medium is extremely complicated and depends on a very large number of parameters. One of the most important observable quantities that can be inferred from such motion is the mean squared displacement (MSD) which connects time to the average of the squared distance travelled.
Given a continuous random variable $\boldsymbol{X}(t)$ describing the position of a particle at time $t$, MSD is defined as:

   $\mathrm{MSD}\left(\tau\right)$ $=$ $\langle\left|\boldsymbol{X}\left(t+\tau\right)-\boldsymbol{X}\left(t\right)\right|^2\rangle$

When ordinary diffusion takes place, $MSD(\tau)$ is proportional to $\tau$.
In general however:

   $\mathrm{MSD}\left(\tau\right)$ $\sim$ $\tau^\alpha$

Three possibilities therefore arise:

- Subdiffusion when $\alpha < 1$
- Ordinary diffusion when $\alpha = 1$
- Superdiffusion when $\alpha > 1$

Whenever $\alpha \neq 1$ the process is called \emph{anomalous diffusion}. Several models have been proposed to describe this behaviour, most notably Brownian motion (when $\alpha=1$), fractional Brownian motion (when $\alpha \neq 1$) and fractional Langevin equation (when the system evolves from $\alpha=1$ to $\alpha < 1$).


Fractional Brownian motion is a good compromise between simpicity and flexibility.
Being a stochastic process, the position at every time step is a random variable. By introducing a non zero covariance function between increments, fractional Brownian motion extends the concept of random walk allowing to generate different behaviours depending on the $\alpha$ parameter.
