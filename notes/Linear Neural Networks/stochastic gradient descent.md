Stochastic Gradient Descent (SGD) reduces the computational latency of parameter updates by calculating the gradient on a single data point, or a small mini-batch, rather than the entire dataset. While this approach deviates from the mathematically "perfect" steepest descent direction of the full population, the resulting noise is often a functional advantage. High-precision direction is unnecessary for convergence; as long as the update maintains a general downward trajectory, the model can progress while simultaneously using the stochastic "jitter" to escape shallow local minima or saddle points that might trap a standard Batch Gradient Descent algorithm.

The effectiveness of this sampling method is grounded in the fact that a randomly selected subset of data serves as an **unbiased estimator** of the true gradient. Mathematically, if the samples are drawn independently and identically distributed (IID), the expected value of the stochastic update equals the true gradient of the entire cost function: **$E[\nabla_{\theta} L_i(\theta)] = \nabla_{\theta} J(\theta)$**. Because the sample resembles the global distribution, the model receives enough signal to descend the loss landscape, even if individual steps are noisy. To ensure convergence to the global minimum, the learning rate $\eta$ must typically be decayed over time to dampen this variance as the model nears the optimum.

Now well go through the mathematical definition of it:

### The model

$$
\hat{y}^{(i)} = \mathbf{w}^{\top}\mathbf{x}^{(i)} + b
$$
### The loss function
$$
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i = 1}^n J_{(w, b)}^{(i)} = \frac{1}{n} \sum_{i -1}^{n} \frac{1}{2} ((\mathbf{w}^{\top}\mathbf{x}^{i} + b) - y^{i})^{2}
$$
The Minibatch loss
Rather than computing the loss over all ${n}$ examples (which is slow), we sample a random subset $\mathcal{B} \subset \{1, 2, \dots, n\}$ of size $|\mathcal{B}|$. The minibatch loss is the average over this subset:
$$
\mathcal{L}_{\mathcal{B}}(\mathbf{w}, b) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \ell^{(i)}(\mathbf{w}, b) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{1}{2} \left( \mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)} \right)^2
$$
### Derivative of loss function

#### in terms of $\mathbf{w}$
this will be a simple chain rule. So to simplify things let: $\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)} = e^{(i)}$ 

therefore:
$$
\ell^{(i)} = \frac{1}{2}(e^{(i)})^{2}
$$
Now we apply the chain rule:
$$
\frac{\partial \ell^{(i)}}{\partial\mathbf{w}} = \frac{\partial \ell^{(i)}}{\partial e^{(i)}} \cdot \frac{\partial e^{(i)}}{\partial\mathbf{w}}
$$
$$
\therefore
$$
$$
\frac{\partial \ell^{(i)}}{\partial\mathbf{w}} = e^{(i)} \ \cdot \ \mathbf{w}^{(i)} \ =  \ (\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}) \ \mathbf{x}^{(i)}
$$
### in terms of $b$
we do the same chain rule here, but now its simpler because $\frac{\partial e^{(i)}}{\partial b} = 1$ and $\frac{\partial \ell^{(i)}}{\partial e^{(i)}} = e^{(i)}$ 
hence:
$$
\frac{\partial \ell^{(i)}}{\partial b} = e^{(i)} \cdot 1 = \mathbf{w}^{\top} \mathbf{x}^{(i)} + b - y^{(i)}
$$
### Average over the sample
Now we average the per-example gradients over the minibatch $\mathcal{B}$:
$$

$$
