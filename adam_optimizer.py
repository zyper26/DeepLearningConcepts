"""
Optimizers from scratch
Adam  — adaptive moment estimation
AdamW — Adam + decoupled weight decay (production standard)
"""

import numpy as np

class Adam:
    """
    Adam optimizer.
    m = β1*m + (1-β1)*g          ← 1st moment (velocity)
    v = β2*v + (1-β2)*g²         ← 2nd moment (magnitude)
    θ = θ - lr * m̂ / (√v̂ + ε)  ← adaptive update
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = None   # first moment
        self.v     = None   # second moment
        self.t     = 0      # timestep

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        # initialize m and v on first call
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        # update m and v
        self.m = self.beta1*self.m + (1-self.beta1)*grads
        self.v = self.beta2*self.v + (1-self.beta2)*(grads**2)
        # bias correction
        self.t += 1
        m_hat = self.m/(1-self.beta1**self.t)
        v_hat = self.v/(1-self.beta2**self.t)
        # compute and return updated params
        params = params - self.lr *(m_hat/(np.sqrt(v_hat)+self.eps))
        return params

class AdamW:
    """
    AdamW optimizer — Adam + decoupled weight decay.
    Key difference from Adam:
      weight decay applied to params BEFORE gradient step
      → decay independent of adaptive learning rate
      → better generalization (Loshchilov & Hutter 2019)
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = None   # first moment
        self.v     = None   # second moment
        self.t     = 0      # timestep
        self.weight_decay = weight_decay

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        # initialize m and v on first call
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        # update m and v
        self.m = self.beta1*self.m + (1-self.beta1)*grads
        self.v = self.beta2*self.v + (1-self.beta2)*(grads**2)
        # bias correction
        self.t += 1
        m_hat = self.m/(1-self.beta1**self.t)
        v_hat = self.v/(1-self.beta2**self.t)
        # compute and return updated params
        params = params * (1- (self.lr * self.weight_decay))
        return params - self.lr *(m_hat/(np.sqrt(v_hat)+self.eps))
        


'''
Adam advantages:
  ✓ Handles sparse gradients (NLP, embeddings)
  ✓ Adaptive per-parameter LR — good for mixed-scale params
  ✓ Less sensitive to LR choice
  ✓ Works well in non-convex landscapes

SGD advantages:
  ✓ Better generalization on convex problems
  ✓ Less memory (no m, v storage)
  ✓ SGD + momentum often beats Adam on CV tasks
  '''


if __name__ == "__main__":
    def loss(x):     return x ** 2
    def gradient(x): return 2 * x
 
    x_adam  = np.array([10.0])
    x_adamw = np.array([10.0])
    adam    = Adam(lr=0.1)
    adamw   = AdamW(lr=0.1, weight_decay=0.1)
 
    for _ in range(100):
        x_adam  = adam.step(x_adam,   gradient(x_adam))
        x_adamw = adamw.step(x_adamw, gradient(x_adamw))
 
    print(f"Adam  final: {x_adam[0]:.6f}")
    print(f"AdamW final: {x_adamw[0]:.6f}  (weight decay pushes harder to 0)")
 