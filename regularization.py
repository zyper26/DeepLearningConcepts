import numpy as np

def l1_regularization(weights: np.ndarray, lambda_: float) -> tuple:
    loss_penalty = lambda_ * np.sum(np.abs(weights))
    gradient = lambda_ * np.sign(weights)
    return loss_penalty, gradient

def l2_regularization(weights: np.ndarray, lambda_: float) -> tuple:
    loss_penalty = lambda_ * np.sum(weights**2)
    gradient = 2 * lambda_ * weights
    return loss_penalty, gradient

class Dropout:
    def __init__(self, p: float = 0.5):
        # p = probability of dropping a neuron
        self.p = p
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # during training: randomly zero out neurons, scale up survivors
        # during inference: pass through unchanged
        if training:
            mask = np.random.binomial(1, 1 - self.p, size=x.shape)
            return x*mask/(1-self.p)
        return x

if __name__ == "__main__":
    np.random.seed(42)
    weights = np.array([2.0, 0.5, -0.1, 0.01, -3.0])
    lr, lambda_ = 0.1, 0.5
 
    # L1 sparsity demo
    w = weights.copy()
    for _ in range(50):
        _, grad = l1_regularization(w, lambda_)
        w = w - lr * grad
    print(f"L1 after 50 steps: {w.round(4)}  zeros={np.sum(np.abs(w)<1e-4)}/5")
 
    # L2 shrinkage demo
    w = weights.copy()
    for _ in range(50):
        _, grad = l2_regularization(w, lambda_)
        w = w - lr * grad
    print(f"L2 after 50 steps: {w.round(4)}  zeros={np.sum(np.abs(w)<1e-4)}/5")
 
    # Dropout demo
    x       = np.ones(10)
    dropout = Dropout(p=0.5)
    print(f"Dropout training : {dropout.forward(x, training=True)}")
    print(f"Dropout inference: {dropout.forward(x, training=False)}")