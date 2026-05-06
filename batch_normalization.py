import numpy as np

class BatchNorm:
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1):
        self.gamma    = np.ones(num_features)   # learned scale
        self.beta     = np.zeros(num_features)  # learned shift
        self.eps      = eps
        self.momentum = momentum
        # running stats for inference
        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)

    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        # x shape: (batch, features)
        # training: normalize across batch dim=0, update running stats
        if training:
            batch_mean = x.mean(axis = 0)
            batch_var = x.var(axis = 0)

            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*batch_mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*batch_var
            x_norm = (x-batch_mean)/np.sqrt(batch_var + self.eps)
        else:
            x_norm = (x-self.running_mean)/np.sqrt(self.running_var + self.eps)
        return self.gamma*x_norm + self.beta

# ── Test 1: output mean ≈ 0, std ≈ 1 during training ─────────
print("=" * 55)
print("Test 1: output normalized to mean=0, std=1")
print("=" * 55)
bn  = BatchNorm(8)
x   = np.random.randn(32, 8) * 5 + 10   # mean=10, std=5
out = bn.forward(x, training=True)
mean_close = np.allclose(out.mean(axis=0), 0, atol=1e-5)
std_close  = np.allclose(out.std(axis=0),  1, atol=1e-3)
print(f"Output mean: {out.mean(axis=0).round(4)}")
print(f"Output std : {out.std(axis=0).round(4)}")
print(f"PASS: {mean_close and std_close}")
 
 
# ── Test 2: gamma and beta shift output ───────────────────────
print()
print("=" * 55)
print("Test 2: gamma scales, beta shifts output")
print("=" * 55)
bn        = BatchNorm(4)
bn.gamma  = np.array([2.0, 2.0, 2.0, 2.0])
bn.beta   = np.array([1.0, 1.0, 1.0, 1.0])
x         = np.random.randn(32, 4)
out       = bn.forward(x, training=True)
print(f"gamma=2, beta=1 → output mean should be ~1, std ~2")
print(f"Output mean: {out.mean(axis=0).round(4)}")
print(f"Output std : {out.std(axis=0).round(4)}")
print(f"PASS: {np.allclose(out.mean(axis=0), 1, atol=1e-4) and np.allclose(out.std(axis=0), 2, atol=1e-3)}")
 
 
# ── Test 3: running stats updated correctly ───────────────────
print()
print("=" * 55)
print("Test 3: running stats converge to true mean/var")
print("=" * 55)
bn       = BatchNorm(4, momentum=0.1)
true_mean = np.array([3.0, -2.0, 5.0, 0.0])
true_std  = np.array([2.0,  1.0, 3.0, 1.0])
for _ in range(200):
    x = np.random.randn(64, 4) * true_std + true_mean
    bn.forward(x, training=True)
print(f"True mean    : {true_mean}")
print(f"Running mean : {bn.running_mean.round(2)}")
print(f"True var     : {(true_std**2).round(2)}")
print(f"Running var  : {bn.running_var.round(2)}")
mean_ok = np.allclose(bn.running_mean, true_mean, atol=0.3)
var_ok  = np.allclose(bn.running_var, true_std**2, atol=1.0)
print(f"PASS: {mean_ok and var_ok}")
 
 
# ── Test 4: inference uses running stats not batch stats ──────
print()
print("=" * 55)
print("Test 4: inference uses running stats")
print("=" * 55)
bn = BatchNorm(4, momentum=0.1)
# train on data with mean=5
for _ in range(100):
    x = np.random.randn(32, 4) + 5
    bn.forward(x, training=True)
# inference with single sample
x_inf  = np.array([[5.0, 5.0, 5.0, 5.0]])
out    = bn.forward(x_inf, training=False)
print(f"Running mean: {bn.running_mean.round(2)}  (should be ~5)")
print(f"Inference output: {out.round(4)}  (should be ~0 since input≈mean)")
print(f"PASS: {np.allclose(out, 0, atol=0.5)}")
 
 
# ── Test 5: works with batch_size=1 at inference ──────────────
print()
print("=" * 55)
print("Test 5: batch_size=1 works at inference")
print("=" * 55)
bn = BatchNorm(8)
for _ in range(50):
    bn.forward(np.random.randn(32, 8), training=True)
x_single = np.random.randn(1, 8)
try:
    out = bn.forward(x_single, training=False)
    print(f"Input shape : {x_single.shape}")
    print(f"Output shape: {out.shape}")
    print(f"PASS: True")
except Exception as e:
    print(f"FAIL: {e}")
 
 
# ── Test 6: output shape preserved ───────────────────────────
print()
print("=" * 55)
print("Test 6: output shape matches input shape")
print("=" * 55)
for batch, features in [(32, 8), (64, 512), (1, 256)]:
    bn  = BatchNorm(features)
    x   = np.random.randn(batch, features)
    # train first for inference test
    bn.forward(x, training=True)
    out = bn.forward(x, training=False)
    match = out.shape == x.shape
    print(f"  ({batch}, {features}) → {out.shape}  PASS: {match}")
 
 
# ── Test 7: momentum effect ───────────────────────────────────
print()
print("=" * 55)
print("Test 7: higher momentum → faster convergence to batch stats")
print("=" * 55)
bn_slow = BatchNorm(4, momentum=0.01)
bn_fast = BatchNorm(4, momentum=0.5)
for _ in range(20):
    x = np.random.randn(32, 4) + 10
    bn_slow.forward(x, training=True)
    bn_fast.forward(x, training=True)
print(f"True mean           : 10.0")
print(f"Slow (momentum=0.01): {bn_slow.running_mean.mean():.4f}")
print(f"Fast (momentum=0.5) : {bn_fast.running_mean.mean():.4f}")
fast_closer = abs(bn_fast.running_mean.mean() - 10) < abs(bn_slow.running_mean.mean() - 10)
print(f"PASS: {fast_closer}")
 
 
# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 55)
print("BatchNorm vs LayerNorm — interview reference")
print("=" * 55)
print("  BatchNorm: normalize across batch dim=0")
print("             used in CNNs, MLPs")
print("             breaks with batch_size=1 at training")
print("             needs running stats for inference")
print()
print("  LayerNorm: normalize across feature dim=-1")
print("             used in transformers")
print("             works with any batch size")
print("             deterministic at inference")
print()
print("All tests complete")
print("=" * 55)