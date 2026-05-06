import numpy as np

def cross_entropy_loss(logits: np.ndarray, target: int) -> float:
    exponent = np.exp(logits - np.max(logits))
    p = exponent/exponent.sum()
    return float(-np.log(p[target]))

def binary_cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(-np.mean(target*np.log(pred) + ((1-target)*np.log(1-pred))))

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    # mean squared error
    return float(np.mean((pred-target)**2))


# ── Cross Entropy Tests ───────────────────────────────────────
print("=" * 55)
print("Cross Entropy Loss")
print("=" * 55)
 
# Test 1: confident correct prediction → low loss
logits = np.array([10.0, 0.0, 0.0, 0.0])
loss   = cross_entropy_loss(logits, 0)
print(f"Test 1: confident correct (target=0)")
print(f"  logits={logits}, loss={loss:.4f}  (should be ~0.0)")
print(f"  PASS: {loss < 0.1}")
 
# Test 2: confident wrong prediction → high loss
logits = np.array([0.0, 0.0, 0.0, 10.0])
loss   = cross_entropy_loss(logits, 0)
print(f"\nTest 2: confident wrong (target=0, high logit at 3)")
print(f"  logits={logits}, loss={loss:.4f}  (should be ~10.0)")
print(f"  PASS: {loss > 9.0}")
 
# Test 3: uniform logits → loss = log(num_classes)
logits = np.array([1.0, 1.0, 1.0, 1.0])
loss   = cross_entropy_loss(logits, 0)
expected = np.log(4)
print(f"\nTest 3: uniform logits → loss = log(4) = {expected:.4f}")
print(f"  loss={loss:.4f}")
print(f"  PASS: {abs(loss - expected) < 1e-5}")
 
# Test 4: numerical stability with large logits
logits = np.array([1000.0, 999.0, 998.0])
loss   = cross_entropy_loss(logits, 0)
print(f"\nTest 4: numerical stability with large logits")
print(f"  logits={logits}, loss={loss:.4f}  (should not be nan/inf)")
print(f"  PASS: {not np.isnan(loss) and not np.isinf(loss)}")
 
# Test 5: loss always positive
logits = np.random.randn(100)
for target in range(10):
    loss = cross_entropy_loss(logits, target)
    assert loss > 0, f"Loss should be positive, got {loss}"
print(f"\nTest 5: loss always positive (100 random logits, 10 targets)")
print(f"  PASS: True")
 
 
# ── Binary Cross Entropy Tests ────────────────────────────────
print()
print("=" * 55)
print("Binary Cross Entropy")
print("=" * 55)
 
# Test 1: perfect predictions → near zero loss
pred   = np.array([0.999, 0.001, 0.999, 0.001])
target = np.array([1.0,   0.0,   1.0,   0.0  ])
loss   = binary_cross_entropy(pred, target)
print(f"Test 1: perfect predictions")
print(f"  pred={pred}, target={target}")
print(f"  loss={loss:.4f}  (should be ~0.0)")
print(f"  PASS: {loss < 0.02}")
 
# Test 2: completely wrong predictions → high loss
pred   = np.array([0.001, 0.999, 0.001, 0.999])
loss   = binary_cross_entropy(pred, target)
print(f"\nTest 2: completely wrong predictions")
print(f"  pred={pred}, target={target}")
print(f"  loss={loss:.4f}  (should be high > 5.0)")
print(f"  PASS: {loss > 5.0}")
 
# Test 3: random predictions → loss ~log(2)
pred   = np.array([0.5, 0.5, 0.5, 0.5])
target = np.array([1.0, 0.0, 1.0, 0.0])
loss   = binary_cross_entropy(pred, target)
print(f"\nTest 3: 50/50 predictions → loss = log(2) = {np.log(2):.4f}")
print(f"  loss={loss:.4f}")
print(f"  PASS: {abs(loss - np.log(2)) < 1e-5}")
 
# Test 4: asymmetric — loss higher when confident and wrong
pred_right = np.array([0.9])
pred_wrong = np.array([0.1])
target     = np.array([1.0])
loss_right = binary_cross_entropy(pred_right, target)
loss_wrong = binary_cross_entropy(pred_wrong, target)
print(f"\nTest 4: confident correct={loss_right:.4f} vs confident wrong={loss_wrong:.4f}")
print(f"  PASS: {loss_wrong > loss_right}")
 
 
# ── MSE Tests ─────────────────────────────────────────────────
print()
print("=" * 55)
print("MSE Loss")
print("=" * 55)
 
# Test 1: perfect predictions → zero loss
pred   = np.array([1.0, 2.0, 3.0])
target = np.array([1.0, 2.0, 3.0])
loss   = mse_loss(pred, target)
print(f"Test 1: perfect predictions")
print(f"  loss={loss:.4f}  (should be 0.0)")
print(f"  PASS: {loss == 0.0}")
 
# Test 2: known error → verify formula
pred   = np.array([2.0, 4.0])
target = np.array([1.0, 2.0])
loss   = mse_loss(pred, target)
expected = ((2-1)**2 + (4-2)**2) / 2   # (1 + 4) / 2 = 2.5
print(f"\nTest 2: manual calculation")
print(f"  pred={pred}, target={target}")
print(f"  expected={(expected):.4f}, got={loss:.4f}")
print(f"  PASS: {abs(loss - expected) < 1e-10}")
 
# Test 3: loss scales with error magnitude
pred1  = np.array([1.1])
pred2  = np.array([2.0])
target = np.array([1.0])
loss1  = mse_loss(pred1, target)
loss2  = mse_loss(pred2, target)
print(f"\nTest 3: larger error → larger loss")
print(f"  small error={loss1:.4f}, large error={loss2:.4f}")
print(f"  PASS: {loss2 > loss1}")
 
# Test 4: MSE penalizes outliers more than MAE
pred   = np.array([1.0, 1.0, 1.0, 10.0])   # one outlier
target = np.array([0.0, 0.0, 0.0,  0.0])
mse    = mse_loss(pred, target)
mae    = float(np.mean(np.abs(pred - target)))
print(f"\nTest 4: outlier sensitivity")
print(f"  MSE={mse:.4f}  MAE={mae:.4f}  (MSE penalizes outlier much more)")
print(f"  PASS: {mse > mae}")
 
 
# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 55)
print("When to use each loss — interview reference")
print("=" * 55)
print("  CE  — multi-class classification (LLM next token prediction)")
print("  BCE — binary classification, multi-label (spam detection)")
print("  MSE — regression (predicting continuous values)")
print()
print("All tests complete")
print("=" * 55)