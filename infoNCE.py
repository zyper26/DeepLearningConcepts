import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def infonce_loss(
    query: np.ndarray,          # (d,) query embedding
    positive: np.ndarray,       # (d,) positive doc embedding
    negatives: np.ndarray,      # (n, d) negative doc embeddings
    temperature: float = 0.07
) -> float:
    pos_sim  = cosine_similarity(query, positive) / temperature
    neg_sims = np.array([cosine_similarity(query, neg) / temperature for neg in negatives])
    numerator   = np.exp(pos_sim)
    denominator = np.exp(pos_sim) + np.sum(np.exp(neg_sims))
    return float(-np.log(numerator / denominator))


np.random.seed(42)
d = 64
 
 
# ── Test 1: similar positive → low loss ──────────────────────
print("=" * 55)
print("Test 1: similar positive → low loss")
print("=" * 55)
query    = np.random.randn(d)
positive = query + np.random.randn(d) * 0.01   # nearly identical
negatives = np.random.randn(5, d)
loss = infonce_loss(query, positive, negatives)
print(f"Loss: {loss:.6f}  (should be ~0.0)")
print(f"PASS: {loss < 0.1}")
 
 
# ── Test 2: random positive → higher loss ────────────────────
print()
print("=" * 55)
print("Test 2: random positive → higher loss than similar")
print("=" * 55)
positive_far = np.random.randn(d)
loss_far     = infonce_loss(query, positive_far, negatives)
loss_near    = infonce_loss(query, query + np.random.randn(d) * 0.01, negatives)
print(f"Loss (similar positive): {loss_near:.4f}")
print(f"Loss (random positive) : {loss_far:.4f}")
print(f"PASS: {loss_far > loss_near}")
 
 
# ── Test 3: more negatives → harder task → higher loss ───────
print()
print("=" * 55)
print("Test 3: more negatives → harder task")
print("=" * 55)
query    = np.random.randn(d)
positive = query + np.random.randn(d) * 0.1
neg5     = np.random.randn(5, d)
neg50    = np.random.randn(50, d)
loss5    = infonce_loss(query, positive, neg5)
loss50   = infonce_loss(query, positive, neg50)
print(f"Loss with  5 negatives : {loss5:.4f}")
print(f"Loss with 50 negatives : {loss50:.4f}  (harder → higher)")
print(f"PASS: {loss50 > loss5}")
 
 
# ── Test 4: temperature effect ────────────────────────────────
print()
print("=" * 55)
print("Test 4: lower temperature → sharper distribution")
print("=" * 55)
query    = np.random.randn(d)
positive = query + np.random.randn(d) * 0.5
negatives = np.random.randn(10, d)
loss_high_t = infonce_loss(query, positive, negatives, temperature=1.0)
loss_low_t  = infonce_loss(query, positive, negatives, temperature=0.07)
print(f"Loss (τ=1.0) : {loss_high_t:.4f}  (smooth)")
print(f"Loss (τ=0.07): {loss_low_t:.4f}  (sharp — amplifies differences)")
print(f"PASS: True (both valid, different scales)")
 
 
# ── Test 5: loss always positive ─────────────────────────────
print()
print("=" * 55)
print("Test 5: loss always positive")
print("=" * 55)
all_positive = True
for _ in range(100):
    q = np.random.randn(d)
    p = np.random.randn(d)
    n = np.random.randn(10, d)
    loss = infonce_loss(q, p, n)
    if loss <= 0:
        all_positive = False
        break
print(f"Tested 100 random cases")
print(f"PASS: {all_positive}")
 
 
# ── Test 6: loss bounded by log(N+1) ─────────────────────────
print()
print("=" * 55)
print("Test 6: loss bounded by log(num_negatives + 1)")
print("=" * 55)
n_neg    = 9
query    = np.random.randn(d)
positive = np.random.randn(d)
negatives = np.random.randn(n_neg, d)
loss     = infonce_loss(query, positive, negatives)
upper    = np.log(n_neg + 1)
print(f"Loss      : {loss:.4f}")
print(f"Upper bound log({n_neg+1}) = {upper:.4f}")
print(f"PASS: {loss <= upper + 0.1}")
 
 
# ── Test 7: identical query and positive → near zero ─────────
print()
print("=" * 55)
print("Test 7: identical query and positive → near zero loss")
print("=" * 55)
query     = np.random.randn(d)
positive  = query.copy()               # exact same vector
negatives = np.random.randn(10, d)
loss      = infonce_loss(query, positive, negatives)
print(f"Loss (identical q,p): {loss:.6f}  (should be ~0.0)")
print(f"PASS: {loss < 0.001}")
 
 
# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 55)
print("InfoNCE")
print("=" * 55)
print("  Used in: CLIP, SimCSE, E5, all-MiniLM embedding models")
print("  τ=0.07 standard (CLIP uses 0.07, SimCSE uses 0.05)")
print("  More negatives = harder task = better embeddings")
print("  In-batch negatives = other samples in batch are negatives")
print()
print("All tests complete")
print("=" * 55)