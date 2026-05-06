"""
Knowledge Distillation: XLM-RoBERTa (Teacher) → BiLSTM (Student)
Three distillation components:
  1. Output distillation  — soft label KL divergence
  2. Feature distillation — intermediate representation MSE
  3. Combined training loop

Interview reference:
  - T² factor compensates for gradient scale reduction at high temperature
  - KLDiv expects log_softmax input, softmax target
  - BiLSTM h_n[-2], h_n[-1] = last layer forward + backward hidden states
  - 218x compression: 560M → 2.5M params
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── 1. Output Distillation Loss ───────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,    # (batch, num_classes)
    teacher_logits: torch.Tensor,    # (batch, num_classes)
    hard_labels: torch.Tensor,       # (batch,)
    temperature: float = 4.0,
    alpha: float = 0.7
) -> torch.Tensor:
    """
    L = α * T² * KL(student_soft || teacher_soft)
      + (1-α) * CE(student, hard_labels)

    T² compensates for gradient scale reduction at high temperature.
    KLDiv: input=log_softmax(student), target=softmax(teacher)
    """
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kl   = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
    ce   = nn.CrossEntropyLoss()(student_logits, hard_labels)
    return alpha * (temperature ** 2) * kl + (1 - alpha) * ce


# ── 2. Feature Distillation Loss ──────────────────────────────

def feature_distillation_loss(
    student_features: torch.Tensor,  # (batch, seq, student_hidden)
    teacher_features: torch.Tensor,  # (batch, seq, teacher_hidden)
    projection: nn.Linear            # maps student_hidden → teacher_hidden
) -> torch.Tensor:
    """
    Match intermediate representations after projection.
    Projection needed because LSTM hidden != RoBERTa hidden size.
    """
    student_proj = projection(student_features)
    return nn.MSELoss()(student_proj, teacher_features)


# ── 3. BiLSTM Student Model ───────────────────────────────────

class BiLSTMStudent(nn.Module):
    """
    Bidirectional LSTM student model.
    218x smaller than XLM-RoBERTa (2.5M vs 560M params).

    h_n shape: (num_layers*2, batch, hidden_dim)
    h_n[-2] = last layer forward  ← reads left→right
    h_n[-1] = last layer backward ← reads right→left
    concat  = (batch, 2*hidden_dim) full bidirectional context
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_classes, num_layers=2):
        super().__init__()
        self.embedding           = nn.Embedding(vocab_size, embed_dim)
        self.bidirectional_lstm  = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers   = num_layers,
            bidirectional = True,
            batch_first  = True
        )
        self.linear              = nn.Linear(2 * hidden_dim, embed_dim)
        self.dropout             = nn.Dropout(0.3)
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x                    = self.embedding(input_ids)
        lstm_out, (h_n, c_n) = self.bidirectional_lstm(x)
        h_final              = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        x                    = self.dropout(self.linear(h_final))
        logits               = self.classification_head(x)
        return logits, lstm_out   # return both for distillation


# ── 4. Distillation Training Step ─────────────────────────────

def distillation_training_step(
    teacher_model,
    student_model,
    projection,
    optimizer,
    batch,
    temperature=4.0,
    alpha=0.7,
    beta=0.5            # weight for feature distillation
):
    """
    Single training step combining output + feature distillation.
    Teacher is frozen — only student and projection are updated.
    """
    input_ids   = batch["input_ids"]
    hard_labels = batch["labels"]

    # teacher forward — frozen, no gradients
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True
        )
        teacher_logits = teacher_outputs.logits
        # use middle layer for feature distillation (layer 6 of 12)
        teacher_feats  = teacher_outputs.hidden_states[6]

    # student forward
    student_logits, lstm_out = student_model(
        input_ids,
        attention_mask=batch.get("attention_mask")
    )

    # combined loss
    loss_output  = distillation_loss(
        student_logits, teacher_logits, hard_labels, temperature, alpha
    )
    loss_feature = feature_distillation_loss(lstm_out, teacher_feats, projection)
    total_loss   = loss_output + beta * loss_feature

    # backprop
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), loss_output.item(), loss_feature.item()


# ── Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size, embed_dim, hidden_dim = 1000, 128, 256
    num_classes, teacher_hidden       = 5, 768
    batch, seq                        = 4, 20

    input_ids      = torch.randint(0, vocab_size, (batch, seq))
    hard_labels    = torch.randint(0, num_classes, (batch,))
    teacher_logits = torch.randn(batch, num_classes)
    teacher_feats  = torch.randn(batch, seq, teacher_hidden)
    projection     = nn.Linear(2 * hidden_dim, teacher_hidden)
    student        = BiLSTMStudent(vocab_size, embed_dim, hidden_dim, num_classes)

    student_logits, lstm_out = student(input_ids)

    dist_loss  = distillation_loss(student_logits, teacher_logits, hard_labels)
    feat_loss  = feature_distillation_loss(lstm_out, teacher_feats, projection)
    total_loss = dist_loss + 0.5 * feat_loss
    total_loss.backward()

    print(f"Distillation loss : {dist_loss.item():.4f}")
    print(f"Feature dist loss : {feat_loss.item():.4f}")
    print(f"Total loss        : {total_loss.item():.4f}")
    print(f"All grads present : {all(p.grad is not None for p in student.parameters())}")
    print(f"Student params    : {sum(p.numel() for p in student.parameters()):,}")
    print(f"Compression       : {560_000_000 // sum(p.numel() for p in student.parameters())}x vs XLM-RoBERTa")