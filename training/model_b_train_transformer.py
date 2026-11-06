"""
Model B Training Script — Kaggle-Ready
=======================================
Train the ConditionalArchitectTransformer on the enriched dataset.

Usage on Kaggle:
  1. Upload 'enriched_conditional_data.pt' as a dataset
  2. Upload this script as a notebook or .py file
  3. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2 or P100)
  4. Run — trained weights will be saved to /kaggle/working/

The trained model file can then be downloaded and placed at:
  nlp_parser/nlp_parser/model_B/models/final_conditional_architect.pth
"""

import math
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Paths (auto-detect Kaggle vs local) ──
if os.path.exists("/kaggle"):
    # Kaggle environment — adjust the dataset name to match your upload
    DATA_PATH = "/kaggle/input/enriched-mario-data/enriched_conditional_data.pt"
    OUTPUT_DIR = "/kaggle/working"
else:
    # Local environment
    DATA_PATH = "nlp_parser/architect/data/enriched_conditional_data.pt"
    OUTPUT_DIR = "nlp_parser/nlp_parser/model_B/models"

# ── Hyperparameters ──
CONDITION_DIM   = 27
VOCAB_SIZE      = 6       # Tokens: Air(0), Solid(1), Loot(2), Enemy(3), Climbable(4), Player(5)
D_MODEL         = 128
NHEAD           = 4
NUM_LAYERS      = 4
DIM_FEEDFORWARD = 512
GRID_SIZE       = 16
SEQ_LEN         = GRID_SIZE * GRID_SIZE  # 256

EPOCHS          = 150
BATCH_SIZE      = 64
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-5
VAL_SPLIT       = 0.1     # 10% validation
SAVE_EVERY      = 25      # Save checkpoint every N epochs
PATIENCE        = 20      # Early stopping patience (epochs without improvement)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  MODEL ARCHITECTURE                                             ║
# ║  (Exact copy from inference.py — must stay in sync)             ║
# ╚══════════════════════════════════════════════════════════════════╝

class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding using separate row and column embeddings."""
    def __init__(self, d_model: int, max_height: int = 16, max_width: int = 16):
        super().__init__()
        self.row_embed = nn.Embedding(max_height, d_model // 2)
        self.col_embed = nn.Embedding(max_width, d_model // 2)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(seq_len, device=device) // 16
        cols = torch.arange(seq_len, device=device) % 16
        row_pos = self.row_embed(rows)
        col_pos = self.col_embed(cols)
        return torch.cat([row_pos, col_pos], dim=-1).unsqueeze(0)


class ConditionalArchitectTransformer(nn.Module):
    """
    Autoregressive transformer for 16×16 tile grid generation.
    Conditioned on a 27-dimensional spatial vector from Model A.
    """
    def __init__(
        self,
        condition_dim: int = CONDITION_DIM,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
    ):
        super().__init__()
        self.d_model = d_model
        self.condition_proj = nn.Linear(condition_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding2D(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_custom_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask[:, 0] = True
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

    def forward(self, condition: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        seq_len = src.size(1)
        cond_emb = self.condition_proj(condition).unsqueeze(1)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoder(seq_len, src.device)
        combined_src = torch.cat([cond_emb, src_emb], dim=1)
        mask = self.generate_custom_mask(seq_len + 1).to(src.device)
        output = self.transformer(combined_src, mask=mask)
        return self.fc_out(output[:, 1:, :])


# ╔══════════════════════════════════════════════════════════════════╗
# ║  DATASET                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

class TileDataset(Dataset):
    """Wraps the list of {condition, target_grid} dicts into a proper Dataset."""
    def __init__(self, data_list: list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        condition = sample["condition"]          # (27,)
        grid = sample["target_grid"].view(-1)    # (256,) flattened 16×16
        # Input: all tokens except the last (teacher forcing)
        # Target: all tokens except the first
        input_seq = grid[:-1]   # (255,)
        target_seq = grid[1:]   # (255,)
        return condition, input_seq, target_seq


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TRAINING LOOP                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def train():
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Load data ──
    print(f"\n📂 Loading dataset from: {DATA_PATH}")
    raw_data = torch.load(DATA_PATH, map_location="cpu")
    print(f"   Total samples: {len(raw_data)}")

    # ── Train/Val split ──
    dataset = TileDataset(raw_data)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"   Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ──
    model = ConditionalArchitectTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: ConditionalArchitectTransformer")
    print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   Config: d_model={D_MODEL}, heads={NHEAD}, layers={NUM_LAYERS}, ff={DIM_FEEDFORWARD}")

    # ── Optimizer & Loss ──
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # ── Output directory ──
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Training ──
    print(f"\n{'='*60}")
    print(f"  TRAINING START — {EPOCHS} epochs, batch_size={BATCH_SIZE}")
    print(f"  LR={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train phase ──
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for condition, input_seq, target_seq in train_loader:
            condition = condition.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()
            logits = model(condition, input_seq)  # (B, 255, vocab_size)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches

        # ── Validation phase ──
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for condition, input_seq, target_seq in val_loader:
                condition = condition.to(device)
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)

                logits = model(condition, input_seq)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1))
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ── Record history ──
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start

        # ── Logging ──
        improved = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_path = output_path / "best_conditional_architect.pth"
            torch.save(model.state_dict(), str(best_path))
            improved = " ✅ BEST"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(
                f"Epoch {epoch:>3d}/{EPOCHS} | "
                f"Train: {avg_train_loss:.6f} | "
                f"Val: {avg_val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"{elapsed:.1f}s{improved}"
            )

        # ── Checkpoint ──
        if epoch % SAVE_EVERY == 0:
            ckpt_path = output_path / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "history": history,
            }, str(ckpt_path))
            print(f"   💾 Checkpoint saved: {ckpt_path.name}")

        # ── Early stopping ──
        if patience_counter >= PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # ── Save final model ──
    final_path = output_path / "final_conditional_architect.pth"
    torch.save(model.state_dict(), str(final_path))
    print(f"\n💾 Final model saved: {final_path}")

    # Save best as the main file too
    best_path = output_path / "best_conditional_architect.pth"
    if best_path.exists():
        print(f"💾 Best model (val_loss={best_val_loss:.6f}): {best_path}")

    # ── Save training history ──
    history_path = output_path / "training_history.pt"
    torch.save(history, str(history_path))
    print(f"📊 Training history saved: {history_path}")

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Final training loss:  {history['train_loss'][-1]:.6f}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.6f}")
    print(f"  Total epochs run:     {len(history['train_loss'])}")
    print(f"{'='*60}")

    # ── Quick generation test ──
    print("\n🧪 Quick generation test with trained model...")
    test_generation(model, device)


def test_generation(model, device):
    """Generate a few sample levels to visually verify quality."""
    model.eval()
    char_map = {0: "  ", 1: "██", 2: "LL", 3: "EE", 4: "||", 5: "PP"}

    test_cases = [
        {
            "name": "Player bottom-left, 2 enemies mid-right",
            "vector": _make_vector({18 + 6: 1, 0 + 5: 2}),  # player bot-left, enemy mid-right
        },
        {
            "name": "Player mid-center, loot top-center",
            "vector": _make_vector({18 + 4: 1, 9 + 1: 2}),  # player mid-center, loot top-center
        },
        {
            "name": "Empty condition (baseline)",
            "vector": torch.zeros(27),
        },
    ]

    for test in test_cases:
        print(f"\n  Test: {test['name']}")
        condition = test["vector"].unsqueeze(0).to(device)
        generated = torch.tensor([[0]], dtype=torch.long).to(device)

        with torch.no_grad():
            for _ in range(SEQ_LEN - 1):
                logits = model(condition, generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)

        grid = generated.view(GRID_SIZE, GRID_SIZE).cpu().tolist()

        # Token counts
        flat = [t for row in grid for t in row]
        counts = {TOKENS_MAP.get(t, "?"): flat.count(t) for t in range(6) if flat.count(t) > 0}
        print(f"  Tokens: {counts}")

        # ASCII grid
        print("  " + "-" * 34)
        for row in grid:
            line = "".join(char_map.get(int(c), str(int(c))) for c in row)
            print(f"  |{line}|")
        print("  " + "-" * 34)


TOKENS_MAP = {0: "air", 1: "solid", 2: "loot", 3: "enemy", 4: "climb", 5: "player"}


def _make_vector(values: dict) -> torch.Tensor:
    """Helper: create a 27-dim vector with specific indices set."""
    v = torch.zeros(27)
    for idx, val in values.items():
        v[idx] = val
    return v


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    train()
