
import math, json, torch, torch.nn as nn
from pathlib import Path

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class Landmark2TextTransformer(nn.Module):
    def __init__(self, in_feat: int, vocab_size: int,
                 d_model=512, nhead=8, num_enc=6, num_dec=6, d_ff=2048, dropout=0.1, pad_id: int = 0):
        super().__init__()
        self.input_proj = nn.Linear(in_feat, d_model)
        self.pos_enc_in = SinusoidalPositionalEncoding(d_model)
        self.embed_out  = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc_out = SinusoidalPositionalEncoding(d_model)
        self.tf = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, src, src_key_padding_mask, tgt, tgt_key_padding_mask):
        src = self.pos_enc_in(self.input_proj(src))
        tgt = self.pos_enc_out(self.embed_out(tgt))
        L = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        out = self.tf(src=src, tgt=tgt,
                      src_key_padding_mask=src_key_padding_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      tgt_mask=tgt_mask)
        out = self.norm(out)
        return self.head(out)

def _maybe_int_keys(d):
    try:
        return {int(k): v for k, v in d.items()}
    except Exception:
        return d

def load_bundle(bundle_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(bundle_dir)
    ckpt = root / "assets" / "model_phase2_best.pt"
    c2i = root / "assets" / "char2idx.json"
    i2c = root / "assets" / "idx2char.json"

    blob = torch.load(ckpt, map_location="cpu")
    config = blob.get("config", {})
    in_feat = config.get("in_feat")
    vocab_size = config.get("vocab_size")

    # prefer JSON maps; else fallback to ckpt-embedded maps
    if c2i.exists() and i2c.exists():
        char2idx = json.loads(c2i.read_text())
        idx2char = _maybe_int_keys(json.loads(i2c.read_text()))
    else:
        char2idx = blob["char2idx"]
        idx2char = blob["idx2char"]

    PAD_ID = next((v for k, v in char2idx.items() if v == 0), 0)
    model = Landmark2TextTransformer(in_feat=in_feat, vocab_size=vocab_size, pad_id=PAD_ID).to(device)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    return model, char2idx, idx2char, config, device, in_feat, vocab_size
