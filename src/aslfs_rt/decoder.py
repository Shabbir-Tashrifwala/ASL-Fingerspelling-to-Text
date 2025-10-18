
from collections import deque
import numpy as np, torch, torch.nn.functional as F

class TemporalSmoother:
    def __init__(self, vocab_size, k=5, ema=False, alpha=0.4):
        self.vocab_size = vocab_size
        self.k = k
        self.ema = ema
        self.alpha = alpha
        self.buf = deque(maxlen=k)
        self._ema_vec = None
    def update(self, probs):
        p = probs.detach().cpu().float().numpy()
        if self.ema:
            if self._ema_vec is None:
                self._ema_vec = p.copy()
            else:
                self._ema_vec = self.alpha * p + (1 - self.alpha) * self._ema_vec
            return self._ema_vec
        else:
            self.buf.append(p)
            return np.mean(self.buf, axis=0) if len(self.buf) else p

class RealTimeDecoder:
    def __init__(self, model, char2idx, idx2char, device, in_feat, vocab_size,
                 window=64, smooth_k=5, commit_thresh=0.60, commit_patience=3):
        self.model = model
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.device = device
        self.in_feat = in_feat
        self.vocab_size = vocab_size
        self.window = window
        self.queue = deque(maxlen=window)
        self.PAD_ID = next((v for k,v in char2idx.items() if v==0), 0)
        self.BOS_ID = char2idx.get("<bos>", 1)
        self.EOS_ID = char2idx.get("<eos>", 2)
        self.smoother = TemporalSmoother(vocab_size=vocab_size, k=smooth_k)
        self.commit_thresh = commit_thresh
        self.commit_patience = commit_patience
        self._cand_id = None
        self._cand_count = 0
        self.committed = []
    def reset_text(self):
        self.committed.clear()
        self._cand_id, self._cand_count = None, 0
        self.smoother.buf.clear()
    def push_frame(self, x_feat):
        assert x_feat.shape[0] == self.in_feat
        self.queue.append(x_feat)
    @torch.no_grad()
    def step(self):
        if len(self.queue) < 4:
            return None, "".join(self.idx2char.get(t, "") for t in self.committed)
        frames = np.stack(self.queue, axis=0)
        src = torch.from_numpy(frames).unsqueeze(0).to(self.device)
        src_mask = torch.zeros((1, src.size(1)), dtype=torch.bool, device=self.device)
        ys = torch.tensor([[self.BOS_ID] + self.committed], dtype=torch.long, device=self.device)
        ymask = torch.zeros_like(ys, dtype=torch.bool, device=self.device)
        logits = self.model(src, src_mask, ys, ymask)
        next_logits = logits[0, -1]
        probs = F.softmax(next_logits, dim=-1)
        smoothed = self.smoother.update(probs)
        top_id = int(smoothed.argmax()); top_p = float(smoothed[top_id])
        if top_id == self._cand_id:
            self._cand_count += 1
        else:
            self._cand_id = top_id; self._cand_count = 1
        committed_changed = False
        if top_id not in (self.PAD_ID, self.BOS_ID) and top_p >= self.commit_thresh and self._cand_count >= self.commit_patience:
            if not self.committed or self.committed[-1] != top_id:
                if top_id != self.EOS_ID:
                    self.committed.append(top_id); committed_changed = True
            self._cand_count = 0
        text = "".join(self.idx2char.get(t, "") for t in self.committed if t not in (self.PAD_ID, self.BOS_ID, self.EOS_ID))
        return (top_id, top_p, committed_changed), text
