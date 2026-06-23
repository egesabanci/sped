"""Online Speculative Decoding (OSD) — online adaptation of draft model."""

import torch
from collections import deque


class OnlineAdapter:
    """Online adaptation of draft model during inference.

    Maintains a replay buffer of recent (context, accepted_tokens) examples
    and performs lightweight online updates to keep the draft aligned with
    the target model's distribution.
    """

    def __init__(
        self,
        draft_model,
        lr: float = 1e-5,
        buffer_size: int = 128,
        update_freq: int = 16,
        device: str = "cuda",
    ):
        self.draft_model = draft_model
        self.lr = lr
        self.buffer_size = buffer_size
        self.update_freq = update_freq
        self.device = device
        self.buffer = deque(maxlen=buffer_size)
        self.step_count = 0

        # Online optimizer (only LoRA params if PEFT is used)
        self.optimizer = torch.optim.AdamW(
            [p for p in draft_model.parameters() if p.requires_grad],
            lr=lr,
        )

    def observe(
        self,
        context_ids: torch.Tensor,
        accepted_ids: torch.Tensor,
    ):
        """Store a successful speculation for later online update."""
        self.buffer.append((context_ids.cpu(), accepted_ids.cpu()))
        self.step_count += 1

        if self.step_count % self.update_freq == 0:
            self._online_update()

    def _online_update(self):
        """Perform a lightweight gradient step on the replay buffer."""
        if len(self.buffer) < 4:
            return

        batch = list(self.buffer)[-min(32, len(self.buffer)):]
        total_loss = 0.0

        self.draft_model.train()
        for ctx, accepted in batch:
            ctx = ctx.to(self.device)
            accepted = accepted.to(self.device)
            combined = torch.cat([ctx, accepted], dim=-1)
            outputs = self.draft_model(combined)
            # Next-token prediction loss on accepted tokens
            shift_logits = outputs.logits[:, ctx.shape[-1]-1:-1, :]
            shift_labels = accepted
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            total_loss += loss

        avg_loss = total_loss / len(batch)
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.draft_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.draft_model.eval()

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "buffer": list(self.buffer),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.buffer = deque(state["buffer"], maxlen=self.buffer_size)
        self.step_count = state["step_count"]
