"""Draft tree for multi-branch candidate generation and tree attention.

Instead of a single linear draft sequence, the draft model produces a tree
of continuations — branching only at positions where confidence is low.
The target model verifies the entire tree in one pass via tree attention.
"""

from typing import Optional
import torch
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """A single node in the draft tree."""

    token_id: int
    confidence: float
    depth: int
    parent: Optional["TreeNode"] = None
    children: list["TreeNode"] = field(default_factory=list)
    branch_id: int = 0


class DraftTree:
    """Builds and manages a tree of draft token candidates.

    The tree branches only at positions where confidence < threshold,
    enabling the target model to verify multiple possible continuations
    in a single forward pass.

    Reference: EAGLE-2 dynamic draft tree approach.
    """

    def __init__(
        self,
        root_ids: torch.Tensor,
        draft_model,
        draft_tokenizer,
        device: str = "cpu",
        max_depth: int = 5,
        max_branches: int = 5,
        branch_threshold: float = 0.3,
        min_confidence: float = 0.05,
    ):
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.device = device
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.branch_threshold = branch_threshold
        self.min_confidence = min_confidence

        # Root is the last known committed token
        self.root = TreeNode(
            token_id=root_ids[0, -1].item(),
            confidence=1.0,
            depth=0,
        )
        self.nodes: list[TreeNode] = [self.root]
        self._next_branch_id = 0

    def build(self, prefix_ids: torch.Tensor):
        """Build the draft tree by autoregressively expanding from root.

        At each depth, decide whether to branch based on confidence.
        """
        # Start from the root context
        context = prefix_ids.clone()

        # BFS expansion
        current_level = [self.root]
        self._next_branch_id = 0

        for depth in range(self.max_depth):
            next_level = []
            if len(current_level) > self.max_branches:
                # Prune to top-K by confidence
                current_level.sort(key=lambda n: n.confidence, reverse=True)
                current_level = current_level[:self.max_branches]

            for node in current_level:
                # Build the context for this node
                node_path = self._path_to_root(node)
                node_context = torch.cat([
                    prefix_ids,
                    torch.tensor([node_path], device=self.device),
                ], dim=-1)

                # Draft forward pass
                with torch.no_grad():
                    outputs = self.draft_model(node_context)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)

                # Get top-k candidates
                top_probs, top_indices = torch.topk(probs, k=min(5, len(probs)))

                for prob, idx in zip(top_probs, top_indices):
                    if prob.item() < self.min_confidence:
                        continue

                    child = TreeNode(
                        token_id=idx.item(),
                        confidence=prob.item(),
                        depth=depth + 1,
                        parent=node,
                        branch_id=self._next_branch_id,
                    )
                    self._next_branch_id += 1
                    node.children.append(child)
                    self.nodes.append(child)
                    next_level.append(child)

            current_level = next_level
            if not current_level:
                break

    def flatten_with_attention_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten the tree into a single sequence with positional masks.

        Returns:
            flat_sequence: (1, total_nodes) — all node token IDs in BFS order.
            attention_mask: (total_nodes, total_nodes) — causal mask where each
                          node attends to its ancestors.
        """
        # BFS order
        flat = []
        node_to_idx = {}
        queue = [self.root]
        idx = 0

        while queue:
            node = queue.pop(0)
            node_to_idx[node] = idx
            flat.append(node.token_id)
            idx += 1
            queue.extend(node.children)

        total = len(flat)
        attention_mask = torch.zeros((total, total), device=self.device, dtype=torch.bool)

        # Build ancestor mask
        for node, i in node_to_idx.items():
            attention_mask[i, i] = True  # attend to self
            parent = node.parent
            while parent is not None:
                j = node_to_idx[parent]
                attention_mask[i, j] = True
                parent = parent.parent

        return torch.tensor([flat], device=self.device), attention_mask

    def best_path(self, target_logits: torch.Tensor) -> list[int]:
        """Find the best path through the tree given target model logits.

        Accepts the longest prefix with highest confidence, preferring
        paths where the target model agrees with the draft.

        Args:
            target_logits: (total_nodes, vocab_size) — target logits at
                          each tree node position.

        Returns:
            accepted_token_ids: Longest accepted path.
        """
        _, flat_indices = self.flatten_with_attention_mask()
        flat_indices = flat_indices[0]

        node_list = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            node_list.append(node)
            queue.extend(node.children)

        # Walk the tree: at each depth, pick the child with highest
        # agreement between target and draft
        current = self.root
        path = [current.token_id]

        for depth_idx in range(1, self.max_depth + 1):
            level_nodes = [n for n in node_list if n.depth == depth_idx and n.parent == current]
            if not level_nodes:
                break

            # Score each child by agreement
            best_child = None
            best_score = float("-inf")

            for child in level_nodes:
                idx_in_flat = node_list.index(child)
                if idx_in_flat >= target_logits.shape[0]:
                    continue

                target_probs = torch.softmax(target_logits[idx_in_flat], dim=-1)
                agreement = target_probs[child.token_id].item()
                score = agreement * child.confidence

                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                break

            path.append(best_child.token_id)
            current = best_child

        return path

    def _path_to_root(self, node: TreeNode) -> list[int]:
        """Get the token IDs from root to this node (exclusive of root)."""
        path = []
        current = node
        while current.parent is not None:
            path.append(current.token_id)
            current = current.parent
        return list(reversed(path))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        return max(n.depth for n in self.nodes) if self.nodes else 0

    def prune(self):
        """Remove nodes below min_confidence threshold."""
        self.nodes = [n for n in self.nodes if n.confidence >= self.min_confidence or n == self.root]
        # Also clean up parent-child relationships
        valid_ids = {n.token_id for n in self.nodes}
        for node in self.nodes:
            node.children = [c for c in node.children if c in self.nodes]
