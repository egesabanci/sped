"""Core experiment engine — runs grid-search experiments and generates reports.

Separated from CLI to enable testing without model loading.
All model interactions happen through the SpeculativeDecoder interface.
"""

from pathlib import Path
from time import time
from typing import Optional
from sped.core.speculative_decoding import SpeculativeDecoder


class ExperimentEngine:
    """Runs grid-search experiments over speculative decoding hyperparameters.

    Loads models once, then runs all combinations by reconfiguring the
    decoder's parameters between runs.
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        draft_model,
        draft_tokenizer,
        device: str = "cpu",
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.device = device
        self._decoder_cache: dict[str, SpeculativeDecoder] = {}

    def _get_decoder(self, draft_k: int, align_strategy: str) -> SpeculativeDecoder:
        """Get or create a cached decoder for a given config."""
        key = f"{draft_k}_{align_strategy}"
        if key not in self._decoder_cache:
            vocab_aligner = None
            if align_strategy != "none":
                from sped.vocab_agnostic.alignment import VocabAligner
                vocab_aligner = VocabAligner(
                    target_tokenizer=self.target_tokenizer,
                    draft_tokenizer=self.draft_tokenizer,
                    strategy=align_strategy,
                )

            self._decoder_cache[key] = SpeculativeDecoder(
                target_model=self.target_model,
                target_tokenizer=self.target_tokenizer,
                draft_model=self.draft_model,
                draft_tokenizer=self.draft_tokenizer,
                vocab_aligner=vocab_aligner,
                max_draft_tokens=draft_k,
                device=self.device,
            )
        return self._decoder_cache[key]

    def run_single_experiment(
        self,
        draft_k: int,
        temperature: float,
        align_strategy: str,
        prompts: list[str],
        max_tokens: int = 128,
    ) -> dict:
        """Run a single experiment configuration over a set of prompts.

        Returns a dict with config, aggregated metrics, and per-prompt results.
        """
        decoder = self._get_decoder(draft_k, align_strategy)

        prompt_results = []
        for prompt in prompts:
            prompt_start = time()
            output = decoder.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            elapsed = time() - prompt_start

            response = output[len(prompt):] if output.startswith(prompt) else output
            tokens = len(self.target_tokenizer.encode(response))

            prompt_results.append({
                "prompt": prompt[:80],
                "tokens": tokens,
                "time_seconds": round(elapsed, 3),
                "tokens_per_second": round(tokens / max(elapsed, 0.001), 1),
            })

        # Aggregate
        avg_tps = sum(r["tokens_per_second"] for r in prompt_results) / len(prompt_results)
        avg_time = sum(r["time_seconds"] for r in prompt_results) / len(prompt_results)

        return {
            "config": {
                "draft_k": draft_k,
                "temperature": temperature,
                "align_strategy": align_strategy,
            },
            "avg_tokens_per_second": round(avg_tps, 1),
            "avg_time_seconds": round(avg_time, 3),
            "total_tokens": sum(r["tokens"] for r in prompt_results),
            "num_prompts": len(prompts),
            "per_prompt": prompt_results,
        }

    def close(self):
        """Release all resources."""
        self._decoder_cache.clear()


# ── Auto-tuner: Golden-section search for optimal K (#25) ──


class AutoTuner:
    """Finds the optimal draft K using golden-section search.

    Uses a simulated evaluation function to avoid loading models
    multiple times. The decoder is reconfigured for each K value.
    """

    def __init__(
        self,
        decoder: SpeculativeDecoder,
        prompts: list[str],
        max_tokens: int = 64,
    ):
        self.decoder = decoder
        self.prompts = prompts
        self.max_tokens = max_tokens
        self._eval_cache: dict[int, float] = {}

    def search(self, min_k: int = 2, max_k: int = 15) -> int:
        """Run golden-section search to find optimal K.

        Returns the draft K value that maximizes tokens per second.
        """
        a, b = min_k, max_k
        phi = (5 ** 0.5 - 1) / 2  # ~0.618
        tol = 1

        while b - a > tol:
            x1 = int(a + (1 - phi) * (b - a))
            x2 = int(a + phi * (b - a))
            x1 = max(x1, a + 1)
            x2 = min(x2, b - 1)

            if x1 not in self._eval_cache:
                self._eval_cache[x1] = self._evaluate_k(x1)
            if x2 not in self._eval_cache:
                self._eval_cache[x2] = self._evaluate_k(x2)

            if self._eval_cache[x1] > self._eval_cache[x2]:
                b = x2
            else:
                a = x1

        # Final evaluation at all candidates
        best_k = max(
            range(a, b + 1),
            key=lambda k: self._eval_cache.get(k, self._evaluate_k(k)),
        )
        return best_k

    def _evaluate_k(self, k: int) -> float:
        """Measure tokens per second at a given draft K."""
        self.decoder.max_draft_tokens = k
        self.decoder.reset_metrics()

        times = []
        for prompt in self.prompts:
            start = time()
            self.decoder.generate(
                prompt=prompt,
                max_new_tokens=self.max_tokens,
                temperature=0.0,
                verbose=False,
            )
            elapsed = time() - start
            times.append(elapsed)

        total_time = sum(times)
        total_tokens = len(self.prompts) * self.max_tokens
        return total_tokens / max(total_time, 0.001)

    @property
    def evaluated_k_values(self) -> list[tuple[int, float]]:
        """Return list of (K, tokens_per_second) pairs evaluated so far."""
        return sorted(self._eval_cache.items())

    def clear_cache(self):
        """Clear evaluation cache."""
        self._eval_cache.clear()


# ── HTML Report Generator (#24) ──────────────────────────


def generate_html_report(report: dict, path: Path):
    """Generate a self-contained HTML report with comparison tables.

    Includes color-coded results, metadata, and recommendations.
    """
    best = max(
        report["results"],
        key=lambda r: r["avg_tokens_per_second"],
    )

    rows = ""
    for i, r in enumerate(
        sorted(
            report["results"],
            key=lambda x: x["avg_tokens_per_second"],
            reverse=True,
        )
    ):
        highlight = "style='background:#d4edda;font-weight:bold'" if r == best else ""
        rank = i + 1
        speedup_vs_worst = (
            r["avg_tokens_per_second"]
            / max(
                min(
                    x["avg_tokens_per_second"]
                    for x in report["results"]
                ),
                0.01,
            )
            if report["results"]
            else 1.0
        )
        rows += f"""
        <tr {highlight}>
            <td>{rank}</td>
            <td>{r['config']['draft_k']}</td>
            <td>{r['config']['temperature']}</td>
            <td>{r['config']['align_strategy']}</td>
            <td>{r['avg_tokens_per_second']}</td>
            <td>{r['avg_time_seconds']}</td>
            <td>{speedup_vs_worst:.2f}x</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>sped Experiment Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8f9fa; color: #333; }}
  h1 {{ color: #2d3436; font-size: 1.8rem; margin-bottom: 0.5rem; }}
  h2 {{ color: #2d3436; font-size: 1.3rem; margin: 2rem 0 1rem; }}
  .meta {{ color: #636e72; font-size: 0.9rem; line-height: 1.6; margin-bottom: 1rem;
          padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  table {{ width: 100%; border-collapse: collapse; background: white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
  th {{ background: #2d3436; color: white; padding: 0.75rem 1rem; text-align: left;
        font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
  tr:hover {{ background: #f1f3f5; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px;
             font-size: 0.75rem; font-weight: 600; }}
  .badge-best {{ background: #00b894; color: white; }}
  .recommendations {{ background: white; border-radius: 8px; padding: 1.5rem;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 1rem;
                     border-left: 4px solid #00b894; }}
  .recommendations ul {{ margin-left: 1.5rem; margin-top: 0.5rem; }}
  .recommendations li {{ margin-bottom: 0.5rem; line-height: 1.5; }}
  footer {{ margin-top: 2rem; color: #b2bec3; font-size: 0.8rem; text-align: center; }}
  @media (max-width: 768px) {{ body {{ padding: 1rem; }} table {{ font-size: 0.8rem; }} }}
</style>
</head>
<body>
<h1>⚡ sped Experiment Report</h1>
<div class="meta">
  <strong>Target:</strong> {report['metadata']['target_model']}<br>
  <strong>Draft:</strong> {report['metadata']['draft_model']}<br>
  <strong>Device:</strong> {report['metadata']['device']}<br>
  <strong>Prompts:</strong> {report['config']['num_prompts']}
  &middot; <strong>Max tokens:</strong> {report['config']['max_tokens']}
  &middot; <strong>Date:</strong> {report['metadata']['timestamp'][:19]}
</div>

<h2>Results</h2>
<table>
  <thead>
    <tr><th>#</th><th>K</th><th>Temp</th><th>Align</th><th>Avg tok/s</th><th>Avg time (s)</th><th>Rel. speedup</th></tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

<h2>Recommendations</h2>
<div class="recommendations">
  <p>The best configuration is <strong>K={best['config']['draft_k']}</strong>
  at <strong>T={best['config']['temperature']}</strong> with
  <strong>{best['config']['align_strategy']}</strong> alignment,
  achieving <strong>{best['avg_tokens_per_second']} tok/s</strong>.</p>
  <ul>
    <li>Use <code>sped serve --draft-k {best['config']['draft_k']} --temperature {best['config']['temperature']}</code></li>
  </ul>
</div>

<footer>Generated by sped v0.1.0</footer>
</body>
</html>"""
    path.write_text(html)
