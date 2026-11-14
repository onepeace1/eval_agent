"""End-to-end check of a running SkyRL Tinker server, through scottyrl's own code.

Exercises every server call a training run makes, without needing any of the
student TODOs:

  1. create a LoRA training client (loads Qwen3.5-4B on the trainer)
  2. save a sampler snapshot and sample from it (vLLM, with logprobs)
  3. one PPO update on the sampled tokens (forward_backward + optim_step)
  4. save a second snapshot and sample from the updated weights

    TINKER_BASE_URL=http://localhost:9000 uv run python skyrl_tinker_docker/smoke_client.py
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import uuid

import tinker

from scottyrl.tinker_adapter import importance_weight_stats, make_service_client, to_tinker_datum
from scottyrl.types import TokenSequence

BASE_MODEL = os.environ.get("SKYRL_BASE_MODEL", "Qwen/Qwen3.5-4B")


def step(label: str, started: float) -> None:
    print(f"[ok] {label} ({time.monotonic() - started:.1f}s)", flush=True)


async def sample(sampling_client, prompt_tokens: list[int], n: int):
    result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=n,
        sampling_params=tinker.SamplingParams(max_tokens=32, temperature=1.0, top_p=1.0, top_k=-1),
    )
    for seq in result.sequences:
        assert len(seq.tokens) > 0, "empty completion"
        assert seq.logprobs is not None and len(seq.logprobs) == len(seq.tokens), "missing logprobs"
        assert all(math.isfinite(lp) for lp in seq.logprobs), "non-finite logprob"
    return result.sequences


async def main() -> None:
    assert os.environ.get("TINKER_BASE_URL"), "set TINKER_BASE_URL"
    tag = uuid.uuid4().hex[:6]
    t = time.monotonic()
    service = make_service_client()
    training = await service.create_lora_training_client_async(base_model=BASE_MODEL, rank=32)
    tokenizer = training.get_tokenizer()
    step("created LoRA training client (rank 32)", t)

    t = time.monotonic()
    saved = await (await training.save_weights_for_sampler_async(name=f"smoke_{tag}_v0")).result_async()
    sampler = service.create_sampling_client(model_path=saved.path)
    step(f"saved sampler snapshot {saved.path}", t)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Guess a number between 1 and 100. Reply with just the number."}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    prompt = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
    t = time.monotonic()
    seqs = await sample(sampler, prompt, n=4)
    step("sampled 4 completions with logprobs", t)
    for s in seqs:
        print("   ->", repr(tokenizer.decode(s.tokens, skip_special_tokens=True)))

    # One GRPO-shaped batch: prompt tokens masked out, +/-1 advantages on the
    # completions, normalised by the action-token count as TinkerSession does.
    n_prompt = len(prompt)
    batch = [
        TokenSequence(
            tokens=prompt + list(s.tokens),
            mask=[0] * (n_prompt - 1) + [1] * len(s.tokens),
            logprobs=[0.0] * (n_prompt - 1) + [float(x) for x in s.logprobs],
            advantages=[0.0] * (n_prompt - 1) + [1.0 if i % 2 == 0 else -1.0] * len(s.tokens),
        )
        for i, s in enumerate(seqs)
    ]
    scale = 1.0 / sum(seq.num_action_tokens for seq in batch)
    datums = [to_tinker_datum(seq, "ppo", advantage_scale=scale) for seq in batch]
    t = time.monotonic()
    fb_future = await training.forward_backward_async(datums, loss_fn="ppo")
    opt_future = await training.optim_step_async(tinker.AdamParams(learning_rate=1e-5))
    fb = await fb_future.result_async()
    opt = await opt_future.result_async()
    step("PPO forward_backward + optim_step", t)
    print("   forward_backward metrics:", fb.metrics)
    print("   optim_step metrics:", getattr(opt, "metrics", None))
    stats = importance_weight_stats(batch, [out["logprobs"].tolist() for out in fb.loss_fn_outputs])
    print("   trainer vs sampler importance weights:", stats)
    assert 0.5 < stats["importance_weight_mean"] < 2.0, "trainer and sampler logprobs disagree badly"

    t = time.monotonic()
    saved = await (await training.save_weights_for_sampler_async(name=f"smoke_{tag}_v1")).result_async()
    sampler = service.create_sampling_client(model_path=saved.path)
    await sample(sampler, prompt, n=2)
    step(f"saved updated snapshot {saved.path} and sampled from it", t)
    print("SMOKE CLIENT OK")


if __name__ == "__main__":
    asyncio.run(main())
