# How to verify this skill

## Method 1 — Install and run on ninetoothed-examples

```bash
git clone https://github.com/InfiniTensor/ninetoothed-examples
cd ninetoothed-examples
pip install ninetoothed torch triton
pytest tests/ -v
```

All tests should pass. This confirms the reference environment works.

## Method 2 — Agent task test (manual)

Give the agent this task WITHOUT showing the skill:
> "Implement a NineToothed kernel for `relu` (output = max(input, 0)).
>  Write a correctness test comparing against torch.relu."

Then repeat WITH the skill installed. Compare:
- Does the agent correctly use `arrangement` + `application` pattern?
- Does the agent include a non-contiguous test case and state whether it uses
  native stride handling or an explicit contiguous-copy fallback?
- Does the agent add `# noqa: F841` on the output assignment?
- Does the agent use `ntl.*` instead of raw Python math?

## Method 3 — Failure injection test

Deliberately introduce a bug (e.g., use `other=0.0` instead of
`other=float("-inf")` in a softmax Tensor declaration) and check whether
the agent following this skill diagnoses the root cause correctly per
Step 8 (Diagnose failures).

## Expected quality bar

- AI agent correctly implements any elementwise or row-wise reduction
  operator on first attempt (no retry needed)
- Correctness tests pass for fp32/fp16 where covered, and non-contiguous
  inputs are either natively supported or documented as fallback paths
- Agent does not write raw `tl.program_id` or manual offset arithmetic
