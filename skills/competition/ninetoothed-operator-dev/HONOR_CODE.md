# Honor Code

Submission: `ninetoothed-operator-dev` skill for the 2026 Spring AI Competition,
track T3-1-1 (九齿 .skill 创新挑战).

I affirm the following about this submission:

1. **Original work.** This skill — its `SKILL.md`, references, scripts, examples,
   and self-test material — is my own work. I have not plagiarised, and I am not
   passing off anyone else's work as mine.

2. **No bundled secrets or unauthorized data.** The package contains no API keys,
   credentials, tokens, private accounts, or unauthorized/proprietary data.

3. **No hidden evaluation answers, no reward hacking.** The package does not
   contain the organizers' hidden evaluation answers, hard-coded task names, or
   logic that targets or circumvents the evaluation scripts. The files under
   `evaluation/proxy_tasks/` are **my own** offline self-test tasks (with
   reference solutions I wrote) used only to score my own A/B runs; they are
   **never surfaced to the agent during a task** and are not the organizers'
   hidden tasks.

4. **No test tampering or bypassing.** The skill does not instruct the agent to
   delete or weaken tests, fabricate results, bypass evaluation, or perform
   high-risk operations. Correctness is checked against PyTorch references; a
   declared PyTorch fallback (for cases outside the DSL) is always marked, never
   silent.

5. **Offline reproducibility.** The skill runs offline in the organizers'
   environment with no mandatory network dependency. Self-tests were verified on
   an RTX 5090 (ninetoothed 0.25.0, triton 3.4.0, torch 2.8.0+cu128).

6. **Full disclosure.** All external sources, referenced methods, third-party
   dependencies, and the scope of generative-AI assistance are disclosed in
   `REFERENCE.md`. No third-party code was copied verbatim; example-kernel
   patterns are adapted from the NineToothed repository and cited.

Signed: 楚凌志 (GitHub: noCharger)
Date: 2026-07-12
