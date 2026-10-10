# Tests

```bash
pytest skills/competition/ninetoothed-op-dev-skill/tests -q
```

| File | Purpose |
|------|---------|
| `test_skill_structure.py` | SKILL frontmatter, references, scripts, forbidden runtime paths |
| `test_scripts_smoke.py` | `--help` and score_task write |
| `test_pack_whitelist.py` | Packer whitelist (source tree only) |
| `test_pack_post_audit.py` | Post-pack audit of packed output |
| `test_eval_gate.py` | make_task_card / gate_eval / score_task |

No GPU required for this suite.
