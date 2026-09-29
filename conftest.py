# Exclude the competition skill packages from the repository's own pytest run.
# Each skill's self-tests are exercised via its `run_self_tests.sh` instead: they
# require CUDA and use per-example test modules with intentionally repeated
# basenames, which would otherwise collide during repository-wide collection.
collect_ignore = ["skills"]
