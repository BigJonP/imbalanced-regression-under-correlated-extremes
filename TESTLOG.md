# Test log

No CI by decision — the suite is run manually via `uv run scripts/run_tests.py`,
which appends one row per run here and stores the full pytest output under
`results/test_runs/<run_id>/`. The `tree` column records the commit that was
tested; `+dirty` means uncommitted changes were present.

| when | tree | passed | failed | errors | skipped | duration | outcome | full output |
|---|---|---|---|---|---|---|---|---|
