# Test log

No CI by decision — the suite is run manually via `uv run scripts/run_tests.py`,
which appends one row per run here and stores the full pytest output under
`results/test_runs/<run_id>/`. The `tree` column records the commit that was
tested; `+dirty` means uncommitted changes were present.

| when | tree | passed | failed | errors | skipped | duration | outcome | full output |
|---|---|---|---|---|---|---|---|---|
| 2026-09-01 08:44 UTC | ef17b90 | 10 | 0 | 0 | 0 | 2.0s | ok | results/test_runs/20260901-084419 |
| 2026-09-01 09:00 UTC | adbf9d7+dirty | 40 | 0 | 0 | 0 | 5.0s | ok | results/test_runs/20260901-090023 |
| 2026-09-01 10:39 UTC | 4bdc813+dirty | 40 | 0 | 0 | 0 | 4.3s | ok | results/test_runs/20260901-103945 |
| 2026-09-01 12:24 UTC | fe98bc6+dirty | 76 | 0 | 0 | 0 | 5.5s | ok | results/test_runs/20260901-122429 |
