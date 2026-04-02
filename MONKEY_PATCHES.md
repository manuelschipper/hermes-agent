# Monkey Patches

This file tracks local patches carried in this checkout that are not part of upstream Hermes Agent.

## Cron Origin Fallback

- File: `cron/scheduler.py`
- Status: active
- Rationale: some cron jobs use `deliver: origin` but can lose their own `origin` metadata. When that happens, delivery would fail even though sibling jobs in the same `jobs.json` still have a valid destination.
- Local behavior: if a job with `deliver: origin` has no origin, Hermes falls back to the most recently used valid `origin` from another job in the same cron file.
- Impact: preserves cron auto-delivery for existing chat destinations instead of dropping the response.
