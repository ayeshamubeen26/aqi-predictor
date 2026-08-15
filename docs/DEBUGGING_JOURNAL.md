# Debugging Journal: Keeping the Pipeline Honest

This documents four real production issues found and fixed in this project's
automated pipeline, in the order they were actually diagnosed. Each one
initially looked like something else, and each required ruling out the
obvious explanation before finding the real one. This is written up
separately from the README because it's a different kind of record: not
what the system does, but how its actual failures were found and closed.

## Issue 1: A green checkmark that meant nothing

**Symptom.** The hourly feature pipeline showed "Success" on every run in
GitHub Actions. No red flags anywhere. But the AQI dashboard's live data
wasn't changing hour to hour the way it should have.

**What looked like the obvious explanation.** Nothing, at first, since
the workflow was green. The bug was invisible until the logs were read
line by line instead of trusting the pass/fail badge.

**What the logs actually showed.** Every single OpenWeather API call, for
every city, was failing with `401 Client Error: Unauthorized`. The script
caught that error per city, printed a warning, and moved on to the next
city. When every city failed the same way, the script reached the end of
its loop with zero rows to insert, and exited normally, no exception, no
non-zero exit code. GitHub Actions has no way to know a script that
technically completed without crashing actually did nothing useful.

**Root cause.** Two layered problems: an invalid/misconfigured API key
upstream, and a script that had no way to distinguish "ran successfully"
from "ran successfully and accomplished nothing."

**The fix.** `data_fetch.py` now tracks whether any rows were actually
fetched, and calls `sys.exit(1)` if not, turning a silent no-op into a
visibly failed run. This is a general principle worth naming: a pipeline
that can succeed while doing nothing is more dangerous than one that
fails loudly, because nobody goes looking for a problem that appears to
already be working.

## Issue 2: The table nobody was writing to

**Symptom.** Even once real data was flowing in hourly, the dashboard's
forecasts and the "24-Hour AQI Trend" chart weren't reflecting it.

**What looked like the obvious explanation.** That the hourly job simply
wasn't running, or wasn't running often enough.

**What tracing the code actually showed.** The hourly job wrote to a
feature group called `aqi_features`. The models, and the live dashboard,
read from a *different* feature group, `aqi_features_final`, which
includes computed rolling averages, lag features, and forecast targets.
The step that built `aqi_features_final` from the raw table
(`add_rolling_features.py` → `add_targets.py`) was designed as a one-time
batch job for the initial backfill. It was never scheduled to run again.
So every hour, new data landed in `aqi_features`, and just sat there,
disconnected from the table everything downstream actually depended on.

**Root cause.** Two feature groups existed, only one was being kept
current, and the architecture never connected them after the initial
backfill.

**The fix.** A new script, `sync_final_features.py`, reuses the existing
feature-engineering functions (not a rewrite) on a bounded 10-day rolling
window rather than the full history, keeping the hourly cost small, and
runs as a second step in the same hourly workflow. Rows from a few days
ago also get their forecast targets filled in retroactively as real time
catches up to them, since a target like "AQI 72 hours later" literally
can't be known until 72 hours have actually passed.

## Issue 3: The write that succeeded, then reported failing

**Symptom.** Once the sync step existed, it started failing consistently,
roughly 8-10 minutes into every run, red X, no useful error message in
the GitHub Actions log beyond "The Hopsworks Job failed, use the
Hopsworks UI to access the job logs."

**What looked like the obvious explanation.** That the sync script's
logic was wrong, or that recently-inserted data wasn't visible yet due to
a timing/consistency delay (this turned out to be a real, separate,
earlier issue, fixed by making the insert wait for its materialization
job to actually finish before the next step tried to read it).

**What the actual Hopsworks execution logs showed**, after digging past
the outer status and into the underlying Spark job logs: the real data
write (`DeltaStreamer sync completed successfully`) finished cleanly,
every single time, in the first minute or two. The failure always came
*after* that, during an automatic statistics-computation step Hopsworks
runs post-write, which was timing out against an internal metrics
endpoint (`SocketTimeoutException` on a Prometheus `PushGateway` call,
repeating every 20 seconds until the whole job gave up).

**Root cause.** An optional post-write bookkeeping step, unrelated to
whether the actual data landed, was failing due to what looks like a
free-tier infrastructure limitation, and failing the entire job over it.

**The fix, in two layers.** First, disabling `statistics_config` on both
feature groups, since this project never reads those auto-computed
statistics anyway, there's no reason to pay for (or risk failing on) a
step nothing depends on. Second, as a safety net: since the disable
setting only applies going forward and doesn't retroactively help every
possible cluster state, the insert call is also wrapped to catch this
specific failure and treat it as non-fatal, since three separate incident
logs all confirmed the data write itself had already succeeded by the
time this exception fires. The pipeline shouldn't fail over a step whose
only job is producing numbers this project never looks at.

## Issue 4: Five hours in the wrong direction

**Symptom.** The dashboard's "Updated" timestamp was consistently hours
behind the actual current time, by exactly 5 hours.

**What looked like the obvious explanation.** Stale data, or the page
simply not having been refreshed recently.

**What actually explained it.** Pakistan is UTC+5. Streamlit Cloud's
servers run on UTC. `datetime.now()` was returning the server's own
local time, which is UTC, with no conversion to the timezone of the
person actually looking at the page.

**Root cause.** Code that assumed "the server's local time" and "the
user's local time" are the same thing, which is only true by accident,
never by design, for any app not hosted in the same timezone as its
audience.

**The fix.** An explicit `Asia/Karachi` conversion using Python's
built-in `zoneinfo`, rather than depending on whatever timezone happens
to be configured on whichever server the app is deployed to next.

## The pattern across all four

Every one of these bugs was invisible from the outside. Green checkmarks,
clean-looking dashboards, no crashes. Finding them all required the same
approach: don't trust the summary status, read the actual log, and follow
the data itself rather than assuming the code that's supposed to move it
is doing so correctly. That's the difference between a pipeline that
looks like it works and one that's actually been verified to.