# Pearls AQI Predictor — Project Report

*A serverless, multi-city machine learning system for forecasting Air Quality Index 24, 48, and 72 hours ahead across five Pakistani cities.*

Prepared by Ayesha Mubeen

This report documents the full system as built: the data pipeline, the
modeling approach and results, the automation infrastructure, the live
dashboard, and, deliberately, the real production issues encountered
along the way and how each one was diagnosed and resolved. That last
part is included on purpose: building something that works once is
different from understanding why it works, and being able to explain
what breaks and why is the more durable skill.

A short note before diving in: this project did not go the way a clean
tutorial does. Between managing this alongside client work and a full
course load, and hitting real production issues that had nothing to do
with anything a course could have prepared me for, this took longer and
more patience than I expected going in. I am including that honestly
here rather than presenting a polished result as if it arrived that
way, because the debugging process is genuinely the part I am most
proud of, not the parts that worked on the first try.

---

## 1. Project Overview

The goal of this project was to build an end-to-end AQI forecasting
system using a 100% serverless stack: automated hourly data collection,
feature engineering, model training and comparison, and a live
dashboard, covering five Pakistani cities rather than one, and
forecasting three separate horizons (24h, 48h, 72h) rather than a
single point estimate.

**Cities covered:** Karachi, Lahore, Islamabad, Faisalabad, Peshawar.

**What is actually being predicted.** Each horizon predicts the AQI at
one specific future hour, not a daily average. For a row timestamped at
14:00, `target_aqi_24h` is the actual AQI value at 14:00 the following
day, built by shifting the hourly AQI series forward by exactly 24, 48,
and 72 rows. A point forecast like this is a harder, noisier target
than a daily average would be, since AQI genuinely swings within a
single day, but it is the more useful and more honest version of the
problem to solve.

---

## 2. Architecture

The system is split into two independently scheduled pipelines feeding
a shared feature store, plus a prediction layer the dashboard reads
from live.

| Stage | What it does | Runs |
|---|---|---|
| Feature pipeline | Fetches live weather + pollutant data per city from OpenWeather, computes engineered features, writes to the raw feature group | Scheduled, GitHub Actions |
| Feature sync | Promotes recent raw rows into the final, model-ready feature group (rolling/lag features, AQI, forecast targets) | Same schedule, second step |
| Training pipeline | Trains and compares three model types per horizon, registers the winner | Scheduled, weekly |
| Prediction + dashboard | Loads the live registered model, computes a fresh forecast on request | On demand, Streamlit |

**Why two separate feature groups.** The raw feature group
(`aqi_features`) holds exactly what was fetched, one row per city per
hour. The final feature group (`aqi_features_final`) holds the
model-ready version: rolling averages, lag features, computed AQI, and
the three forecast targets. Keeping these separate matters because
targets like `target_aqi_72h` cannot be known until 72 real hours have
actually passed, so the promotion step has to re-run periodically over
a recent window, not just once, to let older rows catch up as their
future values become knowable.

---

## 3. Technology Stack

| Layer | Tool |
|---|---|
| Data source | OpenWeather (current weather + air pollution APIs) |
| Feature store / model registry | Hopsworks (free tier) |
| Modeling | Scikit-learn (Ridge, Random Forest), TensorFlow/Keras (dense neural network) |
| Automation | GitHub Actions (scheduled workflows) |
| Dashboard | Streamlit, deployed on Streamlit Community Cloud |
| Explainability | SHAP (model-type aware: TreeExplainer, LinearExplainer, KernelExplainer) |

---

## 4. Feature Engineering

Each hourly row includes:

- **Cyclical time features**: hour, day-of-week, and wind direction
  encoded as sin/cos pairs rather than raw integers, so hour 23 and
  hour 0 are treated as adjacent instead of maximally distant.
- **Lag features**: PM2.5 at 1h, 3h, and 24h prior.
- **Rolling features**: PM2.5 averaged over 3h, 6h, and 24h windows.
- **Raw pollutants and weather**: PM2.5, PM10, CO, NO2, O3, SO2,
  temperature, humidity, wind speed.
- **Computed AQI**: derived from PM2.5/PM10 using the EPA breakpoint
  formula (May 2024 breakpoints).

**Why AQI is computed, not fetched directly.** OpenWeather returns raw
pollutant concentrations, not an AQI value. AQI itself is a piecewise,
non-linear transform of those concentrations against EPA breakpoint
tables. This conversion happens once, when building historical
targets, not as a second step after prediction: the models are trained
to output AQI directly, using pollutant readings as input features
alongside everything else, not by predicting PM2.5 and then converting
it. Predicting the target directly avoids compounding error through a
second non-linear transform on top of the model's own prediction
error.

---

## 5. Modeling and Results

For each of the three horizons, three model types are trained and
compared against a persistence baseline (predict future AQI = current
AQI), evaluated on a time-based 80/20 split per city, not a random
shuffle, since a random split would leak future information into
training for a time series problem.

**Why the baseline matters.** Exploratory analysis showed current AQI
correlates with AQI 72 hours later at r = 0.70, meaning the naive
"assume no change" guess is a genuinely strong baseline in this domain,
not a strawman. Beating it convincingly is real evidence of learned
signal, not just noise.

| Horizon | Baseline RMSE | Winning model | RMSE | MAE | R² |
|---|---|---|---|---|---|
| 24h | 28.45 | Random Forest | 24.90 | 17.31 | 0.555 |
| 48h | 33.09 | Random Forest | 28.34 | 21.28 | 0.417 |
| 72h | 35.60 | Random Forest | 30.79 | 23.66 | 0.310 |

*Results from a representative training run. Exact figures shift
slightly across retraining cycles as the underlying window of data
moves forward.*

**Reading the results honestly:**

- Random Forest beat the baseline at every horizon, the clearest
  evidence the models learned real structure rather than mimicking
  persistence.
- Error grows with horizon for every model, including the baseline,
  which is expected: air quality further out is inherently less
  predictable from current conditions alone.
- Ridge underperformed the baseline at 72h, air quality's relationship
  to the input features is non-linear enough that a purely linear model
  struggles at longer horizons.
- The neural network never won a single horizon in any run so far.
  Tree-based models fit this feature set better at this data volume,
  worth revisiting with more data or a different architecture rather
  than treated as settled.

**SHAP explainability findings.** Feature importance shifts
meaningfully with horizon. At 24h, the current PM2.5 reading dominates
the prediction. At 72h, the 24-hour rolling average of PM2.5 becomes
co-dominant with the current reading, and single-moment weather
features like wind speed matter less, the model leans more on smoothed
recent history as the forecast window stretches out. This is a
genuinely sensible pattern for a model to have learned, not an
artifact.

---

## 6. Automation and Deployment

- **Feature pipeline**: scheduled via GitHub Actions cron, fetches
  live data and syncs the model-ready feature group.
- **Training pipeline**: scheduled weekly (moved from daily after
  reviewing free-tier compute cost against a slow-moving signal),
  compares all model types fresh and registers whichever wins.
- **Rolling training window**: bounded to the most recent 365 days
  rather than the full history, keeping training time and cost flat
  as the feature store keeps growing, instead of scaling with total
  history size forever.
- **Dashboard**: deployed on Streamlit Community Cloud, reads the live
  registered model and live feature data on every request, not a
  cached snapshot.

---

## 7. Dashboard Features

- Live AQI gauge and current pollutant breakdown per city
- 24-hour AQI trend chart from real stored history
- 3-day forecast with per-horizon confidence context (live model RMSE
  shown alongside each prediction)
- SHAP-based "why this prediction" panel, one explanation per horizon
- Forecast accuracy backtest: the 24h model's past predictions plotted
  against what AQI actually turned out to be, real evidence of
  tracking, not just a claimed RMSE number
- National overview comparing all five cities at once
- Safety precautions that scale to the worst predicted condition
  across the 3-day window, not just the current moment, and name the
  specific pollutant driving concern in that city

---

## 8. Production Issues Diagnosed and Fixed

The system did not work correctly the first time it was deployed. What
follows is a record of five real issues found after the pipeline was
already live, what each looked like on the surface, what the actual
cause turned out to be, and why the fix works, not just that it does.

### 8.1 — A pipeline that succeeded while doing nothing

**Symptom.** GitHub Actions showed "Success" on every hourly run, but
live data on the dashboard was not actually changing.

**Root cause.** Every OpenWeather API call was failing with 401
Unauthorized. The script caught each city's failure individually,
logged a warning, and moved to the next city. When every city failed,
the script still reached the end of its loop and exited normally, no
exception, no non-zero exit code, so GitHub Actions had no way to tell
"ran successfully" from "ran successfully and accomplished nothing."

**Fix.** The script now tracks whether any rows were actually fetched
and calls `sys.exit(1)` if not, converting a silent no-op into a
visibly failed run. A pipeline that can succeed while doing nothing is
more dangerous than one that fails loudly, since nobody goes looking
for a problem that appears to already be working.

### 8.2 — The feature table nobody was writing to

**Symptom.** Even with real data flowing in hourly, forecasts and the
trend chart were not reflecting it.

**Root cause.** The hourly job wrote to the raw feature group
(`aqi_features`). Models and the dashboard read from a separate,
model-ready feature group (`aqi_features_final`), built by a batch
process originally designed to run once, during the initial backfill.
That promotion step was never scheduled to run again, so new hourly
data landed in the raw table and sat there, disconnected from
everything downstream.

**Fix.** A new script promotes recent rows from the raw table into the
final table on the same schedule as collection, reusing the existing
feature-engineering functions rather than duplicating them, over a
bounded recent window rather than the full history, so it stays cheap
to run repeatedly.

### 8.3 — A write that succeeded, then reported failing

**Symptom.** The sync step began failing consistently, roughly 8-10
minutes into every run, with no specific error beyond "the Hopsworks
job failed."

**Root cause.** Reading the actual Hopsworks execution logs (not just
the outer status) showed the real data write completing successfully
every time, in the first minute or two. The failure consistently came
afterward, during an automatic statistics-computation step Hopsworks
runs post-write, timing out against an internal metrics endpoint on
the free-tier cluster.

**Fix.** Statistics computation was disabled for both feature groups,
since this project never reads those auto-computed statistics. As a
safety net, the write is also wrapped to catch this specific failure
and treat it as non-fatal, since the actual data write had already
completed by the time it fires in every observed case.

### 8.4 — Five hours in the wrong direction

**Symptom.** The dashboard's "Updated" timestamp was consistently five
hours behind the actual current time.

**Root cause.** Pakistan is UTC+5. Streamlit Cloud's servers run on
UTC. `datetime.now()` was returning the server's own local time with
no conversion to the timezone of the person actually viewing the page,
an assumption that only happens to be true if an app is hosted in the
same timezone as its audience, which is never guaranteed.

**Fix.** An explicit `Asia/Karachi` conversion using Python's built-in
`zoneinfo` module, rather than depending on whatever timezone the
deployment server happens to run on.

### 8.5 — An intermittent read failure traced to a genuine resource limit

**Symptom.** The feature sync step began failing intermittently with a
dropped-connection error from Hopsworks' Arrow Flight query service,
then, later, began failing on nearly every run.

**Root cause.** The intermittent phase pointed to transient
infrastructure flakiness and was addressed with retry logic. The shift
to near-constant failure was traced to a genuinely different cause: a
Hopsworks free-tier spending alert showing compute throttled once
monthly spend passed budget. Failing jobs in that state die in roughly
20 seconds, a different, faster signature than the earlier timeout
issue, which is what distinguished a resource cap from a
retry-fixable connection blip.

**Fix.** Retry-with-backoff was added for the genuinely transient
case. For the resource-cap case, no code change fixes an exhausted
budget, scheduled runs were paused entirely until the budget resets,
since every additional scheduled attempt in a throttled state spends
further against an already-exhausted allowance for zero benefit.

---

## 9. Known Limitations

- No automated test coverage beyond a placeholder file, a real gap,
  named here rather than hidden.
- The neural network has never won a horizon in any training run so
  far; worth revisiting with more data before concluding tree-based
  models are simply the better fit.
- GitHub Actions' free-tier scheduler does not guarantee exact cron
  timing; observed gaps between hourly runs have ranged up to several
  hours under platform load.
- Hazard alerts are shown on the dashboard but are not pushed anywhere
  (email, SMS); they are only visible to someone actively viewing the
  app.
- The project depends on Hopsworks' free tier, which has a real,
  finite monthly compute budget; this is an active operational
  constraint, not a one-time setup cost.

---

## 10. What This Project Demonstrates

Beyond the forecasting system itself, the more transferable outcome of
this project is the debugging process behind it: reading actual logs
instead of trusting a pass/fail badge, distinguishing a code bug from
a genuine infrastructure limit, and knowing when a fix is a retry
versus when it requires pausing and waiting for a resource to actually
recover. None of the five issues documented here were visible from the
outside, every one looked like a clean, working system until the
underlying data or logs were actually read. That is the pattern worth
carrying forward past this specific project: a system that looks like
it works and one that has actually been verified to are not the same
thing, and the difference is only visible to whoever actually goes
looking.

---

## 11. Personal Reflection

I want to be honest about what this project actually felt like to
build, because I think that matters as much as the final result.

There were points where I genuinely thought the pipeline was done,
where a green checkmark on GitHub Actions felt like the finish line,
only to find out later that green did not mean what I assumed it meant.
The silent 401 failure was the one that humbled me the most early on,
I had looked at that dashboard and trusted it. Learning to stop
trusting a status badge and actually read the log underneath it was
not a lesson I expected to take away from a machine learning project,
but it is probably the one I will carry the longest.

The Hopsworks compute throttle was a different kind of frustrating,
because there was no code fix waiting for me at the end of it. I am
used to problems that resolve once you find the right line to change.
Sitting with a problem that just required patience, and being honest
enough to say "I do not know exactly when this resets" instead of
guessing something reassuring, was its own kind of discipline.

I came into this balancing an agency job, freelance client work, and a
CS degree, and there were evenings this project competed directly with
all of that for my attention. I do not think the finished dashboard
fully shows how much of the actual work was reading a stack trace at
midnight and trying to figure out whether it was my mistake or
something outside my control. I am glad it is documented here instead
of just disappearing into the process, because that part of the work
was real too.
