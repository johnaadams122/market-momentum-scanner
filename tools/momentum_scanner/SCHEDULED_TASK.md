# Scanner Discovery Operations

## Scope

This document describes the tracked scanner discovery launcher,
`launch_scanner_discovery.ps1`: interpreter setup, the default launch command,
the optional Haiku shadow observer, and the heartbeat fields that confirm a
launch. It does not create or modify a Scheduled Task; wiring the launcher into
a scheduler is left to the operator.

## Interpreter Setup

Install the scanner dependencies, plus the `tzdata` time-zone package, into the
interpreter that will run discovery:

```powershell
# Required working directory: repository root.
$RepoRoot = (Resolve-Path -LiteralPath $PWD).Path
$PythonExe = "<path-to-python.exe>"
& $PythonExe -m pip install -r (Join-Path $RepoRoot "tools\momentum_scanner\requirements.txt")
& $PythonExe -m pip install "tzdata"
```

`tzdata` is required on Windows: the scanner uses Python's `zoneinfo` for its
America/New_York session clock, and Windows has no system IANA time-zone
database for `zoneinfo` to read. Installing
`tools\momentum_scanner\requirements-dev.txt` instead also covers it (it adds
`tzdata` on Windows together with the test runner).

The interpreter and state directory are explicit inputs. Credentials remain in
the user environment (or, for the Schwab bearer token, come from the
token-provider module you configure); this launcher never sets or prints them,
and none is placed on a command line. Error messages from failed requests are
reduced or masked as described in SECURITY.md, but this document does not
promise that no credential text could ever reach a log or state file.

## Default Discovery Launch

Run the generic wrapper with an explicit interpreter, existing state directory,
and supported source:

```powershell
# Required working directory: repository root.
$RepoRoot = (Resolve-Path -LiteralPath $PWD).Path
$PythonExe = "<path-to-python.exe>"
$StateDir = "<state-dir>"
$Launcher = Join-Path $RepoRoot "tools\momentum_scanner\launch_scanner_discovery.ps1"
& $Launcher -PythonExe $PythonExe -StateDir $StateDir -Source "alpaca"
```

The wrapper prepares `PYTHONPATH` before starting the scanner. It does not
create the state directory.

## Optional Haiku Shadow Observation

Haiku shadow observation stays off by default. When you start the scanner with
this launcher, add `-EnableHaikuShadow` to turn it on (a direct run of the
daemon module instead sets `SCANNER_JUDGE_SHADOW=1` and an absolute
`SCANNER_SHADOW_LOG_PATH` whose parent folder exists, as the README describes):

```powershell
# Required working directory: repository root.
$RepoRoot = (Resolve-Path -LiteralPath $PWD).Path
$PythonExe = "<path-to-python.exe>"
$StateDir = "<state-dir>"
$Launcher = Join-Path $RepoRoot "tools\momentum_scanner\launch_scanner_discovery.ps1"
& $Launcher -PythonExe $PythonExe -StateDir $StateDir -Source "alpaca" -EnableHaikuShadow
```

Requirements and data flow when the shadow is on:

- `ANTHROPIC_API_KEY` must be set in the environment of the process that runs
  the launcher. Without it the scanner logs that the shadow is disabled and runs
  the Gemma judge alone.
- Each comparison row is appended to the file named by
  `SCANNER_SHADOW_LOG_PATH`. The launcher sets it to `judge_shadow.jsonl` in the
  state directory, which must already exist; without `-EnableHaikuShadow` the
  launcher sets `SCANNER_JUDGE_SHADOW=0` and clears the path.
- Data leaves the machine: for each candidate the news judge evaluates, the
  news headline and summary text (up to 8,000 characters of summary; with
  Alpaca news the summary can be the full article body) are sent to Anthropic's
  API, and the Haiku verdict comes back from it. Do not enable the shadow if that text must stay local.
- The shadow is observe-only and never changes a verdict.

Gemma remains the deciding judge. Haiku runs only as nonblocking observation;
it cannot promote, veto, delay, or replace a Gemma verdict.

The resolver is constructed when the scanner daemon starts, and the wrapper
switch overrides any ambient setting. A running daemon therefore does not change
mode when an environment variable changes. To turn shadow observation off,
remove `-EnableHaikuShadow` from the launch command and restart the scanner
daemon.

## Validation Evidence

After a launch with shadow observation enabled, validate the heartbeat and the
shadow JSONL without exposing provider data. The heartbeat is
`discovery_heartbeat.json` and the startup log is `discovery_daemon.log`, both
in the daemon state directory passed as `-StateDir`. Required evidence is:

- one `judge shadow ACTIVE` startup log line;
- heartbeat `shadow_status == active` and `shadow_worker_alive == true`;
- `shadow_queue_depth`: stays small, well under the queue cap of 20; a depth
  that keeps growing means shadow calls are slower than the judge batch;
- `shadow_submitted` and `shadow_completed`: both rise as candidates are
  judged, and `shadow_completed` trails `shadow_submitted` by at most the
  queue depth plus the one call in flight;
- `shadow_dropped`: stays 0; if it rises, `shadow_dropped_daily_cap` shows how
  many drops came from the daily call cap (250 calls per Eastern-time day)
  rather than from a fault;
- `shadow_write_failures`: stays 0; each failure is a comparison row that was
  not written;
- `shadow_last_error_type`: empty (`null`); a value names the error that
  stopped the shadow worker.

Also confirm the first valid `record_type: comparison` row after a judged
candidate. It must contain populated `primary` and `shadow` verdict evidence,
plus `primary_ts`, `shadow_started_ts`, and `shadow_completed_ts` timing
evidence. The comparison row is observational output only; it never changes a
verdict.
