# Scanning Guide

This guide is about **how** to run ham2mon for what you're trying to do: what you
point the scanner at, whether you filter channels with banks, and what you will
see and record as a result. If instead you want to know **how to write** a
frequency configuration file — every option, every key — see the
[full example frequencies file](./full-example.freqs.yaml), which builds up from
the bare minimum through every configuration option.

The basic idea: ham2mon sweeps a band and can demodulate several channels at
once. A frequency configuration file (`-F`) turns hits into labeled, prioritized
channels, and optional **bank tags** filter which of those channels are actually
heard. You do not have to use either.

The command-line examples below assume an Airspy at a 3 Msps sample rate
(`-a "airspy" -r 3E6`); adapt the `-a` device argument and the `-r` sample rate
to your SDR. Most examples use a single center frequency; a range wider than the
sample rate is only used where sweeping a band is the point of the use case.

## How channel selection works

Every carrier hit is resolved to a list of **bank tags** using a 5-tier
precedence (`FrequencyManager.resolve_banks()`, frequency_manager.py):

1. Explicit single-entry tone match (tier 1)
2. Explicit single entry's base bank (tier 2)
3. Range entry tone match (tier 3)
4. Range entry's base bank (tier 4)
5. Dynamic fallback (tier 5): `SEARCH` if selected, otherwise `UNTAGGED` when
   bank filtering is active, otherwise no tags

When you select banks with `--banks` (or `frequency_policies.active_banks` in
YAML), filtering is **fail-closed**: a channel whose resolved tags do not
intersect the selected set is never demodulated, and a transmission already
running on such a channel has its recording discarded. Without `--banks`, every
channel is monitored (this is called **promiscuous** mode) and bank tags are not
resolved or displayed at all — even if the frequency file declares them.

## Use cases

### 1. No frequency file, no `--banks`

    uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 462.5

**Use this when you just want to scan over a frequency range of interest and
have no idea what may be there.**

What you'll see: every channel in the band is swept and demodulated when active.
The RECEIVER panel's **Banks** row reads `none`, and no `[tag]` blocks appear in
the CHANNELS panel. Recordings and logs carry no bank tag.

### 2. No frequency file, with `--banks SEARCH`

    uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 462.5 --banks SEARCH

**Use this when you have no channel list at all and want every hit explicitly
flagged as an unconfirmed find.**

What you'll see: every channel is still monitored, but each one now displays
`[SEARCH]` in the CHANNELS panel and the **Banks** row reads `SEARCH`.
Recordings and log entries carry the bank `SEARCH`, so you can tell from the
output that a hit was captured outside any configured entry.

### 3. No frequency file, with `--banks UNTAGGED`

    uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 462.5 --banks UNTAGGED

**Use this when you have no channel list at all and prefer hits to be tagged
`untagged` rather than `search`.**

What you'll see: functionally identical to case 2 — every channel monitored and
displayed with `[UNTAGGED]` — differing only in the tag that appears in the
panel and in the bank metadata on recordings and logs.

With no channel list this choice is pure preference — both modes monitor
everything, differing only in the tag. It gains functional weight once you add a
channel list (`-F`): `UNTAGGED` also keeps configured entries that carry no bank
tag (tiers 2 and 4), while `SEARCH` covers only unconfigured hits (tier 5). See
the `--banks NET_A SEARCH` vs `--banks NET_A UNTAGGED` contrast in case 7.

### 4. No frequency file, with `--banks <real bank>` — a pitfall

    uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 462.5 --banks NET_A

**Don't do this: with no channel list, nothing will ever be heard.** Every hit
resolves to `UNTAGGED` (tier 5), which never matches a real bank tag like
`NET_A`, so filtering fails closed on every channel. ham2mon logs a startup
warning: "Active bank(s) ... match no configured frequency/tone banks". This is
the fail-closed safety net working as intended — check the `-F` file and the
`--banks` spelling.

### 5. Frequency file, no banks defined, no `--banks`

    uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 451 -F doc/full-example.freqs.yaml

**Use this when you have a known list of channels and ranges to monitor but
don't need to categorize them.**

What you'll see: all configured channels are monitored (promiscuous mode). The
**Banks** row reads `none` and no `[tag]` blocks appear in the CHANNELS panel —
even for entries whose YAML declares `banks:` — because the panel only renders
bank tags while bank filtering is active. Persisted bank tags are still resolved
unconditionally: a configured entry's tags land in the fixed-field activity log,
json-server / webhook payloads, and sidecar JSON metadata even without `--banks`.
Labels, priorities, and lockouts still work normally. The `-f 451` center places
the full-example channels (`450.100`–`451.800` MHz) inside the 3 MHz window;
point `-f` at your own channel band the same way.

### 6. Frequency file with banks, plus `--banks <set>`

    uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 451 -F doc/full-example.freqs.yaml --banks DISPATCH FIELD

**Use this when you know what you care about and want everything else ignored.**

What you'll see: only channels whose resolved tags intersect `DISPATCH` or
`FIELD` are demodulated and recorded — e.g. `[DISPATCH,FIELD]` on "Command
coordination" and `[FIELD]` on "Wide-area field operations" — and the **Banks**
row reads `DISPATCH, FIELD`. Everything else is silent: "Logistics net" carries
only `[LOG]`, and hits outside all configured entries resolve to `UNTAGGED`,
which is not selected. See the
[Bank Filtering section of the README](../README.md#bank-filtering---banks) for
the syntax of bank tags on singles, ranges, and per-tone rules.

### 7. Custom combinations

**Use this when one policy isn't enough:** you want your priority banks *plus*
unknown or untagged traffic, or a default bank for a geographic range with
specific known channels overriding it.

- **Priority banks + unknown spectrum** — `uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 460.175 -F my-freqs.yaml --banks NET_A SEARCH` (where `my-freqs.yaml` is your file with NET_A-banked channels): NET_A-banked channels plus any unconfigured hit (tagged `SEARCH`).
- **Priority banks + everything unbanked** — `uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 460.175 -F my-freqs.yaml --banks NET_A UNTAGGED`: NET_A-banked channels, plus unconfigured hits *and* configured entries that carry no bank tag (both resolve to `UNTAGGED`). Note `UNTAGGED` is broader than `SEARCH`: it also keeps legacy configured entries that have no `banks:`.
- **Geographic default with per-channel overrides** — a range entry provides the
  default bank, and specific single entries override it. Single entries always
  take precedence over a covering range (tiers 1–2 run before tiers 3–4),
  regardless of file order:

      frequencies:
        - label: "Race weekend range"
          lo: 460.0
          hi: 468.0
          banks: [REGIONAL]
        - label: "Corner Workers"
          single: 466.375
          banks: [TRACK]

  Run with, e.g., `uv run apps/ham2mon.py -a "airspy" -r 3E6 -f 460.0-468.0 -F race-freqs.yaml --banks REGIONAL TRACK` (where `race-freqs.yaml` contains the snippet above) so both tags are active. This is where a range wider than the sample rate is used on purpose: the `-f` range is swept as a series of ~3 MHz windows, stepping to the next one when the current window is quiet. A hit at `466.375` resolves to `[TRACK]`; any other hit in the range resolves to `[REGIONAL]`. If you omit one of the banks from `--banks`, every channel tagged with it goes silent (fail-closed), so list all the banks you want to hear.

## Quick reference

| Configuration | Banks row | CHANNELS tags | What is heard |
|---|---|---|---|
| No `-F`, no `--banks` | `none` | none | everything in the scan band |
| No `-F`, `--banks SEARCH` | `SEARCH` | `[SEARCH]` | everything, tagged SEARCH |
| No `-F`, `--banks UNTAGGED` | `UNTAGGED` | `[UNTAGGED]` | everything, tagged UNTAGGED |
| No `-F`, `--banks <real>` | `<real>` | nothing (all `UNTAGGED`) | nothing + startup warning |
| `-F`, no `--banks` | `none` | none | all configured channels |
| `-F`, `--banks <set>` | `<set>` | `[<set>]` on matches | only matching channels |
| custom `--banks` combos | combo | per-channel tags | per combination |

## Changing banks at runtime

Press `b` in the TUI to edit the active banks in place (Enter applies, Esc
cancels). The new selection takes effect on the next scan cycle; transmissions
already running on a deselected bank finish naturally but their recordings are
discarded. Runtime changes are not persisted across launches. See the
[Bank Filtering section of the README](../README.md#bank-filtering---banks).

## Auditing bank membership with `--list-banks`

To see what each bank actually selects before (or without) running the scanner,
load the frequency file in audit mode:

    uv run apps/ham2mon.py -F my-freqs.yaml --list-banks

This prints each configured bank with its channel members — using the
top-level display label when one is configured, and showing a member's
per-bank label in parentheses when it has one — then exits. No SDR or scan is
started. It is a read-only preview of the frequency file.

## Where the bank tag ends up

- **CHANNELS panel** — `[tag,...]` block before the CTCSS readout, dimmed while
  idle. Only rendered while bank filtering is active.
- **fixed-field activity log** — bank tags joined with commas into a 15-column
  field; multi-tag values are truncated to 15 characters (the tail of the last
  tag is dropped).
- **json-server / webhook** — the full bank tag list is sent unmodified.
- **sidecar JSON metadata** — the full resolved bank list for kept recordings.
