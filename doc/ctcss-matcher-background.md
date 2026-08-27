# CTCSS Matcher — Constant Review

**File under review:** `apps/demodulators/BaseTuner.py`
**Subject:** The constants used by the hybrid CTCSS matcher
**Purpose:** Document where each tuning lever comes from — whether it is an industry standard value, taken from a reference SDR implementation, derived from first principles, or pre-existing ham2mon code — so the change can be reviewed.

---

## 1. Context

ham2mon detects CTCSS (PL) tones to gate recording on analog FM channels. The
current implementation is a *hybrid matcher*:

- **Fast gate (pre-existing):** one GNU Radio `analog.ctcss_squelch_ff` block per
  configured tone. Each block is a Goertzel detector running over a fixed window and
  gates which chain's audio contributes to the output. These chains are unchanged and
  remain in charge of audio routing.
- **Match authority (new):** a short, full-band FFT measurement of the decoded audio
  decides whether a configured tone is actually present. A match is latched only when
  the measured tone falls within tolerance of a configured tone.

Why it exists: an earlier experiment replaced the chains with an FFT measurement that was band-limited to 55–105 Hz (breaking all standard codes above 105 Hz, e.g. the 141.3 Hz tone in `doc/full-example.freqs.yaml`) and used a 2 s window (slowing
match/mismatch decisions). The work was reverted to the committed baseline, then the industry-standard approach was layered on top without re-introducing those regressions.

All constants discussed here live at the top of `class BaseTuner` in
`apps/demodulators/BaseTuner.py`.

---

## 2. Constant table

| Constant | Value | Purpose | Source classification |
|---|---|---|---|
| `_CTCSS_BAND_LO_HZ` | `67.0` | Lower edge of the measured CTCSS band | **Standard** — EIA/RS-220 tone set spans 67.0–254.1 Hz |
| `_CTCSS_BAND_HI_HZ` | `254.1` | Upper edge of the measured CTCSS band | **Standard** — EIA/RS-220 tone set spans 67.0–254.1 Hz |
| `_CTCSS_LP_HZ` | `260.0` | Low-pass cutoff ahead of the measurement tap (voice-falsing guard) | **Principle from ZL2PD** — value derived (just above the 254.1 Hz band edge) |
| `_CTCSS_CONFIRM_MIN_S` | `0.4` | Audio that must accumulate before the measurement is valid | **Aligned with RTLSDR-Airband** ("slow" detector window) and the 0.7 s grace budget |
| `_CTCSS_CONFIRM_MAX_S` | `2.0` | Ring-buffer cap: how much recent audio is measured | **Derived** — ham2mon-specific (short/unthrottled transmissions) |
| `_CTCSS_MIN_PEAK_RATIO` | `3.0` | Peak must exceed the band median by this ratio | **Principle from RTLSDR-Airband / embedded.com** — the factor itself is our margin |
| `_CTCSS_ABSOLUTE_FLOOR` | `0.0001` | Absolute floor to reject silence | **Pre-existing** — mirrors ham2mon's `ctcss_level` |
| `_CTCSS_MATCH_TOLERANCE_HZ` | `1.0` | Acceptance window around a configured tone | **Derived** — below the tightest standard-code spacing; RTLSDR-Airband uses ±5 Hz |
| `_CTCSS_GRACE_PERIOD_S` | `0.7` | Time to decide before flagging a mismatch | **Pre-existing** — committed ham2mon constant |
| `_CTCSS_DETECTOR_LEN` | `4000` | Goertzel window (0.5 s @ 8 kHz) for each chain | **Pre-existing** — committed ham2mon constant |
| `ctcss_level` | `0.0001` | Goertzel chain level threshold | **Pre-existing** — committed ham2mon constant |

**Summary:** the *architecture* (Goertzel chains + full-band relative-power
measurement as authority, band-limited before detection) is the industry approach
embodied by RTLSDR-Airband and the ZL2PD decoder design. The exact numeric values
come from three sources: the EIA standard tone range, values aligned with
RTLSDR-Airband's implementation, and a handful of derived/heuristic choices unique to
ham2mon's topology and test suite.

---

## 3. Reference implementations

### 3.1 RTLSDR-Airband (`src/ctcss.cpp`, `src/squelch.cpp`)

RTLSDR-Airband is the closest reference: its squelch/CTCSS stage is a Goertzel tone
detector whose decision rule is exactly the "relative power" idea we use.

- **Band / tone set:** a hard-coded `standard_tones` vector covering
  `67.0 … 254.1 Hz` (the EIA/RS-220 set). Our band edges are identical.
- **Window / latency:** two detectors per channel — a fast 0.05 s window and a slow
  0.4 s window. The code comment states: *"0.4 sec is required to tell between all the
  standard tones."* Our 0.4 s measurement-validity floor uses the same value and the
  same rationale.
- **Decision rule:** at each window the monitored tones' powers are sorted; the
  configured tone is declared present when its power equals the highest power and is
  greater than the average power of the monitored tones:
  `ctcss_tone_power == tone_powers[0].power && ctcss_tone_power > avg_power`.
  This is the origin of our *peak-vs-median* relative-power check.
- **Adjacent-tone handling:** when a tone is configured, all standard tones within
  **±5 Hz** are excluded from the monitored set. We use a ±1.0 Hz acceptance tolerance
  instead (see §4).

### 3.2 ZL2PD "Developing a CTCSS Decoder"

Andrew Woodfield ZL2PD's decoder write-up is the source of the front-end band-limit
practice:

> A 260 Hz Low Pass (LP) filter is required to filter CTCSS audio tones
> (67–254 Hz) going into the CTCSS decoder. This reduces the potential for incoming
> speech to generate false decodes…

Our `_CTCSS_LP_HZ = 260.0` taps the decoded audio and low-pass filters it before the
measurement, exactly this design. The 260 Hz value is not arbitrary: it sits just
above the highest standard tone (254.1 Hz) and well below speech fundamentals.

### 3.3 embedded.com Goertzel articles

RTLSDR-Airband's `ctcss.cpp` is a port of the two embedded.com articles on Goertzel
CTCSS detection ("Detecting CTCSS tones with Goertzel's algorithm" and "The Goertzel
algorithm"). Those articles also use a relative-power decision:

> …comparing the power at each fᵢ with the sum of all of the other powers.

GNU Radio's `analog.ctcss_squelch_ff` (the chains in ham2mon) is the same Goertzel
technique, so the chains and the measurement are two expressions of one standard
approach.

---

## 4. Rationale for the derived / heuristic values

These are the levers not dictated by a standard or a reference implementation:

- **`_CTCSS_MIN_PEAK_RATIO = 3.0`.** RTLSDR-Airband accepts a tone when its power is
  the *highest* and *above average* — effectively a ratio of ≥1 against the average.
  We require the peak to exceed the band median by 3×. Rationale: the measurement
  scans the full band (not a hand-picked set of candidate tones), so the median is a
  cheap noise-floor estimate; a 3× margin makes the "strongest tone" claim
  unambiguous and suppresses voice/rumble energy. It is a strictness margin, not a
  value copied from another project.

- **`_CTCSS_MATCH_TOLERANCE_HZ = 1.0`.** Adjacent standard codes are separated by
  roughly 2.5–2.6 Hz in the tightest common pairs (71.9/74.4, 97.4/100.0); the
  extended 50-tone list includes an even closer pair (150.0/151.4 at 1.4 Hz).
  ±1.0 Hz is safely below the nearest-code separation, so a transmitted tone can only
  match one configured tone. Parabolic interpolation on the FFT peak (bin width
  1/2.0 s ≈ 0.5 Hz at 8 kHz) keeps the measured frequency accurate to well under 1 Hz
  for a clean tone **once the full 2.0 s ring has filled** (a tone at 254.1 Hz measures
  within ~0.1 Hz at the full ring). Two caveats:
  - At the earliest valid measurement (0.4 s), bins are ~2.5 Hz wide. Mid-band tones
    still interpolate to well under 0.2 Hz even at 10 dB SNR (measured), so the
    tolerance is not stressed there. The one weak spot is the **band top**: the
    254.1 Hz bin sits beyond the `side="right"` band cut, so the peak lands at the edge
    of the band array where parabolic interpolation is skipped — a 254.1 Hz tone reads
    ~1.6 Hz low at the 0.4 s window. In practice the 0.7 s grace budget lets the window
    grow to ≤1 Hz bins before a mismatch can fire, so the tone still matches; this is a
    known precision bound of the earliest measurement, not an observed failure.
  - This is stricter than RTLSDR-Airband's ±5 Hz exclusion; the trade-off is discussed
    in §6.

- **`_CTCSS_CONFIRM_MAX_S = 2.0`.** A pure "last 0.4 s snapshot" (as RTLSDR-Airband
  uses) fails ham2mon's committed test harness, where several CTCSS tests use
  unthrottled file sources that replay an entire 1.5 s transmission in a fraction of a
  second — by the time the match is checked, the tone may already be out of the recent
  window. Measuring a ring of up to 2.0 s keeps the tone visible for short signals
  while bounding staleness (a tone that stopped >2 s ago cannot latch a match, which
  matches the committed Goertzel mute behavior).

- **`_CTCSS_ABSOLUTE_FLOOR = 0.0001`.** Mirrors ham2mon's existing `ctcss_level`.
  Guards the ratio test against measuring pure silence (where both peak and median are
  ~0 and the ratio check is vacuous).

- **`_CTCSS_CONFIRM_MIN_S = 0.4`** interacts with the pre-existing
  `_CTCSS_GRACE_PERIOD_S = 0.7`: the measurement becomes valid at 0.4 s, the chains
  (4000-sample Goertzel window = 0.5 s) latch shortly after, and the match latches
  before the 0.7 s grace expires. The committed match tests assert a latched match at
  0.8 s, confirming this budget.

### 4.1 Measurement state lifecycle and memory profile

The measurement holds three pieces of state: the ring buffer `_ctcss_buffer`
(≤ `_CTCSS_CONFIRM_MAX_S` of audio), the validity counter `_ctcss_samples_seen`, and
the capture tap `_ctcss_capture` (a `blocks.vector_sink_f` drained on every
measurement poll).

- **Reset on every retune.** `_apply_ctcss_config()` calls
  `_reset_ctcss_measurement()` on every `set_center_freq` — both when a channel gets
  CTCSS tones and when it is cleared back to 0 (the empty branch). This mirrors the
  existing per-retune HPF-tap re-apply and guarantees a match/mismatch on a new channel
  can never be decided from the previous channel's leftover samples. `vector_sink_f`
  `reset()` clears the internal buffer without needing `data()`, so the reset is cheap
  and safe.
- **Known residual:** two back-to-back transmissions on the *same* channel with no
  retune in between (second starting within ~2 s of the first) still share the ring.
  If the second carried a *wrong* PL, up to ~2 s of stale audio could bias the first
  measurements. This mirrors the pre-existing chain behavior (Goertzel windows carry
  ~0.5 s across gaps) and is accepted; a reset on the RF-squelch rising edge was
  considered and deliberately not added (extra state, interacts with the mocked
  decision tests).
- **Memory profile (known, accepted):** the capture sink is drained only while
  `is_ctcss_mismatched()` is polled, i.e. for tuned demodulators. Growth is therefore
  bounded per dwell (a typical transmission, or between retunes). A demodulator parked
  idle at `center_freq == 0` is skipped by the scanner loop, so its sink accumulates at
  ~32 KB/s until the next tune event — acceptable for normal scanning cycles, but
  worth knowing for long unattended runs with demodulators left idle. A decoupled
  drain (always-on ring update + scanner call for idle demodulators) was designed but
  deliberately not implemented here.
- **Cross-thread access:** `_ctcss_capture.data()`/`reset()` run on the asyncio scanner
  loop while the GNU Radio scheduler thread concurrently calls `work()` on the block.
  This matches the existing convention of reading `unmuted()` off the Goertzel squelch
  blocks without locking; no new shared-mutable state is introduced.

---

## 5. How the values are pinned by tests

`apps/tests/test_receiver.py`:

- **Committed CTCSS tests** (all still passing) exercise the grace period, chain
  gating, multi-tone configuration, WBFM, recording metadata, and adjacent-tone
  rejection (97.4 vs 100.0).
- **New tests added with this change:**
  - `test_ctcss_match_high_pl_tone` — 141.3 Hz matches (band > 105 Hz; the regression
    the earlier experiment introduced).
  - `test_ctcss_match_band_edge_tone` — 254.1 Hz matches (upper band edge).
  - `test_ctcss_adjacent_high_tone_rejection` — 146.2 Hz does not match a 141.3 Hz
    channel (standard-spacing rejection in the upper band).
  - `test_ctcss_voice_falsing_rejection` → renamed to
    `test_ctcss_adjacent_tone_rejection_150hz` — a 150 Hz in-band narrowband tone
    does not false-match a 100 Hz channel (relative-power adjacent-tone behavior).
  - `test_ctcss_voice_falsing_rejection_broadband` — a broadband speech stand-in
    (harmonic stack up to 2400 Hz + noise, no CTCSS) does not false-match a 100 Hz
    channel; exercises the 260 Hz LP guard against speech-band energy.
  - `test_ctcss_retune_stale_buffer_no_falsing` — a demodulator retuned from a
    wrong-PL channel (88.5 Hz vs 100 Hz config) to a 141.3 Hz channel must match
    141.3 Hz from a *cleared* ring (regression for §4.1's reset-on-retune). Written
    red-first: it failed on the pre-fix tree with `_ctcss_samples_seen == 10880`
    left over from the previous dwell. (The detour through `set_center_freq(0)`
    clears the ring via the empty-tones branch, so it cannot distinguish
    "reset always fires" from "reset only on the empty branch".)
  - `test_ctcss_direct_preempt_stale_buffer_no_falsing` — covers the scanner's
    direct-preemption path (`scanner.py:_assign_channels_to_demodulators` calls
    `set_center_freq(channel.bb, ...)` on a live, mid-transmission demodulator with
    no detune to 0 in between). Retuning A→B while A is still on the air must still
    clear the ring: B matches 141.3 Hz from a clean ring. This is the hot-buffer
    path the detour test above cannot see, pinning that the reset fires on EVERY
    retune (non-empty → non-empty included).
- `test_ctcss_matching_logic_mocked` was updated to bind the measurement authority
  rather than the old chain-only decision.

These tests are what make the derived values (3.0 ratio, 1.0 Hz tolerance, 0.4/2.0 s
windows) concrete and regression-checkable.

---

## 6. Comparison with RTLSDR-Airband and open alternative

| Aspect | RTLSDR-Airband | ham2mon (this change) |
|---|---|---|
| Detection engine | Goertzel bank (chains) | Goertzel chains (gate) + full-band FFT (authority) |
| Band | 67.0–254.1 Hz | 67.0–254.1 Hz |
| Windows | fast 0.05 s, slow 0.4 s | measurement valid ≥0.4 s, ring ≤2.0 s |
| Relative-power rule | highest power && > average (~≥1×) | peak > 3 × median |
| Adjacent-tone rejection | exclude standard tones within ±5 Hz | accept only within ±1.0 Hz of a configured tone |
| Front-end filtering | (audio conditioning in pipeline) | 260 Hz LP guard before measurement |

**Open alternative (not currently implemented):** align more literally with
RTLSDR-Airband —

1. Replace `peak > 3 × median` with "peak is the band maximum **and** > band average"
   (ratio ~1×). This is closer to the reference but less strict against voice/rumble.
2. Replace the ±1.0 Hz acceptance with a ±5 Hz exclusion set (i.e., build the
   candidate-tone set from configured tones plus all standard tones except those
   within ±5 Hz of a configured tone, and require the configured tone to be the
   strongest). This matches the reference but changes rejection semantics at the
   97.4/100.0 and 150.0/151.4 spacings.

Adopting either would change behavior and require re-verifying the adjacent-tone and
voice-falsing tests in §5. The current values were kept because they satisfy the
committed test suite and are stricter on the failure modes (adjacent tones, voice
falsing) that matter for unattended scanning.

---

## 7. References

- RTLSDR-Airband, `src/ctcss.cpp` and `src/squelch.cpp`:
  https://github.com/rtl-airband/RTLSDR-Airband/blob/f8a17d7f/src/ctcss.cpp
  https://github.com/rtl-airband/RTLSDR-Airband/blob/f8a17d7f/src/squelch.cpp
- ZL2PD, "Developing a CTCSS Decoder": https://zl2pd.com/CTCSS_Decoder.html
- Embedded.com, "Detecting CTCSS tones with Goertzel's algorithm":
  https://www.embedded.com/detecting-ctcss-tones-with-goertzels-algorithm/
- Embedded.com, "The Goertzel algorithm": https://www.embedded.com/the-goertzel-algorithm/
- GNU Radio `ctcss_squelch_ff` (the chains' block):
  https://wiki.gnuradio.org/index.php/CTCSS_Squelch
