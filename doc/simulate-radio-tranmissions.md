# Overview

The test case [`test_busy_net.py`](../apps/tests/test_busy_net.py) can simulate a busy radio network. This can be useful for the following purposes:

- Testing the Receiver class with "realistic" scenarios
    - The existing ones in [`test_receiver.py`](../apps/tests/test_receiver.py) are basic tests intended to isolate the functionality
- Generate an IQ file and loading it into ham2mon for viewing

## Testing the Receiver class with "realistic" scenarios

The scope of this test is unit testing the Receiver class. Therefore, functions like priority and locked channel handling cannot be exercised by this test case.

```bash
uv run pytest apps/tests/test_busy_net.py --run-busy-net
```

## Generating an IQ file and loading it into ham2mon

### Generating an IQ file

Generate an IQ file using the [`test_busy_net.py`](../apps/tests/test_busy_net.py) test case:

```bash
uv run pytest apps/tests/test_busy_net.py --run-busy-net --persist-iq
```

**Optional**: See [README.md](../README.md#test-debugging-options) for additional options including generating a plot (png) of the session or retaining the wav file for further analysis.

### Loading the IQ file into ham2mon

Load the file without referencing the frequency configuration:

```bash
uv run apps/ham2mon.py -a "file=test_signals/debug/test_busy_net_realistic_scanning_session_signal_busy_net.iq,rate=1E6,repeat=true,throttle=true,freq=462.550E6" -r 1E6 -t 30 -d 0 -s -70 -v 20 -w -b 16 -n 3 -M 70
```

**Note:** To load the file with the frequency configuration, use the `-F` option (e.g. `-F doc/frequencies-example.yaml`).  The frequencies in the [frequencies-example.yaml](frequencies-example.yaml) file have been synced up with the frequencies in the test case.

**Note:** If your frequency configuration includes CTCSS tones (as used by the example file), you must also enable CTCSS demodulation by specifying the max tones (e.g., `--max-ctcss-tones=3`).

See [README.md](../README.md#help-menu) for information on the command line options.

A future enhancement would be to extend this capability into an integration test framework.
