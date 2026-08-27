# Available Components & Usage Guide

`ham2mon` includes a modular component architecture allowing custom evaluation gatekeepers (`WavGatekeeper`) and notification subscribers (`TransmissionNotifier`) to process recorded radio transmissions.

Below is a guide to available built-in components and how to configure them in your `config.yaml`.

---

## 1. TFLite Classifier Component (`TfliteClassifierComponent`)

* **Type**: `WavGatekeeper`
* **Class Path**: `components.tflite.TfliteClassifierComponent`
* **Description**: Evaluates completed WAV recordings using an isolated TensorFlow Lite (TFLite) machine learning classifier to categorize audio as Voice (`V`), Data (`D`), or Skip/Noise (`S`). The TFLite interpreter runs in an isolated external process via a lightweight helper service (`classification_service.py`), keeping heavy ML runtime dependencies decoupled from the core receiver process.

### Requirements & Installation

1. **Install LiteRT Extra (Recommended)**:
   Install the optional `ai-edge-litert` dependency via `uv`:
   ```bash
   uv sync --extra ai-edge-litert
   ```

### Configuration

Add `TfliteClassifierComponent` to your `config.yaml` under the `components.wav_gatekeeper` section:

```yaml
components:
  wav_gatekeeper:
    class_path: components.tflite.TfliteClassifierComponent
    timeout_sec: 5.0
    config:
      wanted:
        V: true
        D: true
        S: false
      model_path: apps/components/tflite/model/model_1.tflite
```

#### Options:
* `wanted`: Map of category strings (`V` for Voice, `D` for Data, `S` for Skip/Noise) to boolean values (`true` to keep, `false` to discard).
* `model_path`: Path to the `.tflite` model file (defaults to `apps/components/tflite/model/model_1.tflite`).

---

## 2. Silero VAD Component (`SileroVadComponent`)

* **Type**: `WavGatekeeper`
* **Class Path**: `components.silero.component.SileroVadComponent`
* **Description**: Evaluates completed WAV recordings using the official [Silero VAD (Voice Activity Detector)](https://github.com/snakers4/silero-vad) ONNX model to distinguish human voice from noise, static, data bursts, and squelch tails. Transmissions without detected speech are automatically discarded (`keep=False`).

### Requirements & Installation

1. **Install ONNX Runtime Extra**:
   Install the optional `onnxruntime` dependency via `uv`:
   ```bash
   uv sync --extra onnxruntime
   ```

2. **Manual Model Download**:
   Download the official `silero_vad.onnx` model file manually into the component model directory:
   ```bash
   mkdir -p apps/components/silero/model
   curl -L -o apps/components/silero/model/silero_vad.onnx https://raw.githubusercontent.com/snakers4/silero-vad/v4.0/files/silero_vad.onnx
   ```

### Configuration

Add `SileroVadComponent` to your `config.yaml` under the `components.wav_gatekeeper` section:

```yaml
components:
  wav_gatekeeper:
    class_path: components.silero.component.SileroVadComponent
    timeout_sec: 5.0
    config:
      model_path: apps/components/silero/model/silero_vad.onnx
      threshold: 0.5  # Confidence threshold (0.0 to 1.0)
```

#### Options:
* `model_path`: Relative or absolute path to `silero_vad.onnx` (defaults to `apps/components/silero/model/silero_vad.onnx`).
* `threshold`: Floating point confidence threshold between `0.0` and `1.0` (defaults to `0.5`). Higher values make voice detection stricter; lower values make it more sensitive.
* `min_voice_chunks`: Minimum number of 32ms frames required above threshold for voice decision (defaults to `3`). Higher values improve static noise and data burst rejection.
* `max_eval_sec`: Maximum seconds of audio evaluated per file (defaults to `3.0`). Limits long files to a centered evaluation window for faster inference. Set to `0.0` for full file evaluation.

---

## 3. Activity Logger Component (`ActivityLoggerComponent`)

* **Type**: `TransmissionNotifier`
* **Class Path**: `components.activity_logger_component.ActivityLoggerComponent`
* **Description**: Bridges `ham2mon`'s activity logging engine (such as `json-server` or `fixed-field`) with the component notification pipeline. Dispatches metadata records for kept transmissions to external endpoints or files (such as Home Assistant webhooks).

> [!NOTE]
> **Delegation & Configuration:**
> `ActivityLoggerComponent` delegates its destination, format type, and heartbeat interval settings directly to `ham2mon`'s core activity engine. Therefore, specific configuration parameters under the component's own `config: {}` dictionary are **not required**.
>
> To configure channel activity logging destinations, formats, and options, set the `channel_activity:` section in your `config.yaml` (or use CLI options like `--activity-type` and `--activity-dest`). For complete details on supported logging types, destinations, and CLI options, see the [Channel Activity Log File](../../README.md#channel-activity-log-file) section in the main README.

### Configuration Example

```yaml
# 1. Configure channel activity logging parameters:
channel_activity:
  type: json-server                 # Options: none, fixed-field, json-server
  dest: http://127.0.0.1:8000/activity  # Target file path or webhook URL
  interval_sec: 15                    # Periodic heartbeat interval in seconds

# 2. Register the ActivityLoggerComponent notifier:
components:
  notifiers:
    - class_path: components.activity_logger_component.ActivityLoggerComponent
      config: {}
```

---

## 4. Home Assistant MQTT Component (`HomeAssistantMqttComponent`)

* **Type**: `TransmissionNotifier`
* **Class Path**: `components.ha_mqtt.ha_mqtt_notify.HomeAssistantMqttComponent`
* **Description**: Publishes MQTT Discovery configs so entities appear automatically in Home Assistant (no manual `configuration.yaml` editing), then pushes per-transmission state updates. Each `ham2mon` instance registers as one HA device (keyed by `instance`), with two entities: `sensor.<instance>_last_transmission` (frequency plus full record attributes) and `binary_sensor.<instance>_voice_activity` (auto-clears via HA's `off_delay`). Availability is tracked via an MQTT Last Will and Testament plus explicit `online`/`offline` publishes.
* **Bank support**: resolved bank tags are read per-transmission from `TransmissionRecord.banks` and published in the state payload as a JSON list — banks are **not** part of this component's config.

The `last_transmission/state` payload is a JSON object published on `ham2mon/<instance>/last_transmission/state` with: `freq` (MHz, also the sensor state), `duration`, `priority`, `strength`, `ctcss`, `label`, `classification`, `wav_path` (location of the saved clip on the ham2mon host, relative to its working directory), `created`, and `banks`. The sensor reports `unit_of_measurement: "MHz"`. To watch the raw payloads from any LAN host:

```bash
mosquitto_sub -h <broker_host> -u <mqtt_user> -P <mqtt_pass> -t 'ham2mon/#' -v
```

(In Home Assistant: **Settings → Devices & Services → MQTT → Listen to topics**, subscribe to `ham2mon/#`.)

### Requirements & Installation

1. **Install `aiomqtt`** (declared as a core dependency, installed automatically with `uv sync`):
   ```bash
   uv sync
   ```

### Configuration

Register `HomeAssistantMqttComponent` in your `config.yaml` under the `components.notifiers` list:

```yaml
components:
  notifiers:
    - class_path: components.ha_mqtt.ha_mqtt_notify.HomeAssistantMqttComponent
      config:
        instance: "SDR1"          # or ENV: H2M_INSTANCE (defaults to hostname)
        broker_host: "localhost"  # or ENV: MQTT_BROKER_HOST
        broker_port: 1883
        username: null            # or ENV: MQTT_USERNAME
        password: null            # or ENV: MQTT_PASSWORD
        wanted: "V"               # only publish this classification code
        off_delay_sec: 5
        discovery_prefix: "homeassistant"
```

#### Options:
* `instance`: Stable identifier for this `ham2mon` deployment, used for the MQTT topic namespace (`ham2mon/<instance>/...`) and the HA device/entity names. Defaults to the value of `H2M_INSTANCE` or the system hostname.
* `broker_host` / `broker_port`: MQTT broker address (defaults to `localhost:1883`, `MQTT_BROKER_HOST` env override).
* `username` / `password`: MQTT broker credentials (or `MQTT_USERNAME` / `MQTT_PASSWORD` env overrides).
* `wanted`: Classification code to forward to Home Assistant (default `"V"`). Transmissions classified otherwise are still recorded but not published.
* `off_delay_sec`: Seconds before HA auto-resets `voice_activity` to `off` after the last transmission (default `5`).
* `discovery_prefix`: Home Assistant MQTT discovery prefix (default `homeassistant`).

#### Example: Alert automation in Home Assistant

The sensor's state is the last received frequency (`MHz`) and the state JSON is exposed as the entity's attributes, so HA automations can alert on any frequency/tone combination without touching `configuration.yaml`. This example posts a persistent notification whenever FRS channel 14 (467.7125 MHz) with privacy code 8 (88.5 Hz CTCSS) is keyed up:

```yaml
alias: FRS Ch14 Code 8 Alert
description: Notify when FRS channel 14 (467.7125) with privacy code 8 (88.5 Hz) is keyed up
triggers:
  - trigger: state
    entity_id: sensor.ham2mon_sdr1_last_transmission
    attribute: created
conditions:
  - condition: template
    value_template: "{{ state_attr('sensor.ham2mon_sdr1_last_transmission', 'freq') | float(0) | round(4) == 467.7125 }}"
  - condition: template
    value_template: "{{ (state_attr('sensor.ham2mon_sdr1_last_transmission', 'ctcss') or '') | replace('Hz', '') | float(0) | round(1) == 88.5 }}"
actions:
  - action: notify.persistent_notification
    data:
      title: "FRS Channel 14 · Code 8"
      message: "Keyed up at {{ now().strftime('%H:%M:%S') }} — clip: {{ state_attr('sensor.ham2mon_sdr1_last_transmission', 'wav_path') }}"
mode: single
```

Notes:
- **The entity id encodes your instance**: it is `sensor.ham2mon_<instance_lowercase>_last_transmission`. Use the exact entity shown in **Developer Tools → States** (search `ham2mon`), or derive it from the sensor's `friendly_name` (`ham2mon (<instance>) Last Transmission`). The example above uses `SDR1` — if your `instance` is different, every reference must be updated, or the trigger/conditions will silently evaluate against a nonexistent entity (`state_attr` returns `None`, `condition: state` fails).
- The trigger watches the `created` **attribute** (`trigger: state` + `attribute:`) so *every* completed transmission fires, including rapid consecutive same-frequency key-ups. Do not use a template trigger for this: template triggers only re-evaluate when the entity's *state* changes, and since this sensor's state is the frequency, a single-frequency channel changes state only once (on the first transmission).
- Conditions read the `freq` and `ctcss` **attributes** (not the sensor state). The `freq` attribute is always present and compared as a float (`round(4)` absorbs the 4-decimal format and float noise); the `ctcss` check tolerates the `88.5Hz` suffix and fails cleanly (evaluates to `False`) when `ctcss` is `null`.
- The `ctcss` attribute is only non-`null` when CTCSS tone squelch is enabled (`receiver.max_ctcss_tones` > 0) and the frequency entry declares `tones:` (e.g. `tones: [88.5]`).
- Swap `notify.persistent_notification` for `notify.mobile_app_<device>` to push to a phone.
