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
