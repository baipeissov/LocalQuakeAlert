# LocalQuakeAlert — Setup Guide

## Files
- `dashboard.html` — web dashboard (open in browser or serve via server.js)
- `app.html`        — mobile PWA app (install on phone as app)
- `server.js`       — Node.js backend: MQTT + WebSocket + Telegram + ML
- `sw.js`           — Service worker for background push notifications

---

## Quick Start (14 minutes)

### 1. Install dependencies
```bash
npm install express ws mqtt node-telegram-bot-api
```

### 2. Configure server.js
Open `server.js` and set:
```js
telegramToken: 'GET_FROM_BOTFATHER',   // @BotFather on Telegram
telegramChatId: 'GET_FROM_USERINFOBOT' // @userinfobot on Telegram
```

### 3. Run the server
```bash
node server.js
```

### 4. Open dashboard
- Web: http://localhost:3000/dashboard.html
- Mobile app: http://localhost:3000/app.html
  - On phone: tap Share → "Add to Home Screen" to install as PWA

---

## Where to plug in your ML model

Open `server.js` and find the `runMLModel(features)` function (~line 70).

Replace the stub with your model. Three options:

### Option A — Python model (scikit-learn / PyTorch / Keras .h5)
Create `predict.py`:
```python
import sys, pickle, numpy as np
model = pickle.load(open('model.pkl','rb'))  # or tf.keras.models.load_model()
mag, zscore, stalta, sw420 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
X = np.array([[mag, zscore, stalta, sw420]])
pred = model.predict(X)[0]
proba = model.predict_proba(X)[0].max()
# Output must be exactly: SEISMIC 0.94  OR  NORMAL 0.97  OR  NOISE 0.91
print(f"{'SEISMIC' if pred==1 else 'NORMAL'} {proba:.4f}")
```

In server.js `runMLModel()`:
```js
const { execSync } = require('child_process');
const result = execSync(
  `python3 predict.py ${features.mag} ${features.zscore} ${features.stalta} ${features.sw420}`
).toString().trim();
const [prediction, conf] = result.split(' ');
return { prediction, confidence: parseFloat(conf) };
```

### Option B — ONNX model (fastest, runs natively in Node)
```bash
npm install onnxruntime-node
```
```js
const ort = require('onnxruntime-node');
const session = await ort.InferenceSession.create('./model.onnx');
const feeds = {
  input: new ort.Tensor('float32',
    [features.mag, features.zscore, features.stalta, features.sw420],
    [1, 4])
};
const out = await session.run(feeds);
const confidence = out.output.data[0];
return { prediction: confidence > 0.5 ? 'SEISMIC' : 'NORMAL', confidence };
```

### Option C — TensorFlow.js
```bash
npm install @tensorflow/tfjs-node
```
```js
const tf = require('@tensorflow/tfjs-node');
const model = await tf.loadLayersModel('file://./tfmodel/model.json');
const input = tf.tensor2d([[features.mag, features.zscore, features.stalta, features.sw420]]);
const pred = model.predict(input);
const [confidence] = await pred.data();
return { prediction: confidence > 0.5 ? 'SEISMIC' : 'NORMAL', confidence };
```

---

## Features passed to the model
| Feature | Description |
|---------|-------------|
| `mag`    | Vibration magnitude in g |
| `zscore` | Z-score vs rolling baseline |
| `stalta` | STA/LTA ratio |
| `sw420`  | 1 if SW-420 pre-triggered, 0 otherwise |

## Expected output
The model should return one of:
- `SEISMIC` — real earthquake → alarm fires
- `NORMAL`  — normal baseline → no action
- `NOISE`   — known noise (footstep, door, truck) → alarm suppressed

---

## ESP32 MQTT payload format
The ESP32 must publish JSON to topic `lqa/sensor/data`:
```json
{ "mag": 0.003, "zscore": 0.3, "stalta": 1.02, "sw420": 0 }
```

Add to your ESP32 firmware (after alarm detection):
```cpp
#include <WiFi.h>
#include <PubSubClient.h>

WiFiClient espClient;
PubSubClient mqtt(espClient);

void publishData(float mag, float zscore, float stalta, int sw420) {
  char payload[128];
  snprintf(payload, sizeof(payload),
    "{\"mag\":%.4f,\"zscore\":%.2f,\"stalta\":%.2f,\"sw420\":%d}",
    mag, zscore, stalta, sw420);
  mqtt.publish("lqa/sensor/data", payload);
}
```

---

## Threshold update from dashboard
When you change thresholds in the dashboard and click "Push to device",
the server sends MQTT to topic `lqa/device/thresholds`:
```json
{ "zThreshold": 3.5, "staltatreshold": 2.5, "alarmLatch": 30, "mlEnabled": true }
```

Add to ESP32 firmware:
```cpp
void mqttCallback(char* topic, byte* payload, unsigned int len) {
  if (strcmp(topic, "lqa/device/thresholds") == 0) {
    // parse JSON and update Z_THRESHOLD, STALTA_THRESHOLD etc.
    DynamicJsonDocument doc(256);
    deserializeJson(doc, payload, len);
    Z_THRESHOLD    = doc["zThreshold"];
    STALTA_THRESHOLD = doc["staltatreshold"];
  }
}
```