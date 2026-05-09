/**
 * LocalQuakeAlert — Backend Server
 * Node.js + Express + WebSocket + MQTT + Telegram
 *
 * npm install express ws mqtt node-telegram-bot-api
 * node server.js
 */

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const mqtt = require('mqtt');
const TelegramBot = require('node-telegram-bot-api');
const path = require('path');

// ─── CONFIG ─────────────────────────────────────────────────────────────────
const CONFIG = {
    port: 3000,

    // MQTT — free cloud broker, no signup needed for testing
    mqttBroker: 'mqtt://broker.hivemq.com',
    mqttTopicIn: 'lqa/sensor/data',       // ESP32 publishes here
    mqttTopicCmd: 'lqa/device/thresholds', // server publishes here → ESP32 listens

    // Telegram — get token from @BotFather, chat_id from @userinfobot
    telegramToken: 'YOUR_BOT_TOKEN_HERE',
    telegramChatId: 'YOUR_CHAT_ID_HERE',

    // Detection thresholds (can be updated at runtime via /api/thresholds)
    thresholds: {
        zScore: 3.5,
        stalLta: 2.5,
        alarmLatch: 30000,  // ms
        mlEnabled: true,
        mlOverride: true,   // if ML says NOISE with >90% confidence, suppress alarm
    }
};
// ─────────────────────────────────────────────────────────────────────────────

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
app.use(express.json());
app.use(express.static(path.join(__dirname)));  // serves dashboard.html & app.html

// ─── TELEGRAM ────────────────────────────────────────────────────────────────
let bot = null;
if (CONFIG.telegramToken !== 'YOUR_BOT_TOKEN_HERE') {
    bot = new TelegramBot(CONFIG.telegramToken, { polling: false });
}
function sendTelegram(msg) {
    if (!bot) { console.log('[Telegram DEMO]', msg); return; }
    bot.sendMessage(CONFIG.telegramChatId, msg).catch(console.error);
}

// ─── WEBSOCKET — broadcast to all dashboard/app clients ──────────────────────
function broadcastWS(data) {
    const msg = JSON.stringify(data);
    wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) client.send(msg);
    });
}
wss.on('connection', ws => {
    console.log('[WS] client connected');
    ws.send(JSON.stringify({ type: 'welcome', msg: 'LocalQuakeAlert server online' }));
});

// ─── ML MODEL INTEGRATION ────────────────────────────────────────────────────
//
//  ╔══════════════════════════════════════════════════════════════════════════╗
//  ║  THIS IS WHERE YOUR ML MODEL GOES                                       ║
//  ║                                                                          ║
//  ║  Option A — Python model via child_process (scikit-learn, TF, PyTorch): ║
//  ║    const { execSync } = require('child_process');                        ║
//  ║    const result = execSync(                                              ║
//  ║      `python3 predict.py ${features.mag} ${features.zscore} ${features.stalta}` ║
//  ║    ).toString().trim();  // your predict.py prints "SEISMIC 0.94"        ║
//  ║    const [prediction, confidence] = result.split(' ');                  ║
//  ║                                                                          ║
//  ║  Option B — ONNX model (runs natively in Node, very fast):              ║
//  ║    const ort = require('onnxruntime-node');  // npm install onnxruntime-node ║
//  ║    const session = await ort.InferenceSession.create('./model.onnx');    ║
//  ║    const feeds = { input: new ort.Tensor('float32',                     ║
//  ║      [features.mag, features.zscore, features.stalta, features.sw420],  ║
//  ║      [1, 4]) };                                                          ║
//  ║    const results = await session.run(feeds);                             ║
//  ║    // parse results.output.data[0]                                       ║
//  ║                                                                          ║
//  ║  Option C — TensorFlow.js (if your model is .h5 or SavedModel):         ║
//  ║    const tf = require('@tensorflow/tfjs-node');                          ║
//  ║    const model = await tf.loadLayersModel('file://./model/model.json');  ║
//  ║    const input = tf.tensor2d([[features.mag, features.zscore,            ║
//  ║                                features.stalta, features.sw420]]);       ║
//  ║    const pred = model.predict(input);                                    ║
//  ║    const [confidence] = await pred.data();                               ║
//  ║    const prediction = confidence > 0.5 ? 'SEISMIC' : 'NORMAL';          ║
//  ║                                                                          ║
//  ║  FEATURES YOU PASS TO THE MODEL:                                         ║
//  ║    features.mag     — vibration magnitude (g)                            ║
//  ║    features.zscore  — Z-score vs baseline                                ║
//  ║    features.stalta  — STA/LTA ratio                                      ║
//  ║    features.sw420   — 1 if SW-420 pre-triggered, else 0                  ║
//  ║    features.freq    — dominant frequency (Hz) if your ESP32 sends it     ║
//  ║                                                                          ║
//  ║  WHAT TO RETURN:                                                         ║
//  ║    { prediction: 'SEISMIC' | 'NORMAL' | 'NOISE', confidence: 0.0-1.0 }  ║
//  ╚══════════════════════════════════════════════════════════════════════════╝
//
// ─── ML MODEL PROCESS (spawned once, kept alive for speed) ───────────────────
// predict.py is loaded once and runs as a persistent child process.
// This avoids the ~200ms Python startup cost on every prediction.
const { spawn, execSync } = require('child_process');

let mlProcess = null;
let mlReady = false;
let pendingResolves = [];
let outputBuffer = '';

function startMLProcess() {
    try {
        mlProcess = spawn('python3', ['-u', 'predict_server.py']);  // persistent mode
        mlProcess.stdout.on('data', (data) => {
            outputBuffer += data.toString();
            const lines = outputBuffer.split('\n');
            outputBuffer = lines.pop();                               // keep incomplete line
            lines.forEach(line => {
                if (line.trim() && pendingResolves.length > 0) {
                    const resolve = pendingResolves.shift();
                    resolve(line.trim());
                }
            });
        });
        mlProcess.stderr.on('data', d => console.error('[ML stderr]', d.toString().trim()));
        mlProcess.on('close', () => { mlProcess = null; mlReady = false; });
        mlReady = true;
        console.log('[ML] persistent Python process started');
    } catch (e) {
        console.log('[ML] could not start persistent process, using per-call mode');
    }
}
startMLProcess();

async function runMLModel(features) {
    const t0 = Date.now();

    try {
        let result;

        // ── Mode 1: persistent Python process (fastest, ~1-3ms) ──────────────
        if (mlProcess && mlReady) {
            result = await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject(new Error('ML timeout')), 2000);
                pendingResolves.push((line) => { clearTimeout(timeout); resolve(line); });
                mlProcess.stdin.write(
                    `${features.mag} ${features.zscore} ${features.stalta} ${features.sw420 || 0}\n`
                );
            });
        }

        // ── Mode 2: per-call execSync (fallback, ~30-80ms with Python startup) ─
        if (!result) {
            result = execSync(
                `python3 predict.py ${features.mag} ${features.zscore} ${features.stalta} ${features.sw420 || 0}`,
                { timeout: 500, cwd: __dirname }
            ).toString().trim();
        }

        // ── Parse output: "SEISMIC 0.9400 1.234 ensemble" ────────────────────
        const [prediction, confStr, latencyStr, method] = result.split(' ');
        const confidence = parseFloat(confStr) || 0.97;
        const mlLatency = parseFloat(latencyStr) || 0;

        console.log(`[ML] ${prediction} (${(confidence * 100).toFixed(1)}%) via ${method || '?'} in ${mlLatency.toFixed(2)}ms | e2e: ${Date.now() - t0}ms`);
        return { prediction: prediction || 'NORMAL', confidence };

    } catch (err) {
        console.error('[ML] error:', err.message, '— using rule-based fallback');

        // ── Fallback: classical threshold logic (always works, no Python needed) ─
        if (features.zscore > 3.5 && features.stalta > 2.5) {
            return { prediction: 'SEISMIC', confidence: Math.min(0.90, features.zscore / 10) };
        }
        if (features.sw420 && features.zscore < 2.0) {
            return { prediction: 'NOISE', confidence: 0.87 };
        }
        return { prediction: 'NORMAL', confidence: 0.97 };
    }
}

// ─── ALARM STATE ─────────────────────────────────────────────────────────────
let alarmActive = false;
let alarmTimeout = null;

async function processReading(data) {
    /**
     * data from ESP32 via MQTT:
     * { mag: float, zscore: float, stalta: float, sw420: 0|1, ts: timestamp }
     */
    const t0 = Date.now();

    // 1. Run ML model
    let mlResult = { prediction: 'NORMAL', confidence: 0.97 };
    if (CONFIG.thresholds.mlEnabled) {
        mlResult = await runMLModel(data);
    }

    // 2. Alarm decision logic
    //    DUAL CONDITION: Z-score AND STA/LTA must exceed thresholds
    //    PLUS ML check: if ML says NOISE with high confidence, suppress
    const rawAlarm = data.zscore > CONFIG.thresholds.zScore
        && data.stalta > CONFIG.thresholds.stalLta;

    const mlSuppressed = CONFIG.thresholds.mlOverride
        && mlResult.prediction === 'NOISE'
        && mlResult.confidence > 0.90;

    const shouldAlarm = rawAlarm && !mlSuppressed;

    // 3. Fire alarm (with latch to prevent repeat triggers)
    if (shouldAlarm && !alarmActive) {
        alarmActive = true;

        const alarmMsg = [
            '⚠️ *SEISMIC ALARM* — LocalQuakeAlert',
            `Magnitude: ${data.mag.toFixed(4)}g`,
            `Z-score: ${data.zscore.toFixed(2)} (threshold: ${CONFIG.thresholds.zScore})`,
            `STA/LTA: ${data.stalta.toFixed(2)} (threshold: ${CONFIG.thresholds.stalLta})`,
            `ML: ${mlResult.prediction} (${(mlResult.confidence * 100).toFixed(1)}% confidence)`,
            `Time: ${new Date().toISOString()}`,
            `Unit: Almaty Unit-01`
        ].join('\n');

        sendTelegram(alarmMsg);
        console.log('[ALARM TRIGGERED]', alarmMsg);

        // Auto-reset alarm after latch period
        if (alarmTimeout) clearTimeout(alarmTimeout);
        alarmTimeout = setTimeout(() => {
            alarmActive = false;
            console.log('[ALARM RESET]');
        }, CONFIG.thresholds.alarmLatch);
    }

    // 4. Broadcast to all connected dashboard / app clients
    const responseTime = Date.now() - t0;
    broadcastWS({
        mag: data.mag,
        zscore: data.zscore,
        stalta: data.stalta,
        sw420: data.sw420,
        alarm: shouldAlarm,
        ml_prediction: mlResult.prediction,
        ml_confidence: mlResult.confidence,
        response_ms: responseTime,
        ts: Date.now()
    });
}

// ─── MQTT ─────────────────────────────────────────────────────────────────────
const mqttClient = mqtt.connect(CONFIG.mqttBroker);

mqttClient.on('connect', () => {
    console.log('[MQTT] connected to', CONFIG.mqttBroker);
    mqttClient.subscribe(CONFIG.mqttTopicIn, err => {
        if (!err) console.log('[MQTT] subscribed to', CONFIG.mqttTopicIn);
    });
});

mqttClient.on('message', (topic, message) => {
    if (topic !== CONFIG.mqttTopicIn) return;
    try {
        const data = JSON.parse(message.toString());
        // data = { mag, zscore, stalta, sw420 }
        processReading(data);
    } catch (e) {
        console.error('[MQTT] parse error:', e.message);
    }
});

mqttClient.on('error', err => console.error('[MQTT error]', err.message));

// ─── REST API ─────────────────────────────────────────────────────────────────

// GET /api/status — health check
app.get('/api/status', (req, res) => {
    res.json({
        ok: true,
        alarm: alarmActive,
        thresholds: CONFIG.thresholds,
        uptime: process.uptime()
    });
});

// POST /api/thresholds — update thresholds from dashboard/app
app.post('/api/thresholds', (req, res) => {
    const t = req.body;
    if (t.zThreshold) CONFIG.thresholds.zScore = t.zThreshold;
    if (t.staltatreshold) CONFIG.thresholds.stalLta = t.staltatreshold;
    if (t.alarmLatch) CONFIG.thresholds.alarmLatch = t.alarmLatch * 1000;
    if (t.mlEnabled !== undefined) CONFIG.thresholds.mlEnabled = t.mlEnabled;
    if (t.mlOverride !== undefined) CONFIG.thresholds.mlOverride = t.mlOverride;

    // Push updated thresholds to ESP32 via MQTT
    mqttClient.publish(CONFIG.mqttTopicCmd, JSON.stringify(CONFIG.thresholds));
    console.log('[THRESHOLDS] updated:', CONFIG.thresholds);
    res.json({ ok: true, thresholds: CONFIG.thresholds });
});

// POST /api/inject — simulate sensor reading (for demo without hardware)
app.post('/api/inject', async (req, res) => {
    const data = {
        mag: req.body.mag || 0.003,
        zscore: req.body.zscore || 0.3,
        stalta: req.body.stalta || 1.0,
        sw420: req.body.sw420 || 0,
    };
    await processReading(data);
    res.json({ ok: true });
});

// ─── START ────────────────────────────────────────────────────────────────────
server.listen(CONFIG.port, () => {
    console.log(`\n┌─────────────────────────────────────┐`);
    console.log(`│  LocalQuakeAlert Server             │`);
    console.log(`│  http://localhost:${CONFIG.port}               │`);
    console.log(`│  WS:   ws://localhost:${CONFIG.port}            │`);
    console.log(`│  MQTT: ${CONFIG.mqttBroker}  │`);
    console.log(`└─────────────────────────────────────┘\n`);
    console.log('Dashboard: http://localhost:3000/dashboard.html');
    console.log('Mobile app: http://localhost:3000/app.html\n');
});

// ─── DEMO MODE — send fake data every 200ms if no ESP32 connected ─────────────
// Comment this out once real hardware is connected
let demoInterval = setInterval(() => {
    processReading({
        mag: Math.random() * 0.005,
        zscore: 0.2 + Math.random() * 0.3,
        stalta: 1.0 + Math.random() * 0.1,
        sw420: 0
    });
}, 200);

// Stop demo mode as soon as first real MQTT message arrives
mqttClient.once('message', () => {
    clearInterval(demoInterval);
    demoInterval = null;
    console.log('[Demo mode OFF] — real sensor data connected');
});