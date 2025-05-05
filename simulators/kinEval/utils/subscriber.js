const zmq = require("zeromq");
const WebSocket = require("ws");

const sock = new zmq.Subscriber();
sock.connect("tcp://localhost:5555");
sock.subscribe("");
console.log("ZMQ Subscriber connected on port 5555");

// WebSocket server to send data to browser
const wss = new WebSocket.Server({ port: 8080 });
console.log("WebSocket server running on ws://localhost:8080");

function scaleInput(val, inDeadzone = 20, inMax = 100, outMin = 0.01, outMax = 0.5) {
    if (val >= -inDeadzone && val <= inDeadzone) return 0.0;
    if (val > inMax) val = inMax;
    if (val < -inMax) val = -inMax;

    const sign = val > 0 ? 1 : -1;
    val = Math.abs(val);

    const scaled = ((val - inDeadzone) / (inMax - inDeadzone)) * (outMax - outMin) + outMin;
    return scaled * sign;
}

async function fetchDepthAndBroadcast() {
    for await (const [msg] of sock) {
        try {
            const data = JSON.parse(msg.toString());
            // Optionally scale the values here
            const forward = scaleInput(data.forward ?? 0);
            const yaw = data.Yaw ?? 0;

            const payload = JSON.stringify({ forward, yaw });
            console.log(payload);
            // Broadcast to all connected clients
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(payload);
                }
            });
        } catch (err) {
            console.error("Failed to parse ZMQ message:", msg.toString());
        }
    }
}

fetchDepthAndBroadcast();
