"""Small browser-based remote control for deploy simulations."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Go2Arm Remote</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111318;
      color: #e9edf2;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #111318;
    }
    main {
      width: min(960px, calc(100vw - 32px));
      display: grid;
      gap: 18px;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      font-size: 28px;
      margin: 0;
      font-weight: 650;
    }
    .status {
      color: #9fb0c3;
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    section {
      border: 1px solid #2a303a;
      border-radius: 8px;
      padding: 18px;
      background: #171b22;
    }
    h2 {
      font-size: 16px;
      margin: 0 0 14px;
      color: #cbd5e1;
      font-weight: 620;
    }
    .row {
      display: grid;
      grid-template-columns: 88px 1fr 72px;
      gap: 12px;
      align-items: center;
      min-height: 38px;
    }
    label {
      color: #aab7c7;
      font-size: 14px;
    }
    output {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: #f8fafc;
    }
    input[type="range"] {
      width: 100%;
      accent-color: #47b5ff;
    }
    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    button {
      border: 1px solid #394251;
      border-radius: 6px;
      background: #202633;
      color: #f8fafc;
      padding: 10px 14px;
      font-size: 14px;
      cursor: pointer;
    }
    button:hover {
      background: #293142;
    }
    button.primary {
      border-color: #2f7db3;
      background: #185b89;
    }
    @media (max-width: 760px) {
      .grid { grid-template-columns: 1fr; }
      header { display: block; }
      .row { grid-template-columns: 76px 1fr 64px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Go2Arm Remote</h1>
      <div class="status" id="status">connecting</div>
    </header>
    <div class="grid">
      <section>
        <h2>Base Velocity</h2>
        <div id="base"></div>
        <div class="buttons">
          <button class="primary" onclick="zeroBase()">Zero Base</button>
        </div>
      </section>
      <section>
        <h2>End Effector Position</h2>
        <div id="ee"></div>
        <div class="buttons">
          <button onclick="resetEe()">Reset EE</button>
        </div>
      </section>
    </div>
  </main>
  <script>
    const specs = {
      base: [
        ["lin_vel_x", "x", -0.3, 0.8, 0.01],
        ["lin_vel_y", "y", -0.5, 0.5, 0.01],
        ["ang_vel_z", "yaw", -0.8, 0.8, 0.01],
      ],
      ee: [
        ["x", "x", 0.20, 0.55, 0.005],
        ["y", "y", -0.30, 0.30, 0.005],
        ["z", "z", 0.15, 0.55, 0.005],
      ],
    };

    function makeSlider(group, [key, label, min, max, step]) {
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `
        <label>${label}</label>
        <input id="${group}-${key}" type="range" min="${min}" max="${max}" step="${step}" value="0">
        <output id="${group}-${key}-out">0.000</output>
      `;
      row.querySelector("input").addEventListener("input", sendFromSliders);
      return row;
    }

    for (const spec of specs.base) document.getElementById("base").appendChild(makeSlider("base", spec));
    for (const spec of specs.ee) document.getElementById("ee").appendChild(makeSlider("ee", spec));

    function readGroup(group, spec) {
      const values = {};
      for (const [key] of spec) values[key] = Number(document.getElementById(`${group}-${key}`).value);
      return values;
    }

    async function post(payload) {
      const res = await fetch("/command", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      await refresh();
    }

    async function sendFromSliders() {
      await post({base: readGroup("base", specs.base), ee: readGroup("ee", specs.ee)});
    }

    async function zeroBase() {
      await post({base: {lin_vel_x: 0, lin_vel_y: 0, ang_vel_z: 0}});
    }

    async function resetEe() {
      await post({reset_ee: true});
    }

    function setSlider(group, key, value) {
      const slider = document.getElementById(`${group}-${key}`);
      const output = document.getElementById(`${group}-${key}-out`);
      slider.value = value;
      output.value = Number(value).toFixed(3);
    }

    async function refresh() {
      const res = await fetch("/state");
      const state = await res.json();
      setSlider("base", "lin_vel_x", state.base[0]);
      setSlider("base", "lin_vel_y", state.base[1]);
      setSlider("base", "ang_vel_z", state.base[2]);
      setSlider("ee", "x", state.ee[0]);
      setSlider("ee", "y", state.ee[1]);
      setSlider("ee", "z", state.ee[2]);
      document.getElementById("status").textContent =
        `sim t=${state.time.toFixed(2)}s  base z=${state.base_z.toFixed(3)}m`;
    }

    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


class RemoteControlServer:
  def __init__(self, adapter: Any, host: str = "127.0.0.1", port: int = 8765):
    self.adapter = adapter
    self.host = host
    self.port = port
    self.httpd: ThreadingHTTPServer | None = None
    self.thread: threading.Thread | None = None

  @property
  def url(self) -> str:
    return f"http://{self.host}:{self.port}"

  def start(self) -> None:
    adapter = self.adapter

    class Handler(BaseHTTPRequestHandler):
      def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
          self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
          return
        if self.path == "/state":
          self._send_json(adapter.command_state())
          return
        self._send(404, b"not found", "text/plain")

      def do_POST(self) -> None:
        if self.path != "/command":
          self._send(404, b"not found", "text/plain")
          return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        adapter.apply_remote_command(payload)
        self._send_json(adapter.command_state())

      def log_message(self, format: str, *args: Any) -> None:
        return

      def _send_json(self, payload: dict[str, Any]) -> None:
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")

      def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
    self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
    self.thread.start()
    print(f"Command server: {self.url}")
    print("Browser UI is available at the same URL, but it is optional.")

  def stop(self) -> None:
    if self.httpd is not None:
      self.httpd.shutdown()
      self.httpd.server_close()
    if self.thread is not None:
      self.thread.join(timeout=0.5)
