"""Create or update the Planning dashboard in Home Assistant: the new-trip form, the day-to-watch dropdown
and the trips page.

Runs on the HA host, where the SSH session has the Supervisor token:
    ssh <your-ha-host> 'python3 -' < tools/ha_dashboard.py
Uses only the standard library, because HA OS has no third-party Python packages.
"""
import base64, json, os, socket, struct, sys

def ws_connect():
    s = socket.create_connection(("supervisor", 80), timeout=30)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /core/websocket HTTP/1.1\r\nHost: supervisor\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(1)
    if b" 101 " not in resp.split(b"\r\n")[0]:
        raise SystemExit("upgrade failed: " + resp.decode(errors="replace").split("\r\n")[0])
    return s

def send(s, obj):
    data = json.dumps(obj).encode()
    mask = os.urandom(4)
    n = len(data)
    head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536
                            else bytes([0x80 | 127]) + struct.pack(">Q", n))
    s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise SystemExit("connection closed")
        buf += chunk
    return buf

def recv(s):
    payload = b""
    while True:
        b1, b2 = recv_exact(s, 2)
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", recv_exact(s, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", recv_exact(s, 8))[0]
        payload += recv_exact(s, n)
        if b1 & 0x80:
            return json.loads(payload)

s = ws_connect()
assert recv(s)["type"] == "auth_required"
send(s, {"type": "auth", "access_token": os.environ["SUPERVISOR_TOKEN"]})
auth = recv(s)
if auth["type"] != "auth_ok":
    raise SystemExit("auth failed")
msg_id = 0
def call(**kw):
    global msg_id
    msg_id += 1
    send(s, dict(id=msg_id, **kw))
    while True:
        r = recv(s)
        if r.get("id") == msg_id and r.get("type") == "result":
            return r

URL = "dashboard-planning"
existing = call(type="lovelace/dashboards/list")["result"]
if any(d["url_path"] == URL for d in existing):
    print("dashboard already exists, updating its cards")
else:
    r = call(type="lovelace/dashboards/create", url_path=URL, title="Planning", icon="mdi:sail-boat",
             show_in_sidebar=True, require_admin=False, mode="storage")
    print("create:", r["success"], r.get("error"))
config = {"title": "Planning", "views": [{
    "title": "Passage weather", "path": "passage", "icon": "mdi:sail-boat",
    "cards": [
        {"type": "entities", "title": "New trip", "entities": [
            {"entity": "input_text.passage_from", "name": "From"},
            {"entity": "input_text.passage_to", "name": "To"},
            {"entity": "input_datetime.passage_earliest", "name": "Earliest departure"},
            {"entity": "input_button.passage_new_trip", "name": "Start new trip"},
            {"type": "divider"},
            {"entity": "input_button.passage_cancel_all", "name": "Cancel all watches"}]},
        {"type": "entities", "title": "Day to watch", "entities": [
            {"entity": "input_select.passage_watch", "name": "Passage: day to watch"},
            {"entity": "input_button.passage_update_now", "name": "Update now"}]},
        {"type": "iframe", "url": "/local/passage/index.html?v=3", "aspect_ratio": "140%", "title": "Trips",
         "allow_open_top_navigation": True},
    ]}]}
r = call(type="lovelace/config/save", url_path=URL, config=config)
print("save:", r["success"], r.get("error"))
r = call(type="lovelace/config", url_path=URL)
print("readback cards:", [c["type"] for c in r["result"]["views"][0]["cards"]])
