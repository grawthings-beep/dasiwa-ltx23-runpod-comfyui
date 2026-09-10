#!/usr/bin/env python3
"""Small dependency-free status page used while WAN assets are provisioned."""

import argparse
import html
import json
import os
import pathlib
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_STATUS = {
    "state": "initializing",
    "phase": "container",
    "message": "WAN loop runtime is initializing.",
}


def read_status(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else dict(DEFAULT_STATUS)
    except (OSError, ValueError, TypeError):
        return dict(DEFAULT_STATUS)


def write_status(path, updates, reset=False, preserve_failure=False):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(DEFAULT_STATUS) if reset else read_status(path)
    if preserve_failure and data.get("state") == "failed":
        return data
    data.update({key: value for key, value in updates.items() if value is not None})
    data["updated_at"] = int(time.time())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return data


def render_page(data):
    state = html.escape(str(data.get("state", "initializing")))
    phase = html.escape(str(data.get("phase", "container")))
    message = html.escape(str(data.get("message", "Initializing")))
    detail = html.escape(str(data.get("detail", "")))
    failed = data.get("state") == "failed"
    diagnostic_hidden = "" if data.get("diagnostics_available") else "hidden"
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WAN Loop / {state}</title>
<style>
:root{{color-scheme:dark;font-family:Inter,system-ui,sans-serif}}
*{{box-sizing:border-box}}
body{{margin:0;background:#0d1117;color:#e6edf3;display:grid;min-height:100vh;place-items:center}}
main{{width:min(720px,88vw);background:#161b22;border:1px solid #30363d;border-radius:18px;padding:30px}}
h1{{margin:0 0 18px;font-size:26px}} .phase{{color:#8b949e;text-transform:uppercase;letter-spacing:.12em}}
.bar{{height:10px;background:#30363d;border-radius:99px;overflow:hidden;margin:24px 0}}
.bar span{{display:block;width:45%;height:100%;background:#2f81f7;border-radius:99px;animation:move 1.4s ease-in-out infinite alternate}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#b1bac4;font-size:15px;line-height:1.6}}
a.download{{display:inline-block;background:#2f81f7;color:white;padding:14px 18px;border-radius:10px;text-decoration:none}}
[hidden]{{display:none!important}} body.failed .bar{{display:none}}
@keyframes move{{to{{transform:translateX(122%)}}}}
</style></head><body class="{'failed' if failed else ''}"><main>
<div class="phase" id="phase">{phase}</div><h1 id="message">{message}</h1>
<div class="bar"><span></span></div><pre id="detail">{detail}</pre>
<p id="waiting" {'hidden' if failed else ''}>このページは自動更新されます。ComfyUIの準備が終わるまで閉じなくて大丈夫です。</p>
<p id="failed-help" {'' if failed else 'hidden'}>起動は停止しています。待つだけでは生成は始まりません。診断ファイルを保存してください。Podの停止・削除は自動で行いません。起動中のPodには料金が発生する場合があります。</p>
<a id="diagnostics" class="download" href="/diagnostics.json" download="wan-gpu-diagnostics.json" {diagnostic_hidden}>診断ファイルを保存</a>
<p id="diagnostics-note" {diagnostic_hidden}>GPU・Pod IDなどを含みます。公開せず、個別の調査に使用してください。</p>
</main><script>
async function poll(){{
  try{{
    const r=await fetch('/status.json?ts='+Date.now(),{{cache:'no-store'}});
    if(!r.ok) throw new Error('handoff');
    const s=await r.json();
    document.getElementById('phase').textContent=s.phase||'container';
    document.getElementById('message').textContent=s.message||'Initializing';
    let d=s.detail||'';
    if(s.assets_total) d+='\nAssets: '+(s.assets_completed||0)+' / '+s.assets_total;
    if(s.bytes_total) d+='\nVerified: '+((s.bytes_completed||0)/1e9).toFixed(2)+' / '+(s.bytes_total/1e9).toFixed(2)+' GB';
    document.getElementById('detail').textContent=d.trim();
    document.body.classList.toggle('failed',s.state==='failed');
    document.getElementById('waiting').hidden=s.state==='failed';
    document.getElementById('failed-help').hidden=s.state!=='failed';
    document.getElementById('diagnostics').hidden=!s.diagnostics_available;
    document.getElementById('diagnostics-note').hidden=!s.diagnostics_available;
    if(s.state==='handoff') setTimeout(()=>location.reload(),900);
  }}catch(e){{ setTimeout(()=>location.reload(),1200); return; }}
  setTimeout(poll,1000);
}} poll();
</script></body></html>"""


class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def read_diagnostics(path, status):
    """Serve only this boot's bounded JSON, never an arbitrary requested path."""
    if not path or not status.get("boot_id"):
        return None
    try:
        with pathlib.Path(path).open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return None
        report = json.loads(raw)
        if (not isinstance(report, dict) or report.get("schema_version") != 1
                or report.get("boot_id") != status["boot_id"]):
            return None
        return raw
    except (OSError, ValueError, TypeError):
        return None


def make_handler(path, diagnostics_path=None):
    path = pathlib.Path(path)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, content_type, payload, download=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="wan-gpu-diagnostics.json"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            route = self.path.split("?", 1)[0]
            data = read_status(path)
            diagnostics = read_diagnostics(diagnostics_path, data)
            data["diagnostics_available"] = diagnostics is not None
            if route == "/diagnostics.json":
                self._send(200 if diagnostics else 404, "application/json; charset=utf-8",
                           diagnostics or b'{"error":"No diagnostics for this boot"}', download=bool(diagnostics))
            elif route == "/status.json":
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", payload)
            elif route in {"/healthz", "/ping"}:
                payload = json.dumps(
                    {"state": data.get("state", "initializing")}
                ).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", payload)
            else:
                payload = render_page(data).encode("utf-8")
                self._send(200, "text/html; charset=utf-8", payload)

        def log_message(self, _format, *_args):
            return

    return Handler


def serve(path, host, port, diagnostics_path=None):
    server = ReusableServer((host, port), make_handler(path, diagnostics_path))
    print(f"BOOT STATUS: listening on http://{host}:{port}", flush=True)

    def stop(_signum, _frame):
        server.server_close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write")
    write.add_argument("--file", required=True)
    write.add_argument("--state")
    write.add_argument("--phase")
    write.add_argument("--message")
    write.add_argument("--detail")
    write.add_argument("--boot-id")
    write.add_argument("--reset", action="store_true")
    write.add_argument("--preserve-failure", action="store_true")
    write.add_argument("--assets-completed", type=int)
    write.add_argument("--assets-total", type=int)
    write.add_argument("--bytes-completed", type=int)
    write.add_argument("--bytes-total", type=int)

    server = subparsers.add_parser("serve")
    server.add_argument("--file", required=True)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8188)
    server.add_argument("--diagnostics-file")

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args.file, args.host, args.port, args.diagnostics_file)
        return 0

    write_status(
        args.file,
        {
            "state": args.state,
            "phase": args.phase,
            "message": args.message,
            "detail": args.detail,
            "boot_id": args.boot_id,
            "assets_completed": args.assets_completed,
            "assets_total": args.assets_total,
            "bytes_completed": args.bytes_completed,
            "bytes_total": args.bytes_total,
        },
        reset=args.reset,
        preserve_failure=args.preserve_failure,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
