#!/usr/bin/env bash
# Expose the local Vite dev server to your Tailnet over HTTPS with a real,
# browser-trusted certificate (Let's Encrypt via Tailscale).
#
# Why: opening the dev server by LAN IP over plain HTTP (or with a self-signed
# HTTPS cert) makes the browser treat the origin as insecure, so it refuses
# getUserMedia with "the site can't ask for your permission". Tailscale Serve
# terminates TLS with a cert your phone already trusts, and the page is only
# reachable from devices inside your tailnet (not the public internet).
#
# One-time setup (needs sudo; asks for your password):
#   sudo tailscale set --operator="$USER"
#   # Also enable HTTPS in the admin console:
#   #   https://login.tailscale.com/admin/dns  ->  "HTTPS Certificates" -> Enable
#
# Usage:
#   ./scripts/serve_tailscale.sh            # serve http://127.0.0.1:5173 on the tailnet
#   ./scripts/serve_tailscale.sh status     # show current serve config
#   ./scripts/serve_tailscale.sh off        # stop serving
#
# After it runs, open the printed https://<host>.<tailnet>.ts.net URL on your
# phone (Tailscale connected on the phone).

set -euo pipefail

PORT="${PORT:-5173}"
CMD="${1:-up}"

# Resolve this machine's tailnet DNS name, e.g. alfarizi.tail1b6523.ts.net
dns_name() {
  tailscale status --json 2>/dev/null \
    | node -e "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{const j=JSON.parse(d);process.stdout.write((j.Self&&j.Self.DNSName)||'')})" \
    | sed 's/\.$//'
}

case "$CMD" in
  up)
    HOST="$(dns_name)"
    if [[ -z "$HOST" ]]; then
      echo "ERROR: tidak bisa ambil DNS name tailnet. Pastikan 'tailscale status' jalan." >&2
      exit 1
    fi
    echo "Menyajikan http://127.0.0.1:${PORT} ke tailnet sebagai https://${HOST} ..."
    # --bg so it keeps running after this script exits.
    sudo tailscale serve --bg "http://127.0.0.1:${PORT}"
    echo
    echo "=================================================="
    echo "  Buka di HP (Tailscale harus connected):"
    echo "    https://${HOST}"
    echo "=================================================="
    ;;
  status)
    tailscale serve status
    ;;
  off)
    sudo tailscale serve reset
    echo "Serve dimatikan."
    ;;
  *)
    echo "Usage: $0 [up|status|off]" >&2
    exit 1
    ;;
esac
