#!/bin/bash
# DeepSeek-V4-Flash TP4 wedge watchdog (run on the HEAD node).
#
# Catches the HANG that --restart cannot: container stays "Up" but the engine
# core is stuck (API /health unresponsive). A crash is already handled by
# docker --restart; this handles the wedge.
#
# DEFAULT = ALERT-ONLY (logs + writes a flag file; does NOT restart). This is
# deliberate — GB10 has no safe auto-restart precedent yet (see memory
# reference_watchdog_verdict_memory_fencing). Enable auto-restart only once
# trusted, with:  AUTORESTART=1 ds_watchdog.sh
#
# Wedge criteria (all must hold, to avoid false-positives on big prefills):
#   * GET /health fails (short timeout) for FAILS_TO_TRIP consecutive probes
#     — /health returns fast regardless of inference load, so a slow 500K
#       prefill does NOT trip it; only a stuck engine core does.
#   * the container is still "Up" (a crashed/exited one is docker's job).
# Outside a BOOT_GRACE window after any (re)start.
set -uo pipefail

HEALTH="http://localhost:8888/health"
INTERVAL="${INTERVAL:-60}"          # probe cadence (s)
TIMEOUT="${TIMEOUT:-10}"            # per-probe timeout (s)
FAILS_TO_TRIP="${FAILS_TO_TRIP:-5}" # consecutive fails => wedge (~5 min)
BOOT_GRACE="${BOOT_GRACE:-420}"     # ignore first ~7 min after a (re)start
COOLDOWN="${COOLDOWN:-1800}"        # min seconds between auto-restarts
AUTORESTART="${AUTORESTART:-0}"     # 0 = alert-only (default), 1 = auto-restart
LOG="${LOG:-$HOME/dsv4f-watchdog.log}"
FLAG="$HOME/dsv4f-WEDGED.flag"
NODES=(1 2 3 4)                     # ${FABRIC_PREFIX:-10.0.0}.$n

log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG" ; }

restart_fleet(){
  log "RESTART: graceful fleet restart begin"
  docker stop -t 60 dsv4f-tp4 >/dev/null 2>&1 &
  for n in 2 3 4; do ssh -o BatchMode=yes -o ConnectTimeout=6 ${FABRIC_PREFIX:-10.0.0}.$n "docker stop -t 60 dsv4f-tp4 >/dev/null 2>&1" & done
  wait
  ssh -o BatchMode=yes ${FABRIC_PREFIX:-10.0.0}.2 "bash ~/tp4-launch.sh 1" >/dev/null 2>&1
  ssh -o BatchMode=yes ${FABRIC_PREFIX:-10.0.0}.3 "bash ~/tp4-launch.sh 2" >/dev/null 2>&1
  ssh -o BatchMode=yes ${FABRIC_PREFIX:-10.0.0}.4 "bash ~/tp4-launch.sh 3" >/dev/null 2>&1
  bash ~/tp4-launch.sh 0 >/dev/null 2>&1
  log "RESTART: relaunched all 4 ranks"
}

log "watchdog start (AUTORESTART=$AUTORESTART, trip=$FAILS_TO_TRIP x ${INTERVAL}s)"
fails=0
last_restart=0
start_ref=$(date +%s)   # treat script start as a boot grace anchor

while true; do
  now=$(date +%s)
  # boot grace: skip probing right after a (re)start
  if [ $((now - start_ref)) -lt "$BOOT_GRACE" ]; then sleep "$INTERVAL"; continue; fi

  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$HEALTH" 2>/dev/null)
  if [ "$code" = "200" ]; then
    if [ -f "$FLAG" ]; then log "RECOVERED: /health 200 again"; rm -f "$FLAG"; fi
    fails=0
    sleep "$INTERVAL"; continue
  fi

  fails=$((fails + 1))
  status=$(docker ps --filter name=dsv4f-tp4 --format '{{.Status}}' 2>/dev/null)
  log "PROBE-FAIL $fails/$FAILS_TO_TRIP (http=$code, container='$status')"

  if [ "$fails" -ge "$FAILS_TO_TRIP" ]; then
    case "$status" in
      *Up*)
        log "WEDGE DETECTED: /health down ${FAILS_TO_TRIP}x but container Up (engine hung)"
        touch "$FLAG"
        if [ "$AUTORESTART" = "1" ] && [ $((now - last_restart)) -ge "$COOLDOWN" ]; then
          restart_fleet; last_restart=$(date +%s); start_ref=$(date +%s); fails=0
        else
          log "ALERT-ONLY (or cooldown): NOT restarting. Human action or set AUTORESTART=1."
          # keep flag set; re-alert every 5 probes so it stays visible
          fails=0
        fi
        ;;
      *) log "container not Up ('$status') — docker --restart owns this; standing by"; fails=0 ;;
    esac
  fi
  sleep "$INTERVAL"
done
