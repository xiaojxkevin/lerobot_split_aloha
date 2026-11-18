#!/usr/bin/env bash

# Shared, read-only preflight for policy-only dual-follower deployment scripts.
# Set SKIP_FOLLOWER_CAN_CHECK=true only for offline tests; real deployments must
# run scripts/split-aloha/piper_sdk/init_can.sh first and leave this check enabled.

require_follower_can_interfaces() {
    if [[ "${SKIP_FOLLOWER_CAN_CHECK:-false}" == "true" ]]; then
        echo "[warn] skipping follower CAN preflight (SKIP_FOLLOWER_CAN_CHECK=true)" >&2
        return 0
    fi
    command -v ip >/dev/null || {
        echo "Missing 'ip'; install iproute2 and run scripts/split-aloha/piper_sdk/init_can.sh first." >&2
        return 1
    }
    local iface state
    for iface in can_left_slave can_right_slave; do
        state="$(ip -o link show dev "$iface" 2>/dev/null)" || {
            echo "Missing CAN interface '$iface'. Run scripts/split-aloha/piper_sdk/init_can.sh before inference." >&2
            return 1
        }
        if [[ "$state" != *"<"*"UP"*">"* ]]; then
            echo "CAN interface '$iface' is not UP. Run scripts/split-aloha/piper_sdk/init_can.sh before inference." >&2
            return 1
        fi
    done
}
