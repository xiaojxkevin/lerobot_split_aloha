#!/usr/bin/env bash

# Configure exactly the two PiPER follower CAN adapters used by policy-only
# deployment. This intentionally does not reuse the legacy four-arm mappings in
# can_muti_activate.sh / can_config.sh.
set -euo pipefail

BITRATE=1000000
# Preserve the established four-arm workstation's follower assignments. The
# leader adapters may be absent; these two bus locations remain sufficient.
LEFT_USB="1-13:1.0"
RIGHT_USB="1-12:1.0"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/split-aloha/piper_sdk/init_can.sh [--left-usb BUS_INFO] [--right-usb BUS_INFO]

BUS_INFO is the value printed by `ethtool -i <can-interface>` (for example
`1-13:1.0`). The two supplied adapters are configured as follows:
  --left-usb   -> can_left_slave (default: 1-13:1.0)
  --right-usb  -> can_right_slave (default: 1-12:1.0)

This command changes CAN interface names and bitrate and therefore requires sudo.
EOF
}

while (($#)); do
    case "$1" in
        --left-usb)
            (($# >= 2)) || { echo "Missing value for --left-usb" >&2; exit 2; }
            LEFT_USB="$2"
            shift 2
            ;;
        --right-usb)
            (($# >= 2)) || { echo "Missing value for --right-usb" >&2; exit 2; }
            RIGHT_USB="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$LEFT_USB" != "$RIGHT_USB" ]] || { echo "left and right USB bus-info must differ." >&2; exit 2; }

for command in ip ethtool sudo; do
    command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 1; }
done

sudo modprobe gs_usb

declare -A BUS_TO_INTERFACE=()
for interface in $(ip -br link show type can | awk '{print $1}'); do
    bus_info="$(sudo ethtool -i "$interface" | awk '$1 == "bus-info:" {print $2}')"
    [[ -n "$bus_info" ]] || { echo "Unable to read bus-info for CAN interface '$interface'." >&2; exit 1; }
    if [[ "$bus_info" == "$LEFT_USB" || "$bus_info" == "$RIGHT_USB" ]]; then
        if [[ -n "${BUS_TO_INTERFACE[$bus_info]:-}" ]]; then
            echo "More than one CAN interface reports bus-info '$bus_info'." >&2
            exit 1
        fi
        BUS_TO_INTERFACE[$bus_info]="$interface"
    fi
done

for bus_info in "$LEFT_USB" "$RIGHT_USB"; do
    [[ -n "${BUS_TO_INTERFACE[$bus_info]:-}" ]] || {
        echo "No CAN interface found for USB bus-info '$bus_info'." >&2
        exit 1
    }
done

configure_interface() {
    local source_interface="$1"
    local target_interface="$2"
    local conflicting

    if [[ "$source_interface" != "$target_interface" ]] && ip link show dev "$target_interface" >/dev/null 2>&1; then
        conflicting="$(sudo ethtool -i "$target_interface" 2>/dev/null | awk '$1 == "bus-info:" {print $2}')"
        echo "Cannot rename '$source_interface' to '$target_interface': name is already used" \
            "(bus-info ${conflicting:-unknown})." >&2
        exit 1
    fi

    sudo ip link set "$source_interface" down
    sudo ip link set "$source_interface" type can bitrate "$BITRATE"
    if [[ "$source_interface" != "$target_interface" ]]; then
        sudo ip link set "$source_interface" name "$target_interface"
    fi
    sudo ip link set "$target_interface" up
}

configure_interface "${BUS_TO_INTERFACE[$LEFT_USB]}" can_left_slave
configure_interface "${BUS_TO_INTERFACE[$RIGHT_USB]}" can_right_slave

verify_interface() {
    local interface="$1"
    local expected_bus="$2"
    local state bitrate bus_info
    state="$(ip -o link show dev "$interface")"
    [[ "$state" == *"<"*"UP"*">"* ]] || { echo "$interface is not UP after configuration." >&2; exit 1; }
    bitrate="$(ip -details link show "$interface" | awk '/bitrate/ {for (i = 1; i <= NF; i++) if ($i == "bitrate") {print $(i + 1); exit}}')"
    [[ "$bitrate" == "$BITRATE" ]] || {
        echo "$interface bitrate is '${bitrate:-unknown}', expected '$BITRATE'." >&2
        exit 1
    }
    bus_info="$(sudo ethtool -i "$interface" | awk '$1 == "bus-info:" {print $2}')"
    [[ "$bus_info" == "$expected_bus" ]] || {
        echo "$interface bus-info is '${bus_info:-unknown}', expected '$expected_bus'." >&2
        exit 1
    }
}

verify_interface can_left_slave "$LEFT_USB"
verify_interface can_right_slave "$RIGHT_USB"
echo "Configured policy-only follower CAN interfaces: can_left_slave, can_right_slave (bitrate ${BITRATE})."
