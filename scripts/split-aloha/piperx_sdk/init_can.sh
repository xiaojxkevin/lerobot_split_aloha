#!/usr/bin/env bash

set -euo pipefail

BITRATE=1000000
# USB bus-info assignments from Sealab's piper_sdk/can_muti_activate.sh.
declare -A USB_BY_TARGET=(
    [can_left_main]="1-6.4.4:1.0"
    [can_right_main]="1-6.4.3:1.0"
    [can_left_slave]="1-1:1.0"
    [can_right_slave]="1-2:1.0"
)
FOLLOWERS_ONLY=false
INTERFACES_ONLY=false

usage() {
    cat <<'EOF'
Usage:
  bash scripts/split-aloha/piperx_sdk/init_can.sh [--followers-only] [--interfaces-only]
      [--left-usb BUS_INFO] [--right-usb BUS_INFO]

Configure the four Sealab CAN interfaces at 1 Mbps by USB bus-info:
  can_left_main    1-6.4.4:1.0
  can_right_main   1-6.4.3:1.0
  can_left_slave   1-1:1.0
  can_right_slave  1-2:1.0

By default, also set both leader controllers to CAN control and verify their
feedback. Mode changes require all six motors on the affected arm to be disabled.
Use --interfaces-only to configure CAN interfaces without changing controller modes.
Use --followers-only for policy-only deployment. Use --left-usb or --right-usb
to override the corresponding follower bus-info from ethtool -i.
Changing CAN interface names and bitrate requires sudo. Mode setup uses the
repository's Python environment through uv.
EOF
}

while (($#)); do
    case "$1" in
        --followers-only)
            FOLLOWERS_ONLY=true
            shift
            ;;
        --interfaces-only)
            INTERFACES_ONLY=true
            shift
            ;;
        --left-usb|--right-usb)
            [[ -n "${2:-}" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
            side="${1#--}"
            USB_BY_TARGET["can_${side%-usb}_slave"]="$2"
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

targets=(can_left_slave can_right_slave)
if [[ "$FOLLOWERS_ONLY" == false ]]; then
    targets+=(can_left_main can_right_main)
fi

declare -A TARGET_BY_USB=()
for target in "${targets[@]}"; do
    bus_info="${USB_BY_TARGET[$target]}"
    if [[ -n "${TARGET_BY_USB[$bus_info]:-}" ]]; then
        echo "Duplicate USB bus-info '$bus_info' for '${TARGET_BY_USB[$bus_info]}' and '$target'." >&2
        exit 2
    fi
    TARGET_BY_USB[$bus_info]="$target"
done

for command in ip ethtool sudo; do
    command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 1; }
done

if [[ "$FOLLOWERS_ONLY" == false && "$INTERFACES_ONLY" == false ]]; then
    command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 1; }
fi

sudo modprobe gs_usb

read_bus_info() {
    sudo ethtool -i "$1" | awk '$1 == "bus-info:" {print $2}'
}

declare -A INTERFACE_BY_TARGET=()
interfaces="$(ip -br link show type can)"
while read -r interface _; do
    [[ -n "$interface" ]] || continue
    bus_info="$(read_bus_info "$interface" 2>/dev/null)" || continue
    [[ -n "$bus_info" ]] || continue
    target="${TARGET_BY_USB[$bus_info]:-}"
    [[ -n "$target" ]] || continue
    if [[ -n "${INTERFACE_BY_TARGET[$target]:-}" ]]; then
        echo "More than one CAN interface reports bus-info '$bus_info'." >&2
        exit 1
    fi
    INTERFACE_BY_TARGET[$target]="$interface"
done <<< "$interfaces"

for target in "${targets[@]}"; do
    source_interface="${INTERFACE_BY_TARGET[$target]:-}"
    [[ -n "$source_interface" ]] || {
        echo "No CAN interface found for '$target' (USB bus-info '${USB_BY_TARGET[$target]}')." >&2
        exit 1
    }
    if [[ "$source_interface" != "$target" ]] && ip link show dev "$target" >/dev/null 2>&1; then
        echo "Cannot rename '$source_interface' to '$target': name is already used." >&2
        exit 1
    fi
done

link_is_up() {
    local state
    state="$(ip -o link show dev "$1")" || return
    local flags="${state#*<}"
    flags="${flags%%>*}"
    [[ ",$flags," == *,UP,* ]]
}

link_bitrate() {
    ip -details link show dev "$1" | awk '$1 == "bitrate" {print $2}'
}

configure_interface() {
    local source_interface="$1"
    local target_interface="$2"
    if [[ "$source_interface" == "$target_interface" ]] &&
        link_is_up "$source_interface" && [[ "$(link_bitrate "$source_interface")" == "$BITRATE" ]]; then
        return
    fi
    sudo ip link set "$source_interface" down
    sudo ip link set "$source_interface" type can bitrate "$BITRATE"
    if [[ "$source_interface" != "$target_interface" ]]; then
        sudo ip link set "$source_interface" name "$target_interface"
    fi
    sudo ip link set "$target_interface" up
}

verify_interface() {
    local interface="$1"
    local expected_bus="$2"
    local bitrate bus_info
    link_is_up "$interface" || { echo "$interface is not UP after configuration." >&2; exit 1; }
    bitrate="$(link_bitrate "$interface")"
    [[ "$bitrate" == "$BITRATE" ]] || {
        echo "$interface bitrate is '${bitrate:-unknown}', expected '$BITRATE'." >&2
        exit 1
    }
    bus_info="$(read_bus_info "$interface")"
    [[ "$bus_info" == "$expected_bus" ]] || {
        echo "$interface bus-info is '${bus_info:-unknown}', expected '$expected_bus'." >&2
        exit 1
    }
}

for target in "${targets[@]}"; do
    configure_interface "${INTERFACE_BY_TARGET[$target]}" "$target"
    verify_interface "$target" "${USB_BY_TARGET[$target]}"
    echo "$target is UP at $BITRATE bit/s (USB bus-info ${USB_BY_TARGET[$target]})."
done

if [[ "$FOLLOWERS_ONLY" == false && "$INTERFACES_ONLY" == false ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
    uv run --no-sync --project "$repo_dir" python "$script_dir/set_leader_can_control.py" \
        can_left_main can_right_main
fi
