#!/usr/bin/env bash
#
# test_hmx_programming_guide.sh — 串跑 HMX 编程指南的所有 demo，用
# hexagon-sim 验证每个 demo 的 bit-exact PASS。
#
# 每个 demo 遵循约定:
#   - 成功时 stdout 含 "[PASS] demoNN"
#   - 失败时 stdout 含 "[FAIL]" 并用 h2_thread_stop(fail_count) 退出
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GUIDE_DIR="$ROOT_DIR/example/hmx_programming_guide"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

H2_INSTALL="$ROOT_DIR/tools/h2-install"

echo "=== Build all demos ==="
bash "$GUIDE_DIR/build.sh"

echo ""
echo "=== Run demos on hexagon-sim ==="

total=0
pass=0

for demo in "$GUIDE_DIR"/demo*; do
    # 只挑可执行 ELF（不要 .c / .sh）
    [ -x "$demo" ] && [ ! -d "$demo" ] || continue
    case "$demo" in
        *.c|*.sh) continue ;;
    esac
    name=$(basename "$demo")
    total=$((total + 1))
    LOG="$GUIDE_DIR/${name}.log"

    echo ""
    echo "--- $name ---"
    hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
        -- "$H2_INSTALL/bin/booter" \
           --ext_power 1 --use_ext 1 --fence_hi 0xfe000000 \
           "$demo" > "$LOG" 2>&1 || true

    if grep -q '\[PASS\]' "$LOG" && ! grep -q '\[FAIL\]' "$LOG"; then
        pass=$((pass + 1))
        grep -E '\[PASS\]|out\[' "$LOG" | sed 's/^/    /'
        echo "    PASS"
    else
        echo "    FAIL — see $LOG"
        grep -E '\[FAIL\]' "$LOG" | head -5 | sed 's/^/    /'
    fi
done

echo ""
echo "=== Summary ==="
echo "  $pass / $total PASS"

if [ "$pass" -eq "$total" ] && [ "$total" -gt 0 ]; then
    echo "PASS: HMX programming guide demos ($total/$total)"
    exit 0
fi
echo "FAIL: $((total - pass)) / $total demos failed" >&2
exit 1
