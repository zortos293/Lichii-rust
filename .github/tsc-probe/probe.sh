#!/bin/bash
set -u
lscpu | grep -E 'Model name|^CPU\(s\)|Hypervisor|Flags' | cut -c1-400
cat /sys/devices/system/clocksource/clocksource0/current_clocksource
gcc -O2 -pthread -o /tmp/warp .github/tsc-probe/warp.c
for p in "0 1" "0 2" "0 3" "1 2" "1 3" "2 3"; do /tmp/warp $p; done
bin() {
  cargo test -q -p lichii-core --lib --no-run --message-format=json 2>/dev/null | python3 -c 'import sys,json
for l in sys.stdin:
  try: j=json.loads(l)
  except: continue
  if j.get("executable") and j.get("target",{}).get("name")=="lichii_core": print(j["executable"])' | tail -1
}
cp "$(bin)" /tmp/core.fixed
git fetch -q --depth=1 origin main
git show FETCH_HEAD:crates/core/src/timing.rs > crates/core/src/timing.rs
cp "$(bin)" /tmp/core.main
git checkout -q crates/core/src/timing.rs
for variant in main fixed; do
  fail=0
  for i in $(seq 1 1500); do
    out=$(/tmp/core.$variant 2>&1) || { fail=$((fail+1)); echo "$out" | grep -m1 'went back'; }
  done
  echo "PROBE $variant: $fail failures in 1500 runs"
done
