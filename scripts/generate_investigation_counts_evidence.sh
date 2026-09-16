#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CASE="$ROOT/examples/investigation-counts"
B=${BAYESITE_BIN:-"$ROOT/target/release/bayesite"}

if [ "${BAYESITE_BIN+x}" != x ]; then
  (cd "$ROOT" && cargo build --release --locked --bin bayesite)
elif [ ! -x "$B" ]; then
  echo "BAYESITE_BIN is not executable: $B" >&2
  exit 1
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/bayesite-count-evidence.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

"$B" inspect --model "$CASE/poisson.json" --data "$CASE/data.json" > "$TMP/inspection.json"
"$B" sample --model "$CASE/poisson.json" --data "$CASE/data.json" \
  --out "$TMP/fit.jsonl" --chains 4 --warmup 250 --draws 250 \
  --max-treedepth 8 --target-accept 0.85 --seed 20260916 > /dev/null
"$B" diagnose --fit "$TMP/fit.jsonl" --out "$TMP/diagnostics.json" > /dev/null
"$B" posterior-check --model "$CASE/poisson.json" --data "$CASE/data.json" \
  --fit "$TMP/fit.jsonl" --seed 20260917 --out "$TMP/check.json" > /dev/null
"$B" capabilities > "$TMP/capabilities.json"

if command -v shasum >/dev/null 2>&1; then
  ENGINE_SHA=$(shasum -a 256 "$B" | awk '{print $1}')
else
  ENGINE_SHA=$(sha256sum "$B" | awk '{print $1}')
fi
ENGINE_BYTES=$(wc -c < "$B" | tr -d ' ')
TARGET=$(rustc -vV | awk '/^host:/ {print $2}')
CAPABILITIES=$(tr -d '\n' < "$TMP/capabilities.json")
printf '{"engine_record":"v0-provisional","binary":{"sha256":"%s","bytes":%s},"target":"%s","profile":"release","capabilities":%s}\n' \
  "$ENGINE_SHA" "$ENGINE_BYTES" "$TARGET" "$CAPABILITIES" > "$TMP/engine.json"

mkdir -p "$CASE/evidence"
for name in inspection.json fit.jsonl diagnostics.json check.json engine.json; do
  mv "$TMP/$name" "$CASE/evidence/$name"
done
printf 'Regenerated %s/evidence with %s\n' "$CASE" "$B"
