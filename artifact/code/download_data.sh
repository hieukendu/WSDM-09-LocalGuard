#!/bin/bash
# Downloads the public data used in the paper. The UMass host is slow, so large
# files are fetched with parallel HTTP range requests and then concatenated.
set -e
RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"
B=https://ciir.cs.umass.edu/downloads/LaMP
ranged() { # url out nparts
  local url=$1 out=$2 n=$3
  local size=$(curl -sI "$url" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
  local chunk=$(( (size + n - 1) / n ))
  for i in $(seq 0 $((n-1))); do
    local s=$((i*chunk)); local e=$((s+chunk-1)); [ $e -ge $size ] && e=$((size-1))
    curl -s --retry 8 --retry-all-errors -r $s-$e -o "$out.part$i" "$url" &
  done
  wait
  for i in $(seq 0 $((n-1))); do cat "$out.part$i"; done > "$out"; rm -f "$out".part*
  [ "$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out")" = "$size" ] || { echo "size mismatch $out"; exit 1; }
}
mkdir -p LaMP_3 LaMP_7 LaMP_QA
ranged $B/LaMP_7/dev/dev_questions.json LaMP_7/dev_questions.json 4
curl -s -o LaMP_7/dev_outputs.json $B/LaMP_7/dev/dev_outputs.json
curl -s -o LaMP_3/dev_outputs.json $B/LaMP_3/dev/dev_outputs.json
curl -sL -o LaMP_QA/lifestyle_validation.json https://huggingface.co/datasets/alireza7/LaMP-QA/resolve/1db3d5fcd986c11161797b934c2f34020ac0f0cf/data/Lifestyle_and_Personal_Development/validation/validation.json
ranged $B/LaMP_3/dev/dev_questions.json LaMP_3/dev_questions.json 16
shasum -a 256 LaMP_*/*.json > SHA256SUMS
echo done > dl.done
