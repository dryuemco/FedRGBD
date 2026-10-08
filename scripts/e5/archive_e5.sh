#!/bin/bash
# Move (never delete) the e5 captures s03/s09/s10 of node_b (frames, capture records,
# s09 locks) under data/raw/camera_pilot/_archive_<stamp>/, md5-checked.  s01 (pilot),
# _exposure/ and session_log.jsonl stay where they are.  Usage: archive_e5.sh <pilot_root> <stamp> [extra dirs...]
set -eu
cd "$1"; STAMP=$2; shift 2
A="_archive_$STAMP"
[ -e "$A" ] && { echo "ABORT: $A exists"; exit 2; }
LIST=$(mktemp)
find node_b -path node_b/_exposure -prune -o -type f \( -name 's03_*' -o -name 's09_*' -o -name 's10_*' \) -print | LC_ALL=C sort > "$LIST"
N=$(wc -l < "$LIST")
mkdir -p "$A"
xargs md5sum < "$LIST" > "$A/BEFORE.md5"
while read -r f; do mkdir -p "$A/$(dirname "$f")"; mv "$f" "$A/$f"; done < "$LIST"
for d in "$@"; do [ -d "$d" ] && mv "$d" "$A/$d" && echo "moved dir $d"; done
(cd "$A" && md5sum -c --quiet BEFORE.md5) && echo "md5 OK: $N files moved to $A"
LEFT=$(find node_b -path node_b/_exposure -prune -o -type f \( -name 's03_*' -o -name 's09_*' -o -name 's10_*' \) -print | wc -l)
echo "e5 files left outside the archive: $LEFT"
echo "s01 files in place: $(find node_b -type f -name 's01_*' | wc -l)"
rm -f "$LIST"
