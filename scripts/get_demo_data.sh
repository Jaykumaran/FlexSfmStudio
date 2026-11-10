#!/usr/bin/env bash
# Download the demo dataset: OpenDroneMap "aukerman" - 77 drone photos (Sony DSC-WX220) with GPS in EXIF, public domain (CC0 1.0).
# Source: https://github.com/OpenDroneMap/odm_data_aukerman   (about 520 MB)
#   ./scripts/get_demo_data.sh [target folder]      default: ./demo_data/aukerman
set -euo pipefail
DST="${1:-demo_data/aukerman}"; mkdir -p "$DST/images"
BASE=https://raw.githubusercontent.com/OpenDroneMap/odm_data_aukerman/master
command -v curl >/dev/null || { echo "curl is needed"; exit 1; }
echo "==> licence"; curl -fsSL "$BASE/license.txt" -o "$DST/LICENSE.txt"; head -1 "$DST/LICENSE.txt"
echo "==> image list"
curl -fsSL https://api.github.com/repos/OpenDroneMap/odm_data_aukerman/contents/images \
  | grep -o '"name": *"[^"]*\.JPG"' | sed 's/.*"\([^"]*\.JPG\)"/\1/' > "$DST/images.txt"
N=$(wc -l < "$DST/images.txt"); echo "    $N images"
echo "==> downloading (resumes / skips files already there)"
cd "$DST/images"
xargs -P 6 -I{} sh -c '[ -s "{}" ] || curl -fsSL --retry 3 "'"$BASE"'/images/{}" -o "{}"' < ../images.txt
echo "==> done: $(ls | wc -l) images in $DST/images"
cat <<MSG

Try it:
  flexsfm new demo_project --source $DST/images --gps_source exif
  flexsfm all demo_project            # or open it in the app:  ./flexsfm.sh gui demo_project
MSG
