#!/usr/bin/env bash
# Assemble the AMP humanoid code used in the TOPPO paper:
#   MimicKit @ 2ed1e6c  +  patches/0001-...patch  +  overlay/  +  the DeepMimic walk clip.
#
# Usage:   bash setup.sh [TARGET_DIR]        (default: ./mimickit_toppo next to this file)
# Needs:   git, curl, tar, and a Python environment with requirements.txt installed
#          (torch and numpy are used to convert the motion clip).
# Env:     PYTHON (default: python), MIMICKIT_URL (default: the GitHub repository).
#
# Every file this script writes is checked against MANIFEST.sha256 at the end.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$HERE/mimickit_toppo}"
PYTHON="${PYTHON:-python}"
MIMICKIT_URL="${MIMICKIT_URL:-https://github.com/xbpeng/MimicKit}"
MIMICKIT_COMMIT="2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69"
DEEPMIMIC_COMMIT="1f915c52fcd4b95b5f5f15b759ae91bd81e9a801"
WALK_URL="https://raw.githubusercontent.com/xbpeng/DeepMimic/${DEEPMIMIC_COMMIT}/data/motions/humanoid3d_walk.txt"

if [ -e "$TARGET" ]; then
    echo "error: $TARGET already exists; remove it or pass another directory" >&2
    exit 1
fi

echo "[1/5] cloning MimicKit and checking out ${MIMICKIT_COMMIT:0:7}"
git clone --quiet "$MIMICKIT_URL" "$TARGET"
git -C "$TARGET" -c advice.detachedHead=false checkout --quiet "$MIMICKIT_COMMIT"

echo "[2/5] applying the TOPPO patch (8 modified upstream files)"
git -C "$TARGET" apply "$HERE/patches/0001-toppo-modifications-to-mimickit-2ed1e6c.patch"

echo "[3/5] copying overlay/ (new modules, configs, tools, tests)"
# tar rather than cp -R, so that Python caches left in overlay/ (e.g. by running its tests
# in place) are not copied into the tree.
( cd "$HERE/overlay" && tar --exclude=__pycache__ --exclude='*.pyc' --exclude=.pytest_cache -cf - . ) \
    | ( cd "$TARGET" && tar -xf - )

echo "[4/5] fetching the DeepMimic walk clip (DeepMimic @ ${DEEPMIMIC_COMMIT:0:7}, MIT)"
DM_DIR="$TARGET/external/deepmimic"
mkdir -p "$DM_DIR/data/motions"
cp "$HERE/third_party/deepmimic/LICENSE" "$HERE/third_party/deepmimic/README.md" \
   "$HERE/third_party/deepmimic/SHA256SUMS" "$DM_DIR/"
curl -fsSL "$WALK_URL" -o "$DM_DIR/data/motions/humanoid3d_walk.txt"

sha256_of() {
    "$PYTHON" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
}
expect() {  # expect <file> <name in third_party/deepmimic/SHA256SUMS>
    local want got
    want="$(awk -v n="$2" '$2 == n {print $1}' "$HERE/third_party/deepmimic/SHA256SUMS")"
    got="$(sha256_of "$1")"
    if [ "$want" != "$got" ]; then
        echo "error: sha256($1) = $got, expected $want" >&2
        exit 1
    fi
}
expect "$DM_DIR/data/motions/humanoid3d_walk.txt" humanoid3d_walk.txt

echo "[5/5] converting the clip to MimicKit format"
( cd "$TARGET" && PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    tools/deepmimic_to_mimickit/deepmimic_to_mimickit.py \
    --input_file external/deepmimic/data/motions/humanoid3d_walk.txt \
    --output_file data/motions/humanoid/deepmimic_humanoid_walk.pkl )
expect "$TARGET/data/motions/humanoid/deepmimic_humanoid_walk.pkl" deepmimic_humanoid_walk.pkl

echo "verifying the assembled tree against MANIFEST.sha256"
"$PYTHON" - "$HERE/MANIFEST.sha256" "$TARGET" <<'PY'
import hashlib, os, sys
manifest, root = sys.argv[1], sys.argv[2]
bad = 0
n = 0
for line in open(manifest, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    want, rel = line.split(None, 1)
    path = os.path.join(root, rel)
    n += 1
    if not os.path.isfile(path):
        print("MISSING  " + rel); bad += 1; continue
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != want:
        print("CHANGED  " + rel); bad += 1
if bad:
    sys.exit("%d of %d files do not match MANIFEST.sha256" % (bad, n))
print("all %d files match MANIFEST.sha256" % n)
PY

echo "done: $TARGET"
