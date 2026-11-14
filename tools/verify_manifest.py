#!/usr/bin/env python3
"""Verify (or copy and verify) the files of a manifest.  Manifest line: <bytes> <sha256> <path relative to the root>.

  verify_manifest.py <manifest> <root>                 check sizes and sha256 of <root>/<path>
  verify_manifest.py <manifest> <root> --from <dir>    copy <dir>/<path> to <root>/<path> first (missing files only)
Exit code 0 only if every file is present with the recorded size and sha256."""
import hashlib, os, shutil, sys

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def main():
    man, root = sys.argv[1], sys.argv[2]
    src = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
    bad = 0; n = 0; tot = 0
    for line in open(man):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        size, digest, rel = line.split(None, 2)
        dst = os.path.join(root, rel)
        if src and not os.path.exists(dst):
            if not os.path.isfile(os.path.join(src, rel)):
                print("MISSING", rel, "(not in the source)"); bad += 1; continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(src, rel), dst)
        if not os.path.isfile(dst):
            print("MISSING", rel); bad += 1; continue
        if os.path.getsize(dst) != int(size) or sha256(dst) != digest:
            print("MISMATCH", rel); bad += 1; continue
        n += 1; tot += int(size)
    print(f"manifest {os.path.basename(man)}: {n} files OK ({tot} bytes), {bad} bad")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
