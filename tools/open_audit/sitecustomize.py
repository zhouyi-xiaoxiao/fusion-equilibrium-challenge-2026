# Optional file-read audit. Active only when FEC_OPEN_AUDIT=<directory> is set and this directory (tools/open_audit) is on
# PYTHONPATH: every path opened by Python code in this process (and in forked workers) is appended once to
# <directory>/opens_<pid>.tsv as "<path>\t<mode>". Used to list the generated data a training run reads (weights/GENERATED.sha256).
# Native reads by C libraries (e.g. Arrow parquet readers) are not seen; the training path reads no parquet file.
import os
import sys

_d = os.environ.get("FEC_OPEN_AUDIT")
if _d:
    os.makedirs(_d, exist_ok=True)
    _log = open(os.path.join(_d, f"opens_{os.getpid()}.tsv"), "a", buffering=1)
    _seen = set()

    def _hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            try:
                key = (os.fsdecode(args[0]), str(args[1]))
            except Exception:  # noqa: BLE001
                return
            if key not in _seen:
                _seen.add(key)
                _log.write(f"{key[0]}\t{key[1]}\n")

    sys.addaudithook(_hook)
