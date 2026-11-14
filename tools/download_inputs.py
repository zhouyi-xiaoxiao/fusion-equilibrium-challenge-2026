#!/usr/bin/env python3
"""Download one configuration of a challenge dataset (one parquet file per shot) from the Hugging Face Hub.

  download_inputs.py --repo Sophelio/fusion-equilibrium-challenge-final --config diii_d_private_test --out <dir> [--workers 16]

Files are taken from the repository folder data/<config>/ (override with --folder); if that folder holds no parquet file, the
data_files pattern of the configuration in the dataset card (README.md, `configs:`) is used. Every file is checked for the parquet magic
and, where the Hub reports it, for its size. A manifest (name, bytes, sha256) is written to <out>.manifest.tsv.
A token is only needed for a gated dataset: it is read from the HF_TOKEN environment variable and never printed."""
import argparse, concurrent.futures as cf, fnmatch, hashlib, os, sys
from pathlib import Path

def card_patterns(api, repo, rev, config):
    """data_files patterns of one configuration in the dataset card (YAML front matter of README.md, key `configs`)."""
    import yaml
    from huggingface_hub import hf_hub_download
    text = Path(hf_hub_download(repo, "README.md", repo_type="dataset", revision=rev, token=os.environ.get("HF_TOKEN") or None)).read_text()
    assert text.startswith("---"), "dataset card without YAML front matter"
    meta = yaml.safe_load(text.split("---", 2)[1]) or {}
    for c in meta.get("configs", []):
        if c.get("config_name") == config:
            df = c.get("data_files", [])
            df = [df] if isinstance(df, (str, dict)) else df
            return [p for d in df for p in ([d] if isinstance(d, str) else [d["path"]] if isinstance(d.get("path"), str) else d.get("path", []))]
    raise SystemExit(f"configuration {config} not in the dataset card of {repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--folder", default=None); ap.add_argument("--revision", default="main"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    folder = a.folder or f"data/{a.config}"
    info = api.dataset_info(a.repo, revision=a.revision)
    print("dataset", a.repo, "revision", info.sha, flush=True)
    try:
        items = [e for e in api.list_repo_tree(a.repo, path_in_repo=folder, repo_type="dataset", revision=info.sha, recursive=False)
                 if getattr(e, "path", "").endswith(".parquet")]
    except Exception as ex:  # folder missing on the Hub
        print(f"no folder {folder}: {type(ex).__name__}", flush=True); items = []
    if not items and a.folder is None:
        pats = card_patterns(api, a.repo, info.sha, a.config)
        print(f"data/{a.config}/ holds no parquet file; dataset card patterns of {a.config}: {pats}", flush=True)
        items = [e for e in api.list_repo_tree(a.repo, repo_type="dataset", revision=info.sha, recursive=True)
                 if getattr(e, "path", "").endswith(".parquet") and any(fnmatch.fnmatchcase(e.path, q) for q in pats)]
        folder = ",".join(pats)
    assert items, f"no parquet files under {folder} (try --folder)"
    names = [Path(e.path).name for e in items]
    assert len(set(names)) == len(names), "two files with the same name in different folders (pass --folder)"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def one(e):
        dst = out / Path(e.path).name
        if not (dst.exists() and (e.size is None or dst.stat().st_size == e.size)):
            p = hf_hub_download(a.repo, e.path, repo_type="dataset", revision=info.sha, local_dir=str(out / ".dl"), token=os.environ.get("HF_TOKEN") or None)
            os.replace(p, dst)
        with open(dst, "rb") as f:
            h = hashlib.sha256(); b = f.read()
            assert b[-4:] == b"PAR1", f"not a parquet file: {dst}"
            h.update(b)
        assert e.size is None or len(b) == e.size, f"size mismatch {dst}"
        return dst.name, len(b), h.hexdigest()

    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = sorted(ex.map(one, items))
    Path(str(out) + ".manifest.tsv").write_text(f"# {a.repo} {folder} revision {info.sha}\n" + "".join(f"{n}\t{s}\t{d}\n" for n, s, d in rows))
    print(f"DOWNLOAD_OK {len(rows)} files, {sum(r[1] for r in rows)} bytes -> {out}")

if __name__ == "__main__":
    sys.exit(main())
