"""Plans a dti training run over a parquet dataset on Hugging Face.

    python data/plan.py HuggingFaceFW/fineweb-edu 'sample/10BT/*.parquet' fineweb-edu.json
    dti-coordinator --dataset fineweb-edu.json

Workers download their slices straight from Hugging Face, so the plan says
exactly what to fetch: one slice per parquet row group, the byte range of its
text column chunk, and how to decode it. The planner reads only each file's
footer (two range requests per file), never the text itself, and pins the
dataset to a commit so every worker sees the same bytes.

The coordinator decides what gets trained on. Each --filter (e.g.
`int_score>=3`, `language_score>0.95`) is evaluated here on that column, which
is small next to the text, and the plan records which rows of each slice
workers keep. Slices with no rows left are dropped. The first row group is
held out as validation text for the coordinator.

Needs pyarrow. Only public datasets work, since workers download without a
token, and text columns must be snappy-compressed or uncompressed (true of
FineWeb and of Hugging Face's own parquet conversions).
"""

import argparse
import fnmatch
import io
import json
import operator
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np
import pyarrow.parquet as pq

HF = "https://huggingface.co"
VAL_BYTES = 64 << 10
CODECS = {"SNAPPY": "SNAPPY", "UNCOMPRESSED": "UNCOMPRESSED"}
OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt}


class RemoteFile(io.RawIOBase):
    """A seekable, read-only remote file; every read is one range request."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def api(path):
    with urllib.request.urlopen(f"{HF}/api/{path}") as r:
        return json.load(r), r.headers


def list_files(repo, sha, pattern):
    prefix = pattern.split("*", 1)[0].rpartition("/")[0]
    url = f"{HF}/api/datasets/{repo}/tree/{sha}/{urllib.parse.quote(prefix)}?recursive=true"
    files = []
    while url:
        with urllib.request.urlopen(url) as r:
            files += json.load(r)
            next_link = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = next_link and next_link.group(1)
    return sorted(
        (f for f in files if f["type"] == "file" and fnmatch.fnmatchcase(f["path"], pattern)),
        key=lambda f: f["path"],
    )


def parse_filter(text):
    m = re.fullmatch(r"\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*", text)
    if not m:
        sys.exit(f"bad --filter {text!r}: expected e.g. 'int_score>=3'")
    column, op, value = m.groups()
    try:
        value = float(value)
    except ValueError:
        value = value.strip("'\"")
    return column, OPS[op], value


def keep_mask(pf, row_group, filters):
    """Rows of `row_group` that pass every filter (nulls fail)."""
    rows = pf.metadata.row_group(row_group).num_rows
    mask = np.ones(rows, dtype=bool)
    if filters:
        table = pf.read_row_group(row_group, columns=sorted({c for c, _, _ in filters}))
        for column, op, value in filters:
            col = table.column(column)
            valid = ~np.asarray(col.is_null())
            values = np.asarray(col.fill_null(value).to_numpy(zero_copy_only=False))
            mask &= valid & op(values, value)
    return mask


def text_chunk(pf, url, row_group, column):
    """Where `column`'s chunk of `row_group` is and how a worker decodes it."""
    schema = pf.schema
    index = next((i for i in range(len(schema)) if schema.column(i).path == column), None)
    if index is None:
        sys.exit(f"{url}: no column {column!r}")
    desc = schema.column(index)
    if desc.physical_type != "BYTE_ARRAY" or desc.max_repetition_level != 0 or desc.max_definition_level > 1:
        sys.exit(f"{url}: column {column!r} is not a flat string column")
    c = pf.metadata.row_group(row_group).column(index)
    if c.compression not in CODECS:
        sys.exit(f"{url}: column {column!r} uses {c.compression}; workers decode only SNAPPY or UNCOMPRESSED")
    start = c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset
    return {
        "kind": "parquet",
        "url": url,
        "offset": start,
        "len": c.total_compressed_size,
        "data_page_offset": c.data_page_offset - start,
        "num_values": c.num_values,
        "nullable": desc.max_definition_level == 1,
        "codec": CODECS[c.compression],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", help="dataset repo, e.g. HuggingFaceFW/fineweb-edu")
    ap.add_argument("files", help="glob of parquet files in the repo, e.g. 'sample/10BT/*.parquet'")
    ap.add_argument("out", help="plan file to write; validation text goes next to it")
    ap.add_argument("--revision", default="main", help="branch, tag or commit (pinned to a commit either way)")
    ap.add_argument("--column", default="text", help="the text column to train on")
    ap.add_argument("--filter", action="append", default=[], help="keep rows where e.g. 'int_score>=3'; repeatable")
    ap.add_argument("--max-files", type=int, help="plan only the first N files")
    ap.add_argument("--max-row-groups", type=int, help="plan only the first N row groups of each file")
    args = ap.parse_args()
    filters = [parse_filter(f) for f in args.filter]

    sha = api(f"datasets/{args.repo}/revision/{urllib.parse.quote(args.revision, safe='')}")[0]["sha"]
    files = list_files(args.repo, sha, args.files)[: args.max_files]
    if not files:
        sys.exit(f"no files in {args.repo}@{sha} match {args.files!r}")

    slices, val, kept_rows, total_rows = [], b"", 0, 0
    for n, f in enumerate(files):
        url = f"{HF}/datasets/{args.repo}/resolve/{sha}/{urllib.parse.quote(f['path'])}"
        pf = pq.ParquetFile(RemoteFile(url, f["size"]))
        groups = range(min(pf.metadata.num_row_groups, args.max_row_groups or sys.maxsize))
        print(f"[{n + 1}/{len(files)}] {f['path']}: {len(groups)} row groups", file=sys.stderr)
        for g in groups:
            mask = keep_mask(pf, g, filters)
            if n == 0 and g == 0:
                docs = pf.read_row_group(0, columns=[args.column]).column(0).to_pylist()
                val = "\n".join(d for d, k in zip(docs, mask) if k and d).encode()[:VAL_BYTES]
                continue
            total_rows += len(mask)
            kept_rows += int(mask.sum())
            if not mask.any():
                continue
            chunk = text_chunk(pf, url, g, args.column)
            chunk["keep"] = None if mask.all() else np.packbits(mask, bitorder="little").tolist()
            slices.append({"bytes": chunk["len"], "source": chunk})

    if not slices or len(val) < 2:
        sys.exit("nothing left to train on (or no validation text) after filtering")
    stem = os.path.splitext(args.out)[0]
    val_path = f"{stem}.val.txt"
    with open(val_path, "wb") as fh:
        fh.write(val)
    source = f"{args.repo}@{sha[:12]} {args.files}" + "".join(f" [{f}]" for f in args.filter)
    with open(args.out, "w") as fh:
        json.dump({"source": source, "val": os.path.basename(val_path), "slices": slices}, fh)
    gb = sum(s["bytes"] for s in slices) / 1e9
    print(
        f"wrote {args.out}: {len(slices)} slices, {gb:.2f} GB to download, "
        f"{kept_rows}/{total_rows} documents kept; validation text in {val_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
