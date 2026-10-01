"""Build the agreed corpus expansion. No training or original-project writes.

Run with Python 3.12 and numpy, tokenizers (the supplied baseline venv works).
Downloads use verified HTTPS, revision-checked HF Dataset Viewer responses.
All sampling, filtering, token accounting, and packing are deterministic.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import functools
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
import time
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINE = Path(r"C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2")
WORK = ROOT.parents[1] / "work" / "corpus_20m_v1"
SOURCES = [
    dict(id="fineweb_edu", dataset="HuggingFaceFW/fineweb-edu", config="sample-10BT",
         train=10_000_000, validation=500_000, license="odc-by"),
    dict(id="wikipedia_en", dataset="wikimedia/wikipedia", config="20231101.en",
         train=3_000_000, validation=150_000, license="cc-by-sa-3.0 AND gfdl"),
    dict(id="cosmopedia_v2", dataset="HuggingFaceTB/smollm-corpus", config="cosmopedia-v2",
         train=2_000_000, validation=100_000, license="odc-by (repository metadata)"),
]
POLICY = dict(version=1, min_words=80, max_characters=120_000,
              min_alpha_fraction=0.60, max_duplicate_line_fraction=0.25,
              min_unique_word_fraction=0.15, shingle_words=5, jaccard_threshold=0.80,
              minhash_permutations=128, lsh_bands=32, lsh_rows=4,
              eval_exact_ngram_words=13, sampling_seed=20260919,
              validation_probability=0.05, fineweb_min_int_score=3,
              fineweb_min_language_score=0.9, wikipedia_max_disallowed_title_fraction=0,
              note="Heuristic screening, not human review or proof of zero semantic overlap.")
WORDS = re.compile(r"\w+", re.UNICODE)
UNITS = re.compile(r"\s+|[\w\u200c\u200d]+|[^\w\s]+", re.UNICODE)
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def log(event, **kw):
    print(json.dumps(dict(event=event, **kw), ensure_ascii=True), flush=True)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def gz_writer(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0)


def request(url, raw=False):
    global NEXT_REQUEST
    for attempt in range(10):
        try:
            # Stay below anonymous Viewer quotas; parallel workers must share pacing.
            with REQUEST_LOCK:
                delay = max(0.0, NEXT_REQUEST - time.monotonic())
                while delay > 0:
                    time.sleep(min(delay, 30))
                    delay = max(0.0, NEXT_REQUEST - time.monotonic())
                NEXT_REQUEST = time.monotonic() + 3.5
            req = urllib.request.Request(url, headers={"User-Agent": "Corpus20MResearch/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                body = r.read()
                return (body if raw else json.loads(body)), dict(r.headers)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == 9:
                raise
            log("network_retry", attempt=attempt + 1, error=str(e)[:180])
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                with REQUEST_LOCK:
                    NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic() + 60)
            else:
                time.sleep(min(2 ** attempt, 15))


def viewer(endpoint, **params):
    return request("https://datasets-server.huggingface.co/" + endpoint + "?" + urllib.parse.urlencode(params))


def snapshot_baseline():
    inventory_path = ROOT / "reports/baseline_snapshot.json"
    if inventory_path.exists():
        inv = json.loads(inventory_path.read_text())
        for e in inv["files"]:
            if digest_file(ROOT / e["path"]) != e["sha256"]:
                raise RuntimeError("Snapshot hash mismatch: " + e["path"])
        return inv
    files = []
    for dirname in ["data/frozen_corpus_v1", "data/tokenized_v2", "artifacts/tokenizer_v2"]:
        for src in sorted((BASELINE / dirname).rglob("*")):
            if not src.is_file():
                continue
            rel = Path("baseline") / src.relative_to(BASELINE)
            dst = ROOT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            before = digest_file(src)
            shutil.copy2(src, dst)
            assert digest_file(dst) == before
            files.append(dict(path=rel.as_posix(), sha256=before, bytes=dst.stat().st_size))
    for name in ["canonical.py", "tokenizer.py", "tokenizer_candidates.py", "__init__.py"]:
        src = BASELINE / "src/era6" / name
        dst = ROOT / "vendor/era6" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files.append(dict(path=dst.relative_to(ROOT).as_posix(), sha256=digest_file(dst), bytes=dst.stat().st_size))
    inv = dict(original=str(BASELINE), files=files, original_modified=False)
    write_json(inventory_path, inv)
    return inv


class FastTokenizer:
    """Cache identical pretoken merges; preserve the frozen tokenizer's IDs."""
    def __init__(self):
        sys.path.insert(0, str(ROOT / "vendor"))
        from era6.tokenizer_candidates import load_lossless_tokenizer
        path = ROOT / "baseline/artifacts/tokenizer_v2/tokenizer.json"
        self.reference = load_lossless_tokenizer(path)
        self.specials = self.reference.special_token_ids
        self.pieces = self.reference._payloads
        self.hash = self.reference.tokenizer_hash
        assert self.reference._initialization == "bytes"
        assert not self.reference._initial_symbol_ids
        self.pattern = re.compile("|".join(re.escape(x) for x in sorted(self.specials, key=lambda x: (-len(x), x))))
        self.unit = functools.lru_cache(maxsize=200_000)(self._unit)

    def _unit(self, s):
        ids = self.reference._byte_ids
        symbols = [(ids[b], i, i + 1) for i, b in enumerate(s.encode())]
        return tuple(x[0] for x in self.reference._apply_merges(symbols))

    def encode(self, text):
        out = [self.specials["<bos>"]]
        pos = 0
        for match in self.pattern.finditer(text):
            for m in UNITS.finditer(text, pos, match.start()):
                out.extend(self.unit(m.group()))
            out.append(self.specials[match.group()])
            pos = match.end()
        for m in UNITS.finditer(text, pos):
            out.extend(self.unit(m.group()))
        out.append(self.specials["<eos>"])
        return np.asarray(out, dtype="<u2")

    def verify(self, text, ids, parity=False):
        decoded = b"".join(self.pieces[int(i)] for i in ids[1:-1]).decode()
        assert decoded == text, "Tokenizer round trip failed"
        if parity:
            expected = self.reference.encode(text, add_bos=True, add_eos=True)
            assert ids.tolist() == expected, "Cached tokenizer differs from original"


def words(text):
    return WORDS.findall(unicodedata.normalize("NFKC", text).casefold())


def shingles(ws, n):
    return np.unique(np.fromiter((int.from_bytes(hashlib.blake2b(" ".join(ws[i:i+n]).encode(), digest_size=8).digest(), "little")
                                 for i in range(max(0, len(ws) - n + 1))), dtype=np.uint64))


class Dedupe:
    def __init__(self, db):
        self.db = db
        bloom_path = WORK / "training_13gram_bloom.bin"
        self.bloom = np.memmap(bloom_path, mode="r+" if bloom_path.exists() else "w+", dtype="u1", shape=(2**28,))
        self.bands = defaultdict(list)
        self.exact = set()
        self.parents = set()
        self.eval_grams = set()
        rng = np.random.default_rng(20260919)
        self.a = rng.integers(1, 2**32, size=128, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**32, size=128, dtype=np.uint64)
        for rid, exact, parent, sig in db.execute("SELECT id,exact,parent,signature FROM docs"):
            self.exact.add(exact)
            if parent:
                self.parents.add(parent)
            self._index(rid, sig)
        for (blob,) in db.execute("SELECT hashes FROM evaluation_grams"):
            self.eval_grams.update(np.frombuffer(zlib.decompress(blob), dtype="<u8").tolist())

    def signature(self, sh):
        # 32-bit affine minhash; band candidates are confirmed with full 64-bit Jaccard.
        value = np.full(128, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, len(sh), 2048):
            x = sh[start:start+2048] & np.uint64(0xffffffff)
            h = ((x[:, None] * self.a + self.b) % np.uint64(4294967311))
            value = np.minimum(value, h.min(axis=0))
        return value.astype("<u8").tobytes()

    def _index(self, rid, sig):
        for b in range(32):
            self.bands[(b, sig[b*32:(b+1)*32])].append(rid)

    def features(self, text):
        ws = words(text)
        exact = hashlib.sha256(" ".join(ws).encode()).hexdigest()
        sh = shingles(ws, 5)
        return ws, exact, sh, self.signature(sh)

    def reason(self, ws, exact, sh, sig, parent, validation=False):
        if parent and parent in self.parents:
            return "existing_parent"
        if exact in self.exact:
            return "normalized_exact_duplicate"
        ng = shingles(ws, 13)
        if any(int(v) in self.eval_grams for v in ng):
            return "evaluation_13word_overlap"
        if validation and self.bloom_matches(ng):
            return "validation_overlap_training_conservative_bloom"
        candidates = set()
        for b in range(32):
            candidates.update(self.bands.get((b, sig[b*32:(b+1)*32]), ()))
        for rid in candidates:
            (blob,) = self.db.execute("SELECT shingles FROM docs WHERE id=?", (rid,)).fetchone()
            other = np.frombuffer(zlib.decompress(blob), dtype="<u8")
            if min(len(sh), len(other)) < 0.8 * max(len(sh), len(other)):
                continue
            intersection = np.intersect1d(sh, other, assume_unique=True).size
            if intersection / max(1, len(sh) + len(other) - intersection) >= 0.8:
                return "near_duplicate_jaccard_0.8"
        return None

    def bloom_indices(self, hashes):
        h1 = hashes & np.uint64(0xffffffff)
        h2 = (hashes >> np.uint64(32)) | np.uint64(1)
        for k in range(4):
            idx = (h1 + np.uint64(k) * h2) & np.uint64(2**31-1)
            yield (idx >> np.uint64(3)).astype(np.int64), (np.uint8(1) << (idx & np.uint64(7)).astype(np.uint8))

    def bloom_matches(self, hashes):
        matches = np.ones(len(hashes), dtype=bool)
        for indices, bits in self.bloom_indices(hashes):
            matches &= (self.bloom[indices] & bits) != 0
        return bool(matches.any())

    def add(self, exact, sh, sig, parent, permission, ws=None):
        c = self.db.execute("INSERT INTO docs(exact,parent,signature,shingles,permission) VALUES(?,?,?,?,?)",
                            (exact, parent, sig, zlib.compress(sh.astype("<u8").tobytes()), permission))
        rid = c.lastrowid
        self.exact.add(exact)
        if parent:
            self.parents.add(parent)
        self._index(rid, sig)
        if permission in ("validation", "never_train") and ws is not None:
            ng = shingles(ws, 13)
            self.eval_grams.update(ng.tolist())
            self.db.execute("INSERT INTO evaluation_grams(hashes) VALUES(?)", (zlib.compress(ng.astype("<u8").tobytes()),))
        elif ws is not None:
            for indices, bits in self.bloom_indices(shingles(ws, 13)):
                np.bitwise_or.at(self.bloom, indices, bits)
        return rid


def database():
    WORK.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(WORK / "curation.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
    CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY,exact TEXT,parent TEXT,signature BLOB,shingles BLOB,permission TEXT);
    CREATE TABLE IF NOT EXISTS evaluation_grams(hashes BLOB);
    CREATE TABLE IF NOT EXISTS accepted(id INTEGER PRIMARY KEY,source TEXT,permission TEXT,row_idx INTEGER,record TEXT,tokens BLOB,loss_count INTEGER);
    CREATE TABLE IF NOT EXISTS pages(source TEXT,offset INTEGER,stats TEXT,PRIMARY KEY(source,offset));
    """)
    db.commit()
    return db


def baseline_index(db, tok):
    done = db.execute("SELECT value FROM meta WHERE key='baseline_complete'").fetchone()
    dd = Dedupe(db)
    if done:
        return dd
    if db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]:
        raise RuntimeError("Incomplete baseline index: use a fresh --work directory to recover")
    counts = Counter()
    duplicates = 0
    for path in sorted((ROOT / "baseline/data/frozen_corpus_v1").glob("*/*.jsonl.gz")):
        permission = path.name.split(".")[0]
        for record in read_jsonl(path):
            ws, exact, sh, sig = dd.features(record["text"])
            parent = ""
            if record["source_id"].startswith("wikipedia"):
                parent = "wikipedia:" + str(record.get("parent_upstream_id", record["upstream_id"]))
            duplicates += int(exact in dd.exact)
            dd.add(exact, sh, sig, parent, permission, ws)
            counts[permission] += 1
            if sum(counts.values()) % 2000 == 0:
                log("index_baseline", documents=sum(counts.values()), eval_ngrams=len(dd.eval_grams))
    # Check the fast encoder against the authoritative tokenizer on diverse baseline records.
    for path in sorted((ROOT / "baseline/data/frozen_corpus_v1").glob("*/*.jsonl.gz")):
        for k, rec in enumerate(read_jsonl(path)):
            if k >= 3:
                break
            ids = tok.encode(rec["text"])
            tok.verify(rec["text"], ids, parity=True)
    report = dict(records=dict(counts), normalized_exact_duplicates_observed=duplicates,
                  inherited_baseline_preserved=True, baseline_rewritten=False,
                  eval_13word_hashes=len(dd.eval_grams), tokenizer_parity_records=66)
    write_json(ROOT / "reports/baseline_index.json", report)
    db.execute("INSERT INTO meta(key,value) VALUES('baseline_complete',?)", (json.dumps(report),))
    db.commit()
    dd.bloom.flush()
    log("baseline_index_complete", **report)
    return dd


def source_lock(source):
    path = ROOT / "sources" / source["id"] / "lock.json"
    if path.exists():
        return json.loads(path.read_text())
    ds = source["dataset"]
    hub, _ = request("https://huggingface.co/api/datasets/" + ds)
    rev = hub["sha"]
    splits, _ = viewer("splits", dataset=ds)
    assert any(s["config"] == source["config"] and s["split"] == "train" for s in splits["splits"])
    preview, headers = viewer("rows", dataset=ds, config=source["config"], split="train", offset=0, length=1)
    assert headers.get("x-revision") == rev
    info = dict(**source, revision=rev, upstream_split="train", num_rows=preview["num_rows_total"],
                features=preview["features"], license_metadata=hub.get("cardData", {}).get("license"),
                webpage="https://huggingface.co/datasets/"+ds,
                sampling="Seeded random disjoint blocks of 100 rows across the complete config; shuffle rows within blocks.",
                token_counting="Frozen 8192-token tokenizer; target-aligned loss mask; BOS excluded, EOS included.")
    path.parent.mkdir(parents=True, exist_ok=True)
    card, _ = request(f"https://huggingface.co/datasets/{ds}/raw/{rev}/README.md", raw=True)
    (path.parent / "UPSTREAM_README.md").write_bytes(card)
    info["readme_sha256"] = hashlib.sha256(card).hexdigest()
    write_json(path, info)
    log("source_locked", source=source["id"], revision=rev, rows=info["num_rows"])
    return info


def fetch_page(lock, offset):
    cache = WORK / "raw_pages" / lock["id"] / f"{offset:012d}.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            payload = json.load(f)
            assert payload["revision"] == lock["revision"], "Cached page revision differs from source lock"
            return payload
    result, headers = viewer("rows", dataset=lock["dataset"], config=lock["config"], split="train", offset=offset, length=100)
    if headers.get("x-revision") != lock["revision"]:
        raise RuntimeError("Dataset revision changed during download: " + lock["id"])
    payload = dict(revision=lock["revision"], offset=offset, response=result)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_suffix(".tmp")
    with gz_writer(temp) as f:
        f.write(canonical(payload))
    os.replace(temp, cache)
    return payload


def clean(row, source):
    text = row.get("text", "")
    if not isinstance(text, str) or not text:
        return None, "missing_text"
    if len(text) > POLICY["max_characters"]:
        return None, "too_long"
    if "\ufffd" in text or "\x00" in text:
        return None, "encoding_damage"
    if source == "fineweb_edu":
        if row.get("int_score", 0) < 3 or row.get("language_score", 0) < 0.9:
            return None, "fineweb_score_or_language"
    if source == "wikipedia_en":
        title = row.get("title", "")
        if title.startswith(("List of ", "Lists of ", "Category:", "Template:", "Portal:", "File:")) or "(disambiguation)" in title:
            return None, "wiki_list_or_disambiguation"
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Remove obvious contact data. Preserve technical numbers and useful punctuation.
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    ws = words(text)
    if len(ws) < 80:
        return None, "too_short"
    nonspace = sum(not c.isspace() for c in text)
    if sum(c.isalpha() for c in text) / max(1, nonspace) < 0.60:
        return None, "low_prose"
    if len(set(ws)) / len(ws) < 0.15:
        return None, "repetitive_words"
    lines = [s.strip().casefold() for s in text.splitlines() if s.strip()]
    if 1 - len(set(lines)) / max(1, len(lines)) > 0.25:
        return None, "repeated_lines"
    if re.search(r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,})", text):
        return None, "credential_pattern"
    return text, None


def ingest(db, dd, tok, source):
    lock = source_lock(source)
    sid = source["id"]
    totals = Counter({p: int(n) for p, n in db.execute("SELECT permission,SUM(loss_count) FROM accepted WHERE source=? GROUP BY permission", (sid,))})
    accepted_count = db.execute("SELECT COUNT(*) FROM accepted WHERE source=?", (sid,)).fetchone()[0]
    complete_pages = {r[0] for r in db.execute("SELECT offset FROM pages WHERE source=?", (sid,))}
    blocks = list(range(lock["num_rows"] // 100))
    rng = random.Random(POLICY["sampling_seed"] + int.from_bytes(hashlib.sha256(sid.encode()).digest()[:4], "little"))
    rng.shuffle(blocks)
    # Holdout overlap screening must include all baseline training too. A persistent
    # DB of baseline/new-training 13-grams is queried only for candidate holdouts.
    targets = {"train": source["train"], "validation": source["validation"]}
    remaining = (b * 100 for b in blocks if b * 100 not in complete_pages)
    failures = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while any(totals[p] < targets[p] for p in targets):
            offsets = [next(remaining) for _ in range(4)]
            pages = list(pool.map(lambda o: fetch_page(lock, o), offsets))
            for page in pages:
                stats = Counter()
                rows = page["response"]["rows"]
                random.Random(page["offset"] + POLICY["sampling_seed"]).shuffle(rows)
                for wrapper in rows:
                    if all(totals[p] >= targets[p] for p in targets):
                        break
                    stats["seen"] += 1
                    if wrapper.get("truncated_cells"):
                        stats["truncated_viewer_row"] += 1
                        continue
                    row = wrapper["row"]
                    text, why = clean(row, sid)
                    if why:
                        stats[why] += 1
                        continue
                    ws, exact, sh, sig = dd.features(text)
                    # Stable whole-parent split; never move a record to fill a quota.
                    upstream = str(row.get("id", wrapper["row_idx"]))
                    parent = "wikipedia:" + upstream if sid == "wikipedia_en" else ""
                    split_hash = hashlib.sha256((sid + ":" + upstream + ":20260919").encode()).digest()
                    permission = "validation" if int.from_bytes(split_hash[:8], "little") / 2**64 < 0.05 else "train"
                    if totals[permission] >= targets[permission]:
                        stats[permission + "_quota_filled"] += 1
                        continue
                    why = dd.reason(ws, exact, sh, sig, parent, validation=permission=="validation")
                    if why:
                        stats[why] += 1
                        continue
                    ids = tok.encode(text)
                    tok.verify(text, ids, parity=accepted_count < 15)
                    count = len(ids) - 1
                    rid = "new_" + hashlib.sha256((sid+":"+upstream).encode()).hexdigest()[:24]
                    metadata = {k:row[k] for k in ("url","title","dump","score","int_score","language_score","audience","format","seed_data") if k in row}
                    rec = dict(record_id=rid, group_id=rid, source_id=sid,
                               source_revision=lock["revision"], upstream_id=upstream,
                               capability_lane="general", permission=permission, language="en",
                               license_id=source["license"], text=text,
                               content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                               normalized_sha256=exact, metadata=metadata,
                               hf_dataset=source["dataset"], hf_config=source["config"], hf_split="train",
                               hf_row_idx=wrapper["row_idx"], cleaning_policy_sha256=hashlib.sha256(canonical(POLICY)).hexdigest())
                    docid = dd.add(exact, sh, sig, parent, permission, ws)
                    db.execute("INSERT INTO accepted(id,source,permission,row_idx,record,tokens,loss_count) VALUES(?,?,?,?,?,?,?)",
                               (docid,sid,permission,wrapper["row_idx"],json.dumps(rec,ensure_ascii=False),ids.tobytes(),count))
                    totals[permission] += count
                    accepted_count += 1
                    stats["accepted_" + permission] += 1
                db.execute("INSERT INTO pages(source,offset,stats) VALUES(?,?,?)", (sid,page["offset"],json.dumps(stats)))
                db.commit()
                dd.bloom.flush()
                failures.update(stats)
            log("ingest_progress", source=sid, train_tokens=totals["train"], validation_tokens=totals["validation"], accepted_records=accepted_count)
    log("source_complete", source=sid, **dict(totals))


def export_additions(db, tok):
    report = dict(sources=[], policy=POLICY, tokenizer_hash=tok.hash)
    for source in SOURCES:
        sid = source["id"]
        source_report = dict(source=sid, splits={})
        for permission in ("train", "validation"):
            root = ROOT / "additions" / sid / permission
            root.mkdir(parents=True, exist_ok=True)
            offset = loss_count = records = 0
            with gz_writer(root / "documents.jsonl.gz") as docs, gz_writer(root / "index.jsonl.gz") as index, (root/"tokens.uint16.bin").open("wb") as tf, (root/"loss.uint8.bin").open("wb") as lf:
                for rec, blob, losses in db.execute("SELECT record,tokens,loss_count FROM accepted WHERE source=? AND permission=? ORDER BY id",(sid,permission)):
                    r=json.loads(rec)
                    n=len(blob)//2
                    docs.write(canonical(r)+b"\n")
                    index.write(canonical(dict(record_id=r["record_id"],group_id=r["group_id"],source_id=sid,
                                               token_offset=offset,token_count=n,loss_bearing_token_count=losses,loss_policy="full_text_causal"))+b"\n")
                    tf.write(blob)
                    lf.write(b"\x00"+b"\x01"*(n-1))
                    records+=1;offset+=n;loss_count+=losses
            manifest = dict(source=sid, permission=permission, records=records, token_count=offset,
                            loss_bearing_token_count=loss_count, tokenizer_hash=tok.hash,
                            files={f.name:dict(sha256=digest_file(f),bytes=f.stat().st_size) for f in root.iterdir() if f.name!="manifest.json"})
            write_json(root/"manifest.json",manifest)
            source_report["splits"][permission]=manifest
        counters=Counter()
        for (stats,) in db.execute("SELECT stats FROM pages WHERE source=?",(sid,)):
            counters.update(json.loads(stats))
        source_report["filters"]=dict(counters)
        report["sources"].append(source_report)
    report["added_train_loss_tokens"]=sum(s["splits"]["train"]["loss_bearing_token_count"] for s in report["sources"])
    report["added_validation_loss_tokens"]=sum(s["splits"]["validation"]["loss_bearing_token_count"] for s in report["sources"])
    write_json(ROOT/"reports/additions.json",report)
    return report


def build_catalog():
    report=json.loads((ROOT/"baseline/data/tokenized_v2/tokenized_report.json").read_text())
    shards=[]
    for s in report["shards"]:
        root=ROOT/"baseline/data/tokenized_v2"/s["lane"]/s["permission"]
        shards.append(dict(id=s["shard_id"],lane=s["lane"],permission=s["permission"],
                           tokens=(root/"tokens.uint16.bin").relative_to(ROOT).as_posix(),
                           loss=(root/"loss.uint8.bin").relative_to(ROOT).as_posix(),
                           index=(root/"index.jsonl.gz").relative_to(ROOT).as_posix(),
                           token_count=s["token_count"],loss_count=s["loss_bearing_token_count"]))
    for source in SOURCES:
        for permission in ("train","validation"):
            root=ROOT/"additions"/source["id"]/permission
            m=json.loads((root/"manifest.json").read_text())
            shards.append(dict(id=source["id"]+"-"+permission,lane="general",permission=permission,
                               tokens=(root/"tokens.uint16.bin").relative_to(ROOT).as_posix(),
                               loss=(root/"loss.uint8.bin").relative_to(ROOT).as_posix(),
                               index=(root/"index.jsonl.gz").relative_to(ROOT).as_posix(),
                               token_count=m["token_count"],loss_count=m["loss_bearing_token_count"]))
    catalog=dict(shards=shards,train_permissions=["train","anneal"],
                 training_loss_tokens=sum(s["loss_count"] for s in shards if s["permission"] in ("train","anneal")),
                 validation_loss_tokens=sum(s["loss_count"] for s in shards if s["permission"]=="validation"),
                 tokenizer="baseline/artifacts/tokenizer_v2/tokenizer.json")
    write_json(ROOT/"catalog.json",catalog)
    return catalog


def pack(catalog, target=50_000_000, context=512):
    """One deterministic pass, segmented causal attention, one-token overlap.

    loss_mask at position j says whether token[j] is a supervised TARGET.
    A trainer predicts input_ids[:,1:] from logits[:,:-1] with loss_mask[:,1:].
    Every split continuation carries the preceding token with a zero loss mask.
    """
    pack_root=ROOT/"packed_50m_ctx512"
    pack_root.mkdir(parents=True,exist_ok=True)
    refs=[]; maps={}
    for shard in catalog["shards"]:
        if shard["permission"] not in ("train","anneal"):
            continue
        sid=shard["id"]
        maps[sid]=(np.memmap(ROOT/shard["tokens"],mode="r",dtype="<u2"),np.memmap(ROOT/shard["loss"],mode="r",dtype="u1"))
        for r in read_jsonl(ROOT/shard["index"]):
            if r["loss_bearing_token_count"]:
                refs.append((sid,r))
    random.Random(20260919).shuffle(refs)
    names={"input_ids":"<u2","loss_mask":"u1","segment_ids":"<i2","position_ids":"<u2"}
    handles={k:(pack_root/(k+"."+{"input_ids":"uint16","loss_mask":"uint8","segment_ids":"int16","position_ids":"uint16"}[k]+".bin")).open("wb") for k in names}
    buffers={k:np.zeros(context,dtype=d) for k,d in names.items()}
    buffers["segment_ids"].fill(-1)
    pos=segment=sequence=total=nonpad=overlap=0
    selected=Counter(); selected_records=Counter(); fully_consumed=0; last_partial=False
    index=gz_writer(pack_root/"segments.jsonl.gz")
    def flush():
        nonlocal pos,segment,sequence
        for k,f in handles.items(): f.write(buffers[k].tobytes())
        for k in buffers:buffers[k].fill(-1 if k=="segment_ids" else 0)
        pos=segment=0;sequence+=1
    for sid,r in refs:
        if total>=target:break
        start=r["token_offset"]; n=r["token_count"]
        ids,mask=maps[sid]
        cursor=0;selected_records[sid]+=1
        while cursor<n-1 and total<target:
            if context-pos<2:flush()
            length=min(n-cursor,context-pos)
            chunk_mask=np.array(mask[start+cursor:start+cursor+length],copy=True)
            chunk_mask[0]=0
            losses=int(chunk_mask.sum())
            if total+losses>target:
                locations=np.flatnonzero(chunk_mask)
                length=int(locations[target-total-1])+1
                chunk_mask=chunk_mask[:length]
                losses=int(chunk_mask.sum())
                last_partial=True
            buffers["input_ids"][pos:pos+length]=ids[start+cursor:start+cursor+length]
            buffers["loss_mask"][pos:pos+length]=chunk_mask
            buffers["segment_ids"][pos:pos+length]=segment
            buffers["position_ids"][pos:pos+length]=np.arange(length,dtype="<u2")
            index.write(canonical(dict(sequence=sequence,start=pos,length=length,segment=segment,
                                       shard=sid,record_id=r["record_id"],record_token_start=cursor,
                                       loss_tokens=losses,continuation=cursor>0))+b"\n")
            total+=losses;selected[sid]+=losses;nonpad+=length;overlap+=int(cursor>0)
            pos+=length;segment+=1
            if cursor+length>=n:
                fully_consumed+=1;break
            cursor+=length-1
        if sequence and sequence%10000<3:
            log("packing_progress",loss_tokens=total,sequences=sequence)
    if pos:flush()
    for f in handles.values():f.close()
    index.close()
    assert total==target,(total,target)
    report=dict(status="COMPLETE",context_length=context,sequences=sequence,loss_bearing_tokens=total,
                physical_tokens=sequence*context,nonpadding_tokens=nonpad,padding_tokens=sequence*context-nonpad,
                utilization=nonpad/(sequence*context),continuation_overlap_tokens=overlap,
                selected_loss_tokens_by_shard=dict(selected),selected_records_by_shard=dict(selected_records),
                fully_consumed_records=fully_consumed,last_record_partial=last_partial,
                sampling="One shuffled pass over train/anneal records; no record resampling; stop at exact token budget.",
                attention_policy="causal_within_segment_only",position_policy="reset_per_fragment",
                mask_convention="Target-aligned; multiply shifted labels' losses by loss_mask[:,1:].",
                files={f.name:dict(sha256=digest_file(f),bytes=f.stat().st_size) for f in pack_root.iterdir() if f.name!="packing_report.json"})
    write_json(pack_root/"packing_report.json",report)
    return report


def verify(catalog, packing):
    checks={}
    for s in catalog["shards"]:
        ids=np.memmap(ROOT/s["tokens"],mode="r",dtype="<u2")
        mask=np.memmap(ROOT/s["loss"],mode="r",dtype="u1")
        assert len(ids)==len(mask)==s["token_count"]
        assert int(mask.sum())==s["loss_count"]
        assert int(ids.max())<8192
        assert np.all(mask<=1)
    checks["all_shard_sizes_ids_and_loss_counts"]=True
    assert not any("validation" in k or "never_train" in k for k in packing["selected_loss_tokens_by_shard"])
    checks["no_evaluation_shards_in_packing"]=True
    root=ROOT/"packed_50m_ctx512"; shape=(packing["sequences"],512)
    ids=np.memmap(root/"input_ids.uint16.bin",mode="r",dtype="<u2",shape=shape)
    loss=np.memmap(root/"loss_mask.uint8.bin",mode="r",dtype="u1",shape=shape)
    seg=np.memmap(root/"segment_ids.int16.bin",mode="r",dtype="<i2",shape=shape)
    pos=np.memmap(root/"position_ids.uint16.bin",mode="r",dtype="<u2",shape=shape)
    assert int(loss.sum())==50_000_000
    for i in range(0,len(ids),2048):
        l=loss[i:i+2048];s=seg[i:i+2048];p=pos[i:i+2048]
        assert not l[:,0].any()
        boundaries=np.ones(s.shape,dtype=bool);boundaries[:,1:]=s[:,1:]!=s[:,:-1]
        assert not l[boundaries].any()
        assert not p[boundaries & (s>=0)].any()
        assert not l[s<0].any()
        same=(s[:,1:]==s[:,:-1])&(s[:,1:]>=0)
        assert np.all((p[:,1:]-p[:,:-1])[same]==1)
    checks["exact_50m_shifted_supervised_targets"]=True
    checks["all_segment_boundaries_and_padding_masked"]=True
    checks["all_positions_reset_and_increment"]=True
    # Source reconstruction for every packed fragment also verifies continuation masks.
    catalog_by_id={s["id"]:s for s in catalog["shards"]}
    maps={}; indexes={}; seen_positions=defaultdict(int); segment_count=0
    for s in catalog["shards"]:
        if s["permission"] in ("train","anneal"):
            indexes[s["id"]]={r["record_id"]:r for r in read_jsonl(ROOT/s["index"])}
            maps[s["id"]]=(np.memmap(ROOT/s["tokens"],mode="r",dtype="<u2"),np.memmap(ROOT/s["loss"],mode="r",dtype="u1"))
    for r in read_jsonl(root/"segments.jsonl.gz"):
        q=r["sequence"];a=r["start"];n=r["length"];sid=r["shard"];rid=r["record_id"]
        record=indexes[sid][rid];offset=record["token_offset"]+r["record_token_start"]
        original,mask=maps[sid]
        assert np.array_equal(ids[q,a:a+n],original[offset:offset+n])
        expected=np.array(mask[offset:offset+n],copy=True);expected[0]=0
        assert np.array_equal(loss[q,a:a+n],expected)
        key=(sid,rid)
        assert r["record_token_start"]==seen_positions[key]
        seen_positions[key]=r["record_token_start"]+n-1
        segment_count+=1
    checks["all_fragments_match_source_tokens_and_masks"]=True
    checks["no_resampled_record_spans"]=True
    inv=json.loads((ROOT/"reports/baseline_snapshot.json").read_text())
    for f in inv["files"]:
        assert digest_file(ROOT/f["path"])==f["sha256"]
        if f["path"].startswith("baseline/"):
            assert digest_file(BASELINE/Path(f["path"]).relative_to("baseline"))==f["sha256"]
    checks["original_baseline_files_unchanged"]=True
    result=dict(status="PASS",checks=checks,packed_fragments_verified=segment_count,
                limitations=["MinHash LSH plus Jaccard is heuristic near-duplicate detection, not exhaustive pairwise comparison.",
                             "Inherited baseline records and source permissions are preserved, not re-curated.",
                             "13-word screening is symmetric for additions: exact hashes against held-out text, conservative Bloom membership against training text; false-positive rejections are possible.",
                             "Data screened automatically; no claim of human review or benchmark improvement."])
    write_json(ROOT/"reports/verification.json",result)
    log("verification_complete",**result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--work",type=Path)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    global WORK
    if args.work:WORK=args.work.resolve()
    if args.verify_only:
        verify(json.loads((ROOT/"catalog.json").read_text()),json.loads((ROOT/"packed_50m_ctx512/packing_report.json").read_text()))
        from audit_overlap import run
        run()
        return
    start=time.monotonic()
    snapshot_baseline()
    tok=FastTokenizer()
    write_json(ROOT/"config.json",dict(sources=SOURCES,policy=POLICY,packed_training_tokens=50_000_000,context_length=512,baseline=str(BASELINE)))
    db=database()
    dd=baseline_index(db,tok)
    for source in SOURCES:ingest(db,dd,tok,source)
    additions=export_additions(db,tok)
    catalog=build_catalog()
    packing=pack(catalog)
    result=verify(catalog,packing)
    from audit_overlap import run
    run()
    write_json(ROOT/"reports/build.json",dict(status=result["status"],elapsed_seconds=time.monotonic()-start,
                                              added_train_loss_tokens=additions["added_train_loss_tokens"],
                                              added_validation_loss_tokens=additions["added_validation_loss_tokens"],
                                              training_pool_loss_tokens=catalog["training_loss_tokens"],
                                              packed_training_tokens=packing["loss_bearing_tokens"]))
    log("BUILD_COMPLETE",training_pool=catalog["training_loss_tokens"],packed=packing["loss_bearing_tokens"])


if __name__=="__main__":main()
