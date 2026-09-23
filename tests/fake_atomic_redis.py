"""D-288.4: one shared, thread-safe fake Redis whose `eval` emulates -- by
script identity, never by guessing -- the exact atomic Lua scripts the
production modules ship:

- `pending_watch_listen_review._CAS_LUA` (versioned compare-and-set of one
  JSON record);
- `redis_atomic_list.APPEND_IF_ABSENT_LUA` (append-if-absent to a bounded
  JSON list, matched on caller-named fields; returns the surviving entry).

Every access, including each whole `eval`, runs under one lock, so the
fake is a real race-free primitive for multi-thread proofs (the same
precedent `test_cutsell_d288_editorial_slot_resolution_observability.py`
set), not a simulation of one. `get_hook`/`set_hook`, when set, fire
OUTSIDE the lock so a test can force a chosen interleaving between a read
and the write that follows it -- exactly the "carreras después de la
última lectura y dentro de las escrituras" the D-288.3 review asked to be
reproduced first.
"""
from __future__ import annotations

import json
import threading

from cutsell_worker import pending_watch_listen_review as pwl
from cutsell_worker import redis_atomic_list


class FakeAtomicRedis:
    def __init__(self):
        self.data: dict[str, object] = {}
        self._lock = threading.RLock()
        self.get_hook = None
        self.set_hook = None

    def get(self, key):
        if self.get_hook is not None:
            self.get_hook(key)
        with self._lock:
            return self.data.get(key)

    def set(self, key, value, **_kwargs):
        if self.set_hook is not None:
            self.set_hook(key)
        with self._lock:
            self.data[key] = value
        return True

    def zadd(self, key, mapping):
        with self._lock:
            bucket = self.data.setdefault(f"__zset__{key}", {})
            bucket.update(mapping)
        return len(mapping)

    def eval(self, script, numkeys, *keys_and_args):
        with self._lock:
            if script == pwl._CAS_LUA:
                return self._cas(*keys_and_args)
            if script == redis_atomic_list.APPEND_IF_ABSENT_LUA:
                return self._append_if_absent(*keys_and_args)
            raise AssertionError("FakeAtomicRedis.eval: unknown script -- add an explicit emulation")

    def _cas(self, key, expected_version, new_value):
        current = self.data.get(key)
        if current is None:
            return "missing"
        decoded = json.loads(current)
        if str(decoded.get("version")) != str(expected_version):
            return "conflict"
        self.data[key] = new_value
        return "ok"

    def _append_if_absent(self, key, match_json, record_json, max_len):
        raw = self.data.get(key)
        items = json.loads(raw) if raw else []
        if not isinstance(items, list):
            items = []
        match = json.loads(match_json)
        for existing in items:
            if isinstance(existing, dict) and all(existing.get(k) == v for k, v in match.items()):
                return json.dumps(existing, ensure_ascii=False)
        record = json.loads(record_json)
        items.insert(0, record)
        self.data[key] = json.dumps(items[: int(max_len)], ensure_ascii=False)
        return record_json
