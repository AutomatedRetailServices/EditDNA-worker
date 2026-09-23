"""D-288.4 (blocker 2): one atomic "append this record to a bounded JSON
list UNLESS an entry matching these fields already exists -- and return
whichever entry survives" primitive, shared by `render_versions.add_
render_version` and `notifications.publish_notification`.

Before this module both did a plain GET -> scan -> SET in Python -- two
concurrent callers could each read "not present", each append, and hand
back two DIFFERENT ids for the same render. A single Lua script makes the
check-and-append ONE atomic Redis operation, so every concurrent caller
receives the SAME stored entry (the first writer's), and a retry after an
interruption finds and returns it rather than minting a second one.
`cjson` is built into real Redis's Lua sandbox (no extension needed);
the newest entry is kept at the front and the list is trimmed to
`max_len`, exactly the shape both callers already stored.
"""
from __future__ import annotations

import json

APPEND_IF_ABSENT_LUA = """
local raw = redis.call('GET', KEYS[1])
local items = {}
if raw then
  local ok, decoded = pcall(cjson.decode, raw)
  if ok and type(decoded) == 'table' then
    items = decoded
  end
end
local match = cjson.decode(ARGV[1])
for _, existing in ipairs(items) do
  if type(existing) == 'table' then
    local all = true
    for field, wanted in pairs(match) do
      if existing[field] ~= wanted then
        all = false
        break
      end
    end
    if all then
      return cjson.encode(existing)
    end
  end
end
local record = cjson.decode(ARGV[2])
table.insert(items, 1, record)
local max_len = tonumber(ARGV[3])
while #items > max_len do
  table.remove(items)
end
redis.call('SET', KEYS[1], cjson.encode(items))
return ARGV[2]
"""


def append_if_absent(client, key: str, *, match: dict, record: dict, max_len: int) -> dict:
    """Atomically append `record` to the JSON list at `key` unless an entry
    whose fields equal every `match` pair already exists; return the entry
    that is stored afterwards (the existing one on a match, else
    `record`). `match` values must be JSON scalars."""
    result = client.eval(
        APPEND_IF_ABSENT_LUA, 1, key,
        json.dumps(match, ensure_ascii=False),
        json.dumps(record, ensure_ascii=False),
        str(int(max_len)),
    )
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return dict(json.loads(result))
