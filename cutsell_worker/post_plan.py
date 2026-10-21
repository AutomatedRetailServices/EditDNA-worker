"""Plan my week: what to post and when, from videos that are already Ready (Single, Combine, Bulk or Mix).

Rules, no AI (the cut is never touched):
- Ready and never posted videos only; the oldest first.
- Never the same product in two slots in a row (a video without a product is its own product).
- Videos of one Mix go on different days, and the same hook of a Mix is never used on two days in a row.
- Not enough videos: the days left short and how many videos are missing are returned ("Videos for X of Y days").
- Posted slots and past days are kept when the week is built again; a slot the creator moved stays where she put it.

One JSON per creator in Redis: {"prefs", "videos": {project_id: {source, posted_at, scheduled_for, caption,
caption_source, caption_log}}, "week": {"start", "built_at", "slots": [{date, time, project_id, pinned}]}}."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from .project_store import _redis_client

DEFAULT_PREFS = {"per_day": 2, "days": [0, 1, 2, 3, 4, 5], "tz": "America/New_York",
                 "reminders": [{"label": "Morning", "time": "09:00", "on": True},
                               {"label": "Evening", "time": "19:00", "on": True},
                               {"label": "Night", "time": "22:00", "on": False}]}
EXTRA_TIMES = ["09:00", "13:00", "19:00", "21:00"]
MAX_PER_DAY = 4
SOURCES = ("single", "combine", "bulk", "mix")


def _scope(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]


def plan_key(user_id: str) -> str:
    return f"cutsell:v1:plan:{_scope(user_id)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(user_id: str, *, client=None) -> dict[str, Any]:
    raw = _redis_client(client).get(plan_key(user_id))
    data = json.loads(raw) if raw else {}
    data.setdefault("prefs", json.loads(json.dumps(DEFAULT_PREFS)))
    data.setdefault("videos", {})
    data.setdefault("week", None)
    return data


def update(user_id: str, change: Callable[[dict], Any], *, client=None):
    r = _redis_client(client)
    key = plan_key(user_id)
    from redis.exceptions import WatchError
    for _ in range(20):
        with r.pipeline() as pipe:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                data = json.loads(raw) if raw else {}
                data.setdefault("prefs", json.loads(json.dumps(DEFAULT_PREFS)))
                data.setdefault("videos", {})
                data.setdefault("week", None)
                result = change(data)
                pipe.multi()
                pipe.set(key, json.dumps(data, ensure_ascii=False))
                pipe.execute()
                return result if result is not None else data
            except WatchError:
                continue
    raise RuntimeError("plan update conflict")


def video(user_id: str, project_id: str, *, client=None) -> dict:
    return load(user_id, client=client)["videos"].get(project_id) or {}


def record_source(user_id: str, project_id: str, source: str, *, client=None) -> None:
    if source not in SOURCES:
        return
    update(user_id, lambda d: d["videos"].setdefault(project_id, {}).update(source=source), client=client)


def set_caption(user_id: str, project_id: str, caption: str, *, source: str, log: dict | None = None,
                client=None) -> dict:
    """source "user" always wins; an AI caption never replaces one that exists."""
    def change(data):
        entry = data["videos"].setdefault(project_id, {})
        if source == "ai" and entry.get("caption"):
            return entry
        entry.update(caption=str(caption).strip()[:2200], caption_source=source, caption_at=_now())
        if log:
            entry["caption_log"] = log
        return entry
    return update(user_id, change, client=client)


def mark_posted(user_id: str, project_id: str, *, posted: bool = True, client=None) -> dict:
    def change(data):
        entry = data["videos"].setdefault(project_id, {})
        entry["posted_at"] = _now() if posted else None
        return entry
    return update(user_id, change, client=client)


def set_prefs(user_id: str, prefs: dict, *, client=None) -> dict:
    def change(data):
        clean = dict(data["prefs"])
        if "per_day" in prefs:
            clean["per_day"] = max(1, min(MAX_PER_DAY, int(prefs["per_day"])))
        if "days" in prefs:
            clean["days"] = sorted({int(d) for d in prefs["days"] if 0 <= int(d) <= 6}) or [0]
        if "tz" in prefs:
            ZoneInfo(str(prefs["tz"]))                       # unknown zone -> error
            clean["tz"] = str(prefs["tz"])
        if "reminders" in prefs:
            rows = []
            for row in prefs["reminders"][:6]:
                hh, mm = str(row.get("time", "09:00")).split(":")[:2]
                rows.append({"label": str(row.get("label") or "Reminder")[:30], "time": f"{int(hh) % 24:02d}:{int(mm) % 60:02d}",
                             "on": bool(row.get("on", True))})
            clean["reminders"] = rows
        data["prefs"] = clean
        return clean
    return update(user_id, change, client=client)


# ---------------------------------------------------------------- the videos that can be planned

def source_of(project: dict, entry: dict, link: dict) -> str:
    if link.get("mix_id"):
        return "mix"
    if entry.get("source") in SOURCES:
        return entry["source"]
    sources = [s for s in project.get("sources") or [] if isinstance(s, dict)]
    return "combine" if len(sources) > 1 else "single"


def candidates(user_id: str, projects: list[dict], *, client=None, plan: dict | None = None) -> list[dict]:
    """Ready videos with their product, Mix code and posting state (from the project library + the catalog)."""
    from . import product_catalog
    plan = plan or load(user_id, client=client)
    catalog = product_catalog.load(user_id, client=client)
    out = []
    for project in projects:
        if project.get("state") != "draft_ready":
            continue
        pid = str(project.get("project_id"))
        link = catalog["videos"].get(pid) or {}
        entry = plan["videos"].get(pid) or {}
        product = catalog["products"].get(link.get("product_id") or "") or {}
        brand = (catalog["brands"].get(product.get("brand_id") or "") or {}).get("name")
        seconds = sum(float(s.get("duration_sec") or 0) for s in project.get("sources") or [] if isinstance(s, dict))
        out.append({"project_id": pid, "title": project.get("title"), "created_at": project.get("created_at") or "",
                    "product_id": link.get("product_id") if product else None, "product": product.get("name"),
                    "brand": brand, "mix_id": link.get("mix_id"), "mix_code": link.get("mix_code"),
                    "source": source_of(project, entry, link), "posted_at": entry.get("posted_at"),
                    "caption": entry.get("caption"), "caption_source": entry.get("caption_source"),
                    "duration_sec": round(seconds, 1)})
    prints = recordings(user_id, [v["project_id"] for v in out if not v.get("mix_id")], client=client)
    week_ids = {s.get("project_id") for s in (plan.get("week") or {}).get("slots") or []}
    return one_per_recording(out, prints, prefer=week_ids)


# ---------------------------------------------------------------- one cut per recording

FIRST_WORDS = 6


def _word(w: dict) -> str:
    return "".join(ch for ch in str(w.get("text") or w.get("w") or "").lower() if ch.isalnum())


def recording_of(snapshot: dict) -> list[tuple[float, str]]:
    """What a cut was made from: per source, its length and the first words heard in it (from every clip of the
    draft, kept or not). Two cuts of the same recording share it whatever was kept; a new upload of the same
    file too (its storage link is new, its sound is not)."""
    draft = snapshot.get("draft") or {}
    words: dict[int, list[tuple[float, str]]] = {}
    for part in ("selected", "alternates", "discarded"):
        for clip in draft.get(part) or []:
            order = int(clip.get("source_order") or 0)
            for i, w in enumerate(clip.get("words") or []):
                t = w.get("start", w.get("s"))
                t = float(t) if t is not None else float(clip.get("start") or 0) + i * 1e-3
                if _word(w):
                    words.setdefault(order, []).append((t, _word(w)))
    out = []
    for k, src in enumerate(snapshot.get("sources") or []):
        order = int(src.get("source_order", k) or 0)
        heard = sorted(set(words.get(order, [])))
        out.append((round(float(src.get("duration_sec") or 0), 2), " ".join(w for _, w in heard[:FIRST_WORDS])))
    return sorted(out)


def same_recording(a: list[tuple[float, str]], b: list[tuple[float, str]]) -> bool:
    if not a or len(a) != len(b):
        return False
    rest = list(b)
    for dur, text in a:
        match = next((x for x in rest if text and x[1] == text and abs(x[0] - dur) <= 1.0), None)
        if match is None:
            return False
        rest.remove(match)
    return True


def recordings(user_id: str, project_ids: list[str], *, client=None) -> dict[str, list]:
    if not project_ids:
        return {}
    from . import draft_store
    target = draft_store._redis_client(client)
    raw = target.mget([draft_store.draft_key(user_id=user_id, project_id=pid) for pid in project_ids])
    out = {}
    for pid, item in zip(project_ids, raw):
        try:
            out[pid] = recording_of(json.loads(item.decode() if isinstance(item, bytes) else item)) if item else []
        except Exception:
            out[pid] = []
    return out


def one_per_recording(videos: list[dict], prints: dict[str, list], *, prefer: set | None = None) -> list[dict]:
    """The same recording cut more than once is planned once: a posted cut wins (the recording was posted), then the
    cut already in this week, then the newest cut. Mix videos are separate videos (their own rules)."""
    prefer = prefer or set()
    groups: list[list[dict]] = []
    for v in videos:
        fp = prints.get(v["project_id"])
        if v.get("mix_id") or not fp:
            groups.append([v])
            continue
        home = next((g for g in groups if not g[0].get("mix_id") and same_recording(prints.get(g[0]["project_id"]) or [], fp)),
                    None)
        if home is None:
            groups.append([v])
        else:
            home.append(v)
    out = []
    for g in groups:
        best = max(g, key=lambda v: (bool(v.get("posted_at")), v["project_id"] in prefer, v.get("created_at") or "",
                                     v["project_id"]))
        if len(g) > 1:
            best = {**best, "other_cuts": len(g) - 1}
        out.append(best)
    keep = {v["project_id"] for v in out}
    return [next(x for x in out if x["project_id"] == v["project_id"]) for v in videos if v["project_id"] in keep]


# ---------------------------------------------------------------- building the week (rules, no AI)

def week_start(today: date) -> date:
    return today - timedelta(days=today.weekday())


def slot_times(prefs: dict) -> list[str]:
    """Clock times of the day's slots: her reminders that are on, then the usual times, earliest first."""
    n = max(1, min(MAX_PER_DAY, int(prefs.get("per_day") or 1)))
    times = sorted({r["time"] for r in prefs.get("reminders") or [] if r.get("on")})
    for extra in EXTRA_TIMES:
        if len(times) >= n:
            break
        if extra not in times:
            times.append(extra)
    return sorted(times[:n]) if len(times) >= n else sorted(times)


def _product_key(video: dict) -> str:
    return video.get("product_id") or ("video:" + video["project_id"])


def _hook(video: dict) -> Optional[str]:
    code = video.get("mix_code") or ""
    return (video.get("mix_id") or "") + ":" + code.split(" · ")[0] if video.get("mix_id") and code else None


def build(videos: list[dict], prefs: dict, today: date, *, keep: list[dict] | None = None) -> dict:
    """The week of `today` (Monday to Sunday). Slots before today and posted / pinned slots in `keep` stay.
    -> {"start", "slots": [{date, time, project_id, pinned}], "missing": {...}}"""
    start = week_start(today)
    times = slot_times(prefs)
    active = sorted({int(d) for d in prefs.get("days") or []})
    by_id = {v["project_id"]: v for v in videos}
    kept = []
    for slot in keep or []:
        d = date.fromisoformat(slot["date"])
        video = by_id.get(slot.get("project_id") or "")
        if d < today or slot.get("pinned") or (video and video.get("posted_at")):
            if start <= d <= start + timedelta(days=6):
                kept.append(dict(slot))
    taken = {s["project_id"] for s in kept if s.get("project_id")}
    pool = sorted((v for v in videos if not v.get("posted_at") and v["project_id"] not in taken),
                  key=lambda v: (v.get("created_at") or "", v["project_id"]))
    days = [start + timedelta(days=i) for i in range(7)]
    open_slots = []
    for d in days:
        if d < today or d.weekday() not in active:
            continue
        used_times = {s["time"] for s in kept if s["date"] == d.isoformat()}
        for t in times:
            if t not in used_times:
                open_slots.append((d, t))
    slots = list(kept)

    def neighbours(d: date, t: str):
        ordered = sorted(slots, key=lambda s: (s["date"], s["time"]))
        before = [s for s in ordered if (s["date"], s["time"]) < (d.isoformat(), t)]
        after = [s for s in ordered if (s["date"], s["time"]) > (d.isoformat(), t)]
        return (before[-1] if before else None), (after[0] if after else None)

    def allowed(v: dict, d: date, t: str) -> bool:
        prev, nxt = neighbours(d, t)
        for other in (prev, nxt):
            if other and other.get("project_id") in by_id and _product_key(by_id[other["project_id"]]) == _product_key(v):
                return False
        if v.get("mix_id"):
            for s in slots:
                o = by_id.get(s.get("project_id") or "")
                if not o or o.get("mix_id") != v["mix_id"]:
                    continue
                gap = abs((date.fromisoformat(s["date"]) - d).days)
                if gap == 0:
                    return False                     # one video of a Mix per day
                if gap == 1 and _hook(o) == _hook(v):
                    return False                     # the same hook never on two days in a row
        return True

    ordered_slots = sorted(open_slots)
    for index, (d, t) in enumerate(ordered_slots):
        options = [v for v in pool if allowed(v, d, t)]
        # Oldest first -- unless one product has so many videos left that they could no longer be spread out
        # (the same product never twice in a row): that product goes now, its oldest video first.
        counts: dict[str, int] = {}
        for v in pool:
            counts[_product_key(v)] = counts.get(_product_key(v), 0) + 1
        crowded = [k for k, n in counts.items() if n > 1 and n >= len(pool) - n]
        urgent = [v for v in options if _product_key(v) in crowded]
        choice = (urgent or options or [None])[0]
        if choice:
            pool.remove(choice)
            slots.append({"date": d.isoformat(), "time": t, "project_id": choice["project_id"], "pinned": False})
    slots.sort(key=lambda s: (s["date"], s["time"]))
    return {"start": start.isoformat(), "slots": slots, "missing": shortfall(slots, prefs, today)}


def shortfall(slots: list[dict], prefs: dict, today: date) -> dict:
    """Days from today with fewer videos than she posts per day, and how many videos would fill them."""
    start = week_start(today)
    per_day = len(slot_times(prefs))
    active = {int(d) for d in prefs.get("days") or []}
    planned = [start + timedelta(days=i) for i in range(7)]
    planned = [d for d in planned if d >= today and d.weekday() in active]
    count = lambda d: sum(1 for s in slots if s["date"] == d.isoformat())
    short = [d for d in planned if count(d) < per_day]
    return {"days": [d.isoformat() for d in short], "videos": sum(per_day - count(d) for d in short),
            "covered_days": len(planned) - len(short), "total_days": len(planned)}


def check(slots: list[dict], videos: list[dict]) -> list[str]:
    """The rules, checked on a finished week (used by the tests and the report)."""
    by_id = {v["project_id"]: v for v in videos}
    problems = []
    ordered = [s for s in sorted(slots, key=lambda s: (s["date"], s["time"])) if s.get("project_id") in by_id]
    for a, b in zip(ordered, ordered[1:]):
        if _product_key(by_id[a["project_id"]]) == _product_key(by_id[b["project_id"]]):
            problems.append(f"same product in a row: {a['date']} {a['time']} / {b['date']} {b['time']}")
    for a in ordered:
        for b in ordered:
            va, vb = by_id[a["project_id"]], by_id[b["project_id"]]
            if a is b or not va.get("mix_id") or va.get("mix_id") != vb.get("mix_id"):
                continue
            gap = (date.fromisoformat(b["date"]) - date.fromisoformat(a["date"])).days
            if gap == 0 and a["time"] < b["time"]:
                problems.append(f"two videos of one Mix on {a['date']}")
            if gap == 1 and _hook(va) == _hook(vb):
                problems.append(f"same hook on {a['date']} and {b['date']}")
    return problems


def today_for(prefs: dict, now: Optional[datetime] = None) -> date:
    return (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(prefs.get("tz") or "UTC")).date()


def build_week(user_id: str, projects: list[dict], *, client=None, now: Optional[datetime] = None) -> dict:
    plan = load(user_id, client=client)
    videos = candidates(user_id, projects, client=client, plan=plan)
    today = today_for(plan["prefs"], now)
    previous = (plan.get("week") or {})
    keep = previous.get("slots") if previous.get("start") == week_start(today).isoformat() else []
    week = build(videos, plan["prefs"], today, keep=keep)
    week["built_at"] = _now()

    def change(data):
        data["week"] = week
        for slot in week["slots"]:
            data["videos"].setdefault(slot["project_id"], {})["scheduled_for"] = f"{slot['date']}T{slot['time']}"
    update(user_id, change, client=client)
    return week


def move(user_id: str, project_id: str, *, day: str, time: str, client=None) -> dict:
    date.fromisoformat(day)
    hh, mm = time.split(":")[:2]
    time = f"{int(hh) % 24:02d}:{int(mm) % 60:02d}"

    def change(data):
        week = data.get("week") or {}
        slots = [s for s in week.get("slots") or [] if s.get("project_id") != project_id]
        slots.append({"date": day, "time": time, "project_id": project_id, "pinned": True})
        week["slots"] = sorted(slots, key=lambda s: (s["date"], s["time"]))
        data["week"] = week
        data["videos"].setdefault(project_id, {})["scheduled_for"] = f"{day}T{time}"
        return week
    return update(user_id, change, client=client)


def remove(user_id: str, project_id: str, *, client=None) -> dict:
    def change(data):
        week = data.get("week") or {}
        week["slots"] = [s for s in week.get("slots") or [] if s.get("project_id") != project_id]
        data["week"] = week
        data["videos"].setdefault(project_id, {})["scheduled_for"] = None
        return week
    return update(user_id, change, client=client)


def view(user_id: str, projects: list[dict], *, client=None, now: Optional[datetime] = None) -> dict:
    """What the app shows: preferences, the summary for Goal, the week with each video's product, Mix code,
    caption and posted state, and the days left short."""
    plan = load(user_id, client=client)
    videos = candidates(user_id, projects, client=client, plan=plan)
    by_id = {v["project_id"]: v for v in videos}
    today = today_for(plan["prefs"], now)
    unposted = [v for v in videos if not v.get("posted_at")]
    products = {_product_key(v) for v in unposted}
    per_day = len(slot_times(plan["prefs"]))
    week = plan.get("week") if (plan.get("week") or {}).get("start") == week_start(today).isoformat() else None
    slots = []
    for slot in (week or {}).get("slots") or []:
        v = by_id.get(slot.get("project_id") or "")
        if v:
            slots.append({**slot, **{k: v.get(k) for k in ("product", "brand", "product_id", "mix_id", "mix_code",
                                                         "source", "posted_at", "caption", "caption_source",
                                                         "duration_sec", "title")}})
    missing = shortfall(slots, plan["prefs"], today) if week else None
    return {"prefs": plan["prefs"], "today": today.isoformat(),
            "summary": {"ready": len(unposted), "products": len(products), "per_day": per_day,
                        "covers_days": len(unposted) // max(1, per_day)},
            "week": {"start": week["start"], "built_at": week.get("built_at"), "slots": slots, "missing": missing}
            if week else None}
