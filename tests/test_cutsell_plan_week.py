"""Plan my week: the week built with rules from Ready videos of the four origins, moving / posting / rebuilding,
and the posting caption written once (fakeredis, no network)."""
import json
from datetime import date, datetime, timezone

import fakeredis
import pytest

from cutsell_worker import draft_store, post_captions, post_plan, product_catalog, project_store


@pytest.fixture()
def r(monkeypatch):
    client = fakeredis.FakeRedis()
    for module in (project_store, draft_store, post_plan, product_catalog):
        monkeypatch.setattr(module, "_redis_client", lambda c=None, _r=client: _r)
    return client


MONDAY = date(2026, 10, 19)


def _video(user, title, *, sources=1, state="draft_ready"):
    p = project_store.create_project(user_id=user, title=title)
    project_store.update_project(user_id=user, project_id=p["project_id"], state=state,
                                 sources=[{"source_asset_id": f"s{i}", "duration_sec": 30} for i in range(sources)])
    return p["project_id"]


def _library(user="u1"):
    """Ready videos of the four origins: 2 single (CeraVe), 1 combine (Dior), 2 bulk (Old Navy, Rhino), a Mix of 4
    (Glow), and one still processing (never planned)."""
    ids = {}
    cerave = product_catalog.create(user, brand="CeraVe", product="Hydrating Cleanser")["product_id"]
    dior = product_catalog.create(user, brand="Dior", product="Lip Glow")["product_id"]
    navy = product_catalog.create(user, brand="Old Navy", product="Denim Jacket")["product_id"]
    rhino = product_catalog.create(user, brand="Rhino", product="Power Bank")["product_id"]
    glow = product_catalog.create(user, brand="Glow Recipe", product="Watermelon Toner")["product_id"]
    ids["single1"], ids["single2"] = _video(user, "a"), _video(user, "b")
    product_catalog.assign(user, [ids["single1"], ids["single2"]], product_id=cerave)
    ids["combine"] = _video(user, "c", sources=3)
    product_catalog.assign(user, [ids["combine"]], product_id=dior)
    for name, product in (("bulk1", navy), ("bulk2", rhino)):
        ids[name] = _video(user, name)
        product_catalog.assign(user, [ids[name]], product_id=product)
        post_plan.record_source(user, ids[name], "bulk")
    codes = ["H1 · B1 · C1", "H2 · B2 · C2", "H1 · B2 · C1", "H2 · B1 · C2"]
    mix = {}
    for k, code in enumerate(codes):
        mix[_video(user, f"mix{k}")] = code
    product_catalog.place_mix(user, "mix_1", mix, product_id=glow)
    ids["mix"] = list(mix)
    ids["processing"] = _video(user, "p", state="processing")
    return ids


def _projects(user="u1"):
    return project_store.list_projects(user_id=user, limit=100)


def test_the_week_follows_the_rules_with_videos_of_every_origin(r):
    ids = _library()
    videos = post_plan.candidates("u1", _projects())
    assert {v["source"] for v in videos} == {"single", "combine", "bulk", "mix"}
    assert ids["processing"] not in {v["project_id"] for v in videos}           # only Ready videos
    prefs = {**post_plan.DEFAULT_PREFS, "per_day": 2, "days": [0, 1, 2, 3, 4, 5]}
    week = post_plan.build(videos, prefs, MONDAY)
    assert len(week["slots"]) == 9                                               # every Ready video, once
    assert post_plan.check(week["slots"], videos) == []
    assert len({s["project_id"] for s in week["slots"]}) == 9
    mix_days = [s["date"] for s in week["slots"] if s["project_id"] in ids["mix"]]
    assert len(set(mix_days)) == 4                                               # one Mix video a day
    assert week["missing"]["videos"] == 12 - 9 and week["missing"]["total_days"] == 6
    assert [s["time"] for s in week["slots"][:2]] == ["09:00", "19:00"]


def test_oldest_first_and_never_the_same_product_in_a_row(r):
    user = "u2"
    cerave = product_catalog.create(user, brand="CeraVe", product="Cleanser")["product_id"]
    dior = product_catalog.create(user, brand="Dior", product="Glow")["product_id"]
    old = [_video(user, f"c{i}") for i in range(4)]
    product_catalog.assign(user, old, product_id=cerave)
    one_dior = _video(user, "d")
    product_catalog.assign(user, [one_dior], product_id=dior)
    videos = post_plan.candidates(user, _projects(user))
    week = post_plan.build(videos, {**post_plan.DEFAULT_PREFS, "per_day": 2, "days": [0, 1]}, MONDAY)
    assert post_plan.check(week["slots"], videos) == []
    ordered = [s["project_id"] for s in week["slots"]]
    assert ordered[0] == old[0] and one_dior in ordered                          # oldest first
    assert week["missing"]["videos"] == 4 - len(week["slots"])                   # CeraVe can't fill every slot


def test_move_post_and_rebuild_keep_her_choices(r):
    ids = _library()
    now = datetime(2026, 10, 19, 14, tzinfo=timezone.utc)                         # Monday 10 AM in New York
    week = post_plan.build_week("u1", _projects(), now=now)
    first = week["slots"][0]["project_id"]
    post_plan.mark_posted("u1", first)
    moved = week["slots"][3]["project_id"]
    post_plan.move("u1", moved, day="2026-10-24", time="20:30")
    week = post_plan.build_week("u1", _projects(), now=now)
    slots = {s["project_id"]: s for s in week["slots"]}
    assert slots[first]["date"] == "2026-10-19"                                   # posted: stays
    assert slots[moved]["date"] == "2026-10-24" and slots[moved]["time"] == "20:30"   # moved by her: stays
    videos = post_plan.candidates("u1", _projects())
    assert post_plan.check(week["slots"], videos) == []
    view = post_plan.view("u1", _projects(), now=now)
    assert view["week"]["slots"][0]["posted_at"] and view["summary"]["ready"] == 8
    post_plan.remove("u1", moved)
    assert moved not in {s["project_id"] for s in post_plan.view("u1", _projects(), now=now)["week"]["slots"]}


def test_slot_times_follow_her_reminders():
    prefs = {"per_day": 3, "reminders": [{"time": "08:30", "on": True}, {"time": "22:00", "on": False}]}
    assert post_plan.slot_times(prefs) == ["08:30", "09:00", "13:00"]
    assert post_plan.slot_times({"per_day": 1, "reminders": [{"time": "19:00", "on": True}]}) == ["19:00"]


# ---------------------------------------------------------------- captions

def _answer(caption, tags):
    return lambda body: ({"caption": caption, "hashtags": tags}, {"input_tokens": 400, "output_tokens": 120}, None)


def test_the_caption_is_written_once_and_hers_is_never_replaced(r):
    ids = _library()
    pid = ids["single1"]
    draft = {"selected": [{"source_order": 0, "start": 0, "words": [{"text": w} for w in "this cleanser is so gentle".split()]}]}
    out = post_captions.caption_once(user_id="u1", project_id=pid, draft=draft,
                                     call=_answer("So gentle on my skin ✨", ["#CeraVe", "#skincare"]))
    assert out["status"] == "written" and out["log"]["cost_usd"] == pytest.approx((400 * 0.10 + 120 * 0.50) / 1e6)
    entry = post_plan.video("u1", pid)
    assert entry["caption"] == "So gentle on my skin ✨ #cerave #skincare #tiktokshop" and entry["caption_source"] == "ai"
    again = post_captions.caption_once(user_id="u1", project_id=pid, draft=draft, call=_answer("x", []))
    assert again["status"] == "exists"
    post_plan.set_caption("u1", pid, "My own words #mine", source="user")
    post_plan.set_caption("u1", pid, "AI again", source="ai")
    assert post_plan.video("u1", pid)["caption"] == "My own words #mine"


def test_no_price_or_offer_the_video_does_not_say():
    said = "this jacket goes with everything"
    assert post_captions.clean("Goes with everything! Now 50% off, grab it.", ["#oldnavy"], said) == \
        "Goes with everything! #oldnavy #tiktokshop"
    said_sale = "it is fifty percent off right now"
    assert "off" in post_captions.clean("Fifty percent off right now.", [], said_sale)


def test_the_api_builds_moves_and_posts(monkeypatch, r):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from cutsell_app import plan_routes
    monkeypatch.setattr(plan_routes, "_caption_missing", lambda user, view: None)
    _library()
    app = FastAPI()
    app.include_router(plan_routes.router)
    client = TestClient(app)
    view = client.put("/v1/plan/prefs", json={"user_id": "u1", "per_day": 2, "days": [0, 1, 2, 3, 4, 5, 6],
                                              "tz": "America/New_York"}).json()
    assert view["summary"]["ready"] == 9 and view["week"] is None
    week = client.post("/v1/plan/week/build", json={"user_id": "u1"}).json()["week"]
    assert week["slots"] and all(s["product"] for s in week["slots"])
    pid = week["slots"][-1]["project_id"]
    moved = client.post("/v1/plan/slots/move", json={"user_id": "u1", "project_id": pid, "date": week["slots"][0]["date"],
                                                     "time": "23:00"}).json()
    assert any(s["project_id"] == pid and s["time"] == "23:00" for s in moved["week"]["slots"])
    posted = client.post("/v1/plan/posted", json={"user_id": "u1", "project_id": pid}).json()
    assert next(s for s in posted["week"]["slots"] if s["project_id"] == pid)["posted_at"]
    assert client.put("/v1/plan/caption", json={"user_id": "u1", "project_id": pid, "caption": "Mine"}).status_code == 200
    assert client.post("/v1/plan/posted", json={"user_id": "u1", "project_id": "nope"}).status_code == 404


def test_the_rules_hold_on_many_random_libraries():
    import random
    rng = random.Random(7)
    for trial in range(300):
        videos = []
        for k in range(rng.randint(0, 16)):
            product = f"p{rng.randint(0, 4)}"
            mix = f"m{rng.randint(0, 1)}" if rng.random() < 0.4 else None
            videos.append({"project_id": f"v{k}", "product_id": product, "created_at": f"2026-10-{rng.randint(1, 18):02d}",
                           "mix_id": mix, "mix_code": f"H{rng.randint(1, 3)} · B1 · C1" if mix else None,
                           "posted_at": None})
        prefs = {"per_day": rng.randint(1, 4), "days": rng.sample(range(7), rng.randint(1, 7)),
                 "reminders": [{"time": "09:00", "on": True}]}
        today = MONDAY + __import__("datetime").timedelta(days=rng.randint(0, 6))
        week = post_plan.build(videos, prefs, today)
        assert post_plan.check(week["slots"], videos) == [], trial
        assert len({s["project_id"] for s in week["slots"]}) == len(week["slots"])


# ---------------------------------------------------------------- one cut per recording

def _cut_of(r, user, title, words, *, duration=106.3, keep=None, created=None):
    """A Ready cut of one recording: the draft holds every clip heard (kept or not); `keep` = which clips are kept."""
    from cutsell_worker import draft_store
    pid = _video(user, title)
    if created:
        key = project_store.project_key(user_id=user, project_id=pid)
        record = json.loads(r.get(key))
        r.set(key, json.dumps({**record, "created_at": created}))
    clips = [{"clip_id": f"c{i}", "source_order": 0, "start": i * 2.0, "end": i * 2.0 + 1.8,
              "words": [{"text": w, "start": i * 2.0 + k * 0.3, "end": i * 2.0 + k * 0.3 + 0.25}
                        for k, w in enumerate(sentence.split())]}
             for i, sentence in enumerate(words)]
    kept = set(range(len(clips))) if keep is None else set(keep)
    snap = {"draft": {"selected": [c for i, c in enumerate(clips) if i in kept],
                      "discarded": [c for i, c in enumerate(clips) if i not in kept]},
            "sources": [{"source_order": 0, "duration_sec": duration, "uri": f"s3://b/{title}.mov"}]}
    r.set(draft_store.draft_key(user_id=user, project_id=pid), json.dumps(snap))
    return pid


PERFUME = ["¿Quién me mandó a comprar esto?", "Si tú tienes problema con los perfumes", "El precio son veinte dólares"]


def test_the_same_recording_cut_three_times_is_planned_once_newest_cut(r):
    user = "u3"
    a = _cut_of(r, user, "p1", PERFUME, keep=[0, 1, 2], created="2026-10-17T10:00:00Z")
    b = _cut_of(r, user, "p2", PERFUME, keep=[1, 2], created="2026-10-18T10:00:00Z", duration=106.1)   # new upload
    c = _cut_of(r, user, "p3", PERFUME, keep=[0, 2], created="2026-10-19T10:00:00Z")
    other = _cut_of(r, user, "w", ["If you have this glass of water", "you should be buying this"], duration=50.0)
    take2 = _cut_of(r, user, "p4", ["Mira, este perfume", "huele a mujer con dinero"], duration=106.3)  # another take
    videos = post_plan.candidates(user, _projects(user))
    ids = {v["project_id"] for v in videos}
    assert ids == {c, other, take2}                                   # the newest cut stands for the recording
    assert next(v for v in videos if v["project_id"] == c)["other_cuts"] == 2
    week = post_plan.build_week(user, _projects(user), now=datetime(2026, 10, 19, 14, tzinfo=timezone.utc))
    assert len(week["slots"]) == 3 and not {a, b} & {s["project_id"] for s in week["slots"]}
    assert post_plan.view(user, _projects(user), now=datetime(2026, 10, 19, 14, tzinfo=timezone.utc))["summary"]["ready"] == 3


def test_a_posted_cut_retires_its_recording_and_the_week_keeps_its_cut(r):
    user = "u4"
    old = _cut_of(r, user, "p1", PERFUME, created="2026-10-17T10:00:00Z")
    new = _cut_of(r, user, "p2", PERFUME, keep=[0, 2], created="2026-10-19T10:00:00Z")
    post_plan.mark_posted(user, old)
    videos = post_plan.candidates(user, _projects(user))
    assert [v["project_id"] for v in videos] == [old] and videos[0]["posted_at"]   # posted: never planned again
    user = "u5"
    old = _cut_of(r, user, "p1", PERFUME, created="2026-10-17T10:00:00Z")
    now = datetime(2026, 10, 19, 14, tzinfo=timezone.utc)
    week = post_plan.build_week(user, _projects(user), now=now)
    assert [s["project_id"] for s in week["slots"]] == [old]
    newer = _cut_of(r, user, "p2", PERFUME, keep=[0], created="2026-10-20T10:00:00Z")
    week = post_plan.build_week(user, _projects(user), now=now)
    assert [s["project_id"] for s in week["slots"]] == [old]          # the cut already planned stays; no twin added


def test_a_combine_of_the_same_two_recordings_in_another_order_is_the_same_recording():
    a = [(64.2, "and if you dont believe me i"), (39.9, "dont buy these old navy henleys")]
    b = [(39.9, "dont buy these old navy henleys"), (64.2, "and if you dont believe me i")]
    assert post_plan.same_recording(sorted(a), sorted(b))
    assert not post_plan.same_recording([(52.0, "pull it latch it")], [(97.5, "pull it latch it")])
    assert not post_plan.same_recording([(50.0, "")], [(50.0, "")])                 # nothing heard: never merged


def test_personal_videos_are_never_planned(r):
    ids = _library()
    personal = ids["single1"]
    post_plan.set_personal("u1", [personal])
    videos = post_plan.candidates("u1", _projects())
    assert personal not in {v["project_id"] for v in videos}
    now = datetime(2026, 10, 19, 14, tzinfo=timezone.utc)
    week = post_plan.build_week("u1", _projects(), now=now)
    assert personal not in {s["project_id"] for s in week["slots"]}
    post_plan.set_personal("u1", [personal], personal=False)
    assert personal in {v["project_id"] for v in post_plan.candidates("u1", _projects())}


def test_the_api_marks_a_video_personal_and_takes_it_out_of_the_week(monkeypatch, r):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from cutsell_app import plan_routes
    monkeypatch.setattr(plan_routes, "_caption_missing", lambda user, view: None)
    _library()
    app = FastAPI()
    app.include_router(plan_routes.router)
    client = TestClient(app)
    week = client.post("/v1/plan/week/build", json={"user_id": "u1"}).json()["week"]
    pid = week["slots"][0]["project_id"]
    view = client.post("/v1/plan/personal", json={"user_id": "u1", "project_id": pid}).json()
    assert pid not in {s["project_id"] for s in view["week"]["slots"]} and view["personal"] == [pid]
    rebuilt = client.post("/v1/plan/week/build", json={"user_id": "u1"}).json()["week"]
    assert pid not in {s["project_id"] for s in rebuilt["slots"]}
    assert client.get("/v1/plan", params={"user_id": "u1"}).json()["personal"] == [pid]
    back = client.post("/v1/plan/personal", json={"user_id": "u1", "project_id": pid, "personal": False}).json()
    assert back["personal"] == []
    assert client.post("/v1/plan/personal", json={"user_id": "u1", "project_id": "nope"}).status_code == 404


# ---------------------------------------------------------------- Goal v2 (flexible)

def test_flex_plan_places_the_count_from_start_on_her_days_and_times(r):
    ids = _library()
    now = datetime(2026, 10, 21, 14, tzinfo=timezone.utc)                       # Wednesday 10 AM in New York
    post_plan.set_prefs("u1", {"times": ["18:45", "08:15", "12:30"], "days": [0, 1, 2, 3, 4], "count": 7,
                               "start": "2026-10-22", "remind": True})
    week = post_plan.build_week("u1", _projects(), now=now)
    assert week["flex"] and week["asked"] == 7 and week["placed"] == 7 and not week["ends_early"]
    dates = sorted({s["date"] for s in week["slots"]})
    assert dates == ["2026-10-22", "2026-10-23", "2026-10-26"]                   # Thu, Fri, (no weekend), Mon
    assert [s["time"] for s in week["slots"][:3]] == ["08:15", "12:30", "18:45"]
    videos = post_plan.candidates("u1", _projects())
    assert post_plan.check(week["slots"], videos) == []
    view = post_plan.view("u1", _projects(), now=now)
    assert view["week"]["end"] == "2026-10-26" and view["prefs"]["per_day"] == 3 and view["prefs"]["remind"] is True
    assert ids["processing"] not in {s["project_id"] for s in week["slots"]}


def test_flex_plan_ends_early_and_says_so_when_the_rules_leave_videos_out(r):
    user = "u6"
    cerave = product_catalog.create(user, brand="CeraVe", product="Cleanser")["product_id"]
    vids = [_video(user, f"c{i}") for i in range(3)]
    product_catalog.assign(user, vids, product_id=cerave)                      # one product only
    post_plan.set_prefs(user, {"times": ["09:00", "13:00"], "days": [0, 1, 2, 3, 4, 5, 6], "count": 3})
    week = post_plan.build_week(user, _projects(user), now=datetime(2026, 10, 21, 14, tzinfo=timezone.utc))
    # one product only: two of its videos would always sit side by side -> 1 placed, the plan says it ends early
    assert week["asked"] == 3 and week["placed"] == 1 and week["ends_early"] is True
    assert post_plan.check(week["slots"], post_plan.candidates(user, _projects(user))) == []


def test_videos_without_a_product_are_counted_as_no_product_not_as_products(r):
    user = "u7"
    cerave = product_catalog.create(user, brand="CeraVe", product="Cleanser")["product_id"]
    tagged = [_video(user, f"c{i}") for i in range(2)]
    product_catalog.assign(user, tagged, product_id=cerave)
    for i in range(3):
        _video(user, f"x{i}")
    summary = post_plan.view(user, _projects(user))["summary"]
    assert summary["ready"] == 5 and summary["products"] == 1 and summary["no_product"] == 3


def test_flex_times_are_cleaned_and_capped_at_eight():
    import pytest as _pytest
    prefs = {"times": ["9:5", "09:05", "23:59"] + [f"1{i}:00" for i in range(9)]}
    assert post_plan.slot_times(prefs) == sorted({"09:05", "23:59"} | {f"1{i}:00" for i in range(9)})[:8]


def test_the_flex_rules_hold_on_many_random_libraries():
    import random
    from datetime import timedelta as _td
    rng = random.Random(11)
    for trial in range(300):
        videos = []
        for k in range(rng.randint(0, 20)):
            product = f"p{rng.randint(0, 5)}" if rng.random() < 0.8 else None
            mix = f"m{rng.randint(0, 1)}" if rng.random() < 0.3 else None
            videos.append({"project_id": f"v{k}", "product_id": product, "created_at": f"2026-10-{rng.randint(1, 18):02d}",
                           "mix_id": mix, "mix_code": f"H{rng.randint(1, 3)} · B1 · C1" if mix else None, "posted_at": None})
        prefs = {"times": [f"{rng.randint(6, 23):02d}:{rng.choice(['00', '15', '30', '45'])}" for _ in range(rng.randint(1, 8))],
                 "days": rng.sample(range(7), rng.randint(1, 7)), "count": rng.randint(1, 25),
                 "start": (MONDAY + _td(days=rng.randint(0, 9))).isoformat()}
        today = MONDAY + _td(days=rng.randint(0, 6))
        week = post_plan.build_flex(videos, prefs, today)
        assert post_plan.check(week["slots"], videos) == [], trial
        assert len({s["project_id"] for s in week["slots"]}) == len(week["slots"])
        assert week["placed"] <= week["asked"] <= max(1, len(videos)) and week["ends_early"] == (week["placed"] < week["asked"])
        assert all(s["date"] >= max(today.isoformat(), prefs["start"]) for s in week["slots"])
        assert all(date.fromisoformat(s["date"]).weekday() in prefs["days"] for s in week["slots"])
        per_day = len(post_plan.slot_times(prefs))
        days = {}
        for s in week["slots"]:
            days[s["date"]] = days.get(s["date"], 0) + 1
        assert all(n <= per_day for n in days.values())
