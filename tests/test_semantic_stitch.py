from worker.semantic_visual_pass import Take, dedup_takes, tag_slot, score_take, stitch_chain

raw = [
    Take(id="A", start=0.0,  end=7.0,  text="Okay, so this new tallow cream is really amazing for dry skin.", face_q=0.95, scene_q=0.92, vtx_sim=0.78),
    Take(id="B", start=7.0,  end=12.0, text="uh, wait no, I said that wrong — let me start again…", face_q=0.60, scene_q=0.60, vtx_sim=0.10),
    Take(id="C", start=12.0, end=22.0, text="This tallow cream hydrates deep into the skin barrier and keeps it glowing for hours.", face_q=0.96, scene_q=0.93, vtx_sim=0.80),
    Take(id="E", start=31.0, end=40.0, text="The cocoa scent makes it feel like dessert but still super lightweight.", face_q=0.95, scene_q=0.92, vtx_sim=0.79),
    Take(id="F", start=40.0, end=60.0, text="I use it every morning before makeup and it makes my skin look radiant all day.", face_q=0.96, scene_q=0.94, vtx_sim=0.80),
]

print("Input takes:", [t.id for t in raw])
clean = dedup_takes(raw)
print("After dedup:", [t.id for t in clean])
for t in clean:
    print(f"Tag {t.id}: {tag_slot(t)}  |  Score: {score_take(t)}")

merged = stitch_chain(clean)
print("Merged chains:")
for m in merged:
    print(m.id, m.start, m.end, m.text[:40], "...")
