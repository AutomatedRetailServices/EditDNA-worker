# V27 comparison: Yaskira RAW08/09

- Commit: `351639264ad6247738fb68b6889603449cceafe5`
- Run: `36440885494`, artifact `10978832574`, both jobs completed; source RAW09 SHA-256 `463e774293a518eecc4d568099183359f81b2b96381b8444ed63d617ee643381`.
- 08: 25.199 / 27 s rounded keep Gold intersected; 1.801 s missing; rendered QC PASS. Gold is only for 120–147, so material before 120 is not rated extra.
- 09: 58.516 / 73 s rounded keep Gold intersected; 14.484 s missing, 6.720 s outside the rounded keep windows; rendered QC PASS but perceptual assessment does **not** authorize automatic delivery. Selected source spans: 4.98–16.98, 19.00–28.14, 50.33–66.13, 73.35–77.07, 77.07–86.53, 106.66–120.62, 120.62–121.78.

The 09 continuation 73.35–86.53 returned without the focused speech dependency firing: V27's material exception guarded its removal. The new V28 verification remains a safeguard for other source runs where the whole-video selector removes the antecedent.

The 09 action 97–106 remains missing. Full Watch+Listen audience region 50–109.5 described measuring and mixing. Source silence 98.435–106.764 intersects the focused probe. Its observed audience action 97.935–104.935 says, in part, "adds powder from a spoon into the water bottle". V27's action detector listed pour/scoop/mix but did not recognize a description using "adds powder"; hence no draft visual candidate was constructed (`v2_focused_visual_action_candidates=[]`). V28 adds a physical transfer description with source silence + focused AV still mandatory. Later review narrowed the transfer pattern to an action directed into a container/surface and ruled out negated/planned/verbal descriptions.

Outstanding: V28 rendered outcome, short extra BTS tail on 09, optional 09 portion 86.53–91, and repeatability. Do not mark either clip commercially approved based only on physical QC.
