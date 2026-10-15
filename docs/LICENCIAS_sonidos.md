# CutSell sounds: licenses

The official sounds of the Audio panel ("CutSell sounds"). The audio files are **not** in this repository (it is public and audio is never committed). They live in the Cloudflare R2 media bucket at `assets/sfx/v1/`, next to `manifest.json`, and the app gets them through `GET /v1/sfx/manifest` (signed links).

| File | Menu name | Original | Pack | Author | License | Source |
|---|---|---|---|---|---|---|
| `tap_select.m4a` | Tap | `Audio/switch_005.ogg` | Interface Sounds | Kenney | CC0 1.0 | https://opengameart.org/sites/default/files/kenney_interfaceSounds.zip |
| `good.m4a` | Good | `Audio/confirmation_002.ogg` | Interface Sounds | Kenney | CC0 1.0 | https://opengameart.org/sites/default/files/kenney_interfaceSounds.zip |
| `wrong.m4a` | Wrong | `Audio/error_008.ogg` | Interface Sounds | Kenney | CC0 1.0 | https://opengameart.org/sites/default/files/kenney_interfaceSounds.zip |
| `glass.m4a` | Glass | `Audio/glass_004.ogg` | Interface Sounds | Kenney | CC0 1.0 | https://opengameart.org/sites/default/files/kenney_interfaceSounds.zip |
| `typing_3s.m4a` | Typing | `Human Typing/human_vel-001.wav`, first 3 s, 0.3 s fade out | Keyboard Soundpack 1.0 | unicaegames | CC0 1.0 | https://opengameart.org/sites/default/files/unicae_games_keyboard_soundpack_1_0.zip |
| `typing.m4a` | Typing (when stretched past 3 s) | `Human Typing/human_vel-001.wav`, 9 s | Keyboard Soundpack 1.0 | unicaegames | CC0 1.0 | https://opengameart.org/sites/default/files/unicae_games_keyboard_soundpack_1_0.zip |

## License check (15 Oct 2026)

- Kenney Interface Sounds, `License.txt`: "License: (Creative Commons Zero, CC0)".
- unicaegames Keyboard Soundpack, `readme.txt`: "Free to use however you like." The OpenGameArt page lists it as CC0.

## How the files were made

All files are AAC 160 kb/s, stereo, 44.1 kHz.

- **Loudness:** the target was -16 LUFS, measured with the sound looped (short clicks are shorter than the 0.4 s loudness window).
- **Peak limit:** a limiter at -1.5 dBTP.
- **Results:**
  - Good, Glass and both Typing files are at -16 to -16.5 LUFS.
  - Wrong is at -17.1 LUFS.
  - Tap is at -21.5 LUFS. It is a very short click: louder than that needs heavy limiting that distorts it.
- **Script:** `tools/make_cutsell_sounds.py` (it also writes `manifest.json`).
- **Change from the original recipe:** a single `loudnorm` pass did not reach -16 LUFS on these short, peaky sounds (it gave -21 to -23 LUFS). The gain was therefore measured and then applied.
