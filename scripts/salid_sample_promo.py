"""Render a promo post kit from a product photo, for checking the look by hand.

    uv run python scripts/salid_sample_promo.py <photo> [--live] [--out DIR]
        [--headline TEXT] [--scene TEXT]

Offline (default) the scene is the original-photo fallback: no model, no cost. `--live` makes ONE
real Nano Banana call on Vertex AI (about $0.05, needs ADC; ask before running it). Open the three
JPEGs and check the Thai text, the layout and that the product is unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from salid import cards, promo
from salid.config import Settings
from salid.llm import make_client

SCENE = "wooden cafe table, soft morning window light, a few coffee beans"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path)
    ap.add_argument("--live", action="store_true", help="one real image-model call (costs money)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/salid-promo"))
    ap.add_argument("--headline", default="ลาเต้เย็น หอมกรุ่น ลด 20%")
    ap.add_argument("--scene", default=SCENE)
    args = ap.parse_args(argv)

    photo = promo.prepare_photo(args.photo.read_bytes())
    if photo is None:
        print(f"{args.photo}: not a readable image", file=sys.stderr)
        return 1
    spec = promo.merge_spec(
        None,
        headline=args.headline,
        scene=args.scene,
        price="59.-",
        call_to_action="สั่งเลยวันนี้",
        default_shop="ร้านบ้านกาแฟ",
    )
    scene = None
    if args.live:
        s = Settings()
        print(f"calling {s.promo_image_model} once (live, billed) ...")
        scene, reason = promo.generate_scene(make_client(s), s.promo_image_model, photo, spec.scene)
        print("AI scene" if scene is not None else f"no AI scene ({reason}): using the fallback")
    if scene is None:
        scene = promo.fallback_scene(photo)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in promo.render_kit(scene, spec, cards.card_fonts()):
        path = args.out / name
        path.write_bytes(data)
        print(f"{path} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
