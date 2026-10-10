"""The species a visitor can pick: which fine-tuned checkpoints and which SAM 3 prompts go with each.

Every option names a ``conf/weights.yaml`` dataset with a default-set LoMa and RDD-LightGlue checkpoint.
Prompts follow the masks the registry entries were built with (``wildmatch prepare``): the species name
first, then the entry's ``retry_prompts``, then "Animal"; ``merge`` is the entry's ``registry.prepare.merge``
(``largest`` keeps one animal per photo, ``union`` joins the pieces of a salamander split by a finger).
CzechLynx ships its own masks, so its prompts ("lynx", then "big cat" as for Leopard) are the Space's choice.
``tests/test_space.py`` checks this table against the registry and ``weights.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Species:
    label: str
    dataset: str  # conf/weights.yaml dataset key
    prompts: Tuple[str, ...]  # SAM 3 text prompts, tried in order until one detects something
    merge: str  # how several SAM 3 detections become one mask


SPECIES: Dict[str, Species] = {
    s.label: s
    for s in (
        Species("Eurasian lynx", "czechlynx_closed", ("lynx", "big cat", "Animal"), "largest"),
        Species("Eurasian lynx (unseen individuals)", "czechlynx_open", ("lynx", "big cat", "Animal"), "largest"),
        Species("Leopard", "leopardid2022", ("leopard", "big cat", "Animal"), "largest"),
        Species("Spotted hyena", "hyenaid2022", ("hyena", "dog", "Animal"), "largest"),
        Species("Nyala", "nyala", ("nyala", "antelope", "deer", "Animal"), "largest"),
        Species("Whale shark", "whaleshark", ("whale shark", "shark", "fish", "Animal"), "largest"),
        Species("Sea turtle", "zindi", ("sea turtle", "turtle", "Animal"), "largest"),
        Species("Sea star", "seastarreid2023", ("sea star", "starfish", "Animal"), "largest"),
        Species("Fire salamander", "salamander", ("Salamander", "Animal"), "union"),
    )
}
DEFAULT_SPECIES = "Eurasian lynx"
MATCHERS = {"LoMa": "loma", "RDD-LightGlue": "rdd-lightglue"}
