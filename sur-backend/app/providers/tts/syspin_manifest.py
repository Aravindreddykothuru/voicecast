"""Pinned SYSPIN voice releases: one commit and the sha256 of every file used.

Every SYSPIN voice is loaded at an exact revision and each file it needs is
hashed before use. A release that changes upstream, a truncated download, or
a swapped file fails loudly at load instead of producing different speech
with nothing to show why. Nothing here imports torch or huggingface_hub, so
the API process can read it (see /api/capabilities, tts_voice_warnings).

extra.py is the Coqui tokenizer/config code SYSPIN ships beside each voice.
It is byte-identical in every release that includes it -- the language-
specific parts are chars.txt and the weights -- so its hash is pinned once.
One release, BengaliFemale, was published without it (issue #6); that voice
borrows the identical file from a sibling release, and says so.

To add or bump a voice: pin the new revision here with every file's sha256,
or the voice refuses to load. That is the point.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class _Pin:
    revision: str
    files: dict[str, str] = field(default_factory=dict)   # filename -> sha256
    ships_extra_py: bool = True


# The one extra.py every complete release ships (9 of 9 hash identically).
EXTRA_PY_SHA256 = "3ac9a2fd0cadc470fa1b3af0f89744f2e2aa02e35589752e3a6870abe090570c"

# Releases to borrow extra.py from, in order, when a voice ships without it.
EXTRA_PY_DONORS = (
    "SYSPIN/tts_vits_coquiai_TeluguFemale",
    "SYSPIN/tts_vits_coquiai_HindiFemale",
    "SYSPIN/tts_vits_coquiai_BengaliMale",
)

SYSPIN_MANIFEST: dict[str, _Pin] = {
    "SYSPIN/tts_vits_coquiai_TeluguFemale": _Pin(
        revision="dbbb16a36333750ef37229044c2ec5e7675aa0c0",
        files={
            "chars.txt": "8b9260f2acc6a129a4d8f49af37cf045ebf0edee45ee735be61a8db83b4948a5",
            "te_female_vits_30hrs.pt": "07ed79f7b1bf93d9759e13fbbbe1e333082724f2ee5cb29cdf94f86b45e298b5",
        },
        ships_extra_py=True,
    ),
    "SYSPIN/tts_vits_coquiai_TeluguMale": _Pin(
        revision="7ec0a2bfa40c635a5961a069646913564032cc4f",
        files={
            "chars.txt": "4bb505a4c0b995ab6f05c595e8997d8ceef67ef78734b07fc696626b5cbcff1d",
            "te_male_vits_30hrs.pt": "1204c5f1296cd606625cefe977409405e7201e2c87b9c7c50535b2966216cfe0",
        },
        ships_extra_py=True,
    ),
    "SYSPIN/tts_vits_coquiai_HindiFemale": _Pin(
        revision="2cddbc1f409b7b5069cd2fa3602768158088254c",
        files={
            "chars.txt": "fc4f2c4f42095971e8c9a615149201ddfcb3e4383786ab447916a0844f7ccf3d",
            "hi_female_vits_30hrs.pt": "2bcfb47f599b36e7cbfec27142604c366e538c17e89980a40519291f92a46327",
        },
        ships_extra_py=True,
    ),
    "SYSPIN/tts_vits_coquiai_HindiMale": _Pin(
        revision="866b03fe16c2c998fc649753e11e1896e1f9a382",
        files={
            "chars.txt": "dc787a7793c78c68cb7d795aaf460adf8304dba2095757b7c80fc4df6ad3592a",
            "hi_male_vits_30hrs.pt": "eb36eca2d90214662f1647e83eb6979ead93b72f269606c6411f52959acf77a8",
        },
        ships_extra_py=True,
    ),
    "SYSPIN/tts_vits_coquiai_KannadaFemale": _Pin(
        revision="19f4edcda5c36318e3606a2ca20993f27441c728",
        files={
            "chars.txt": "0941599aa91df447963b33ce48e2111fff1d6cedaad62ca6a4c57d9898803f66",
            "kn_female_vits_30hrs.pt": "49be422a46afc7714a8ea1cab589d986c3bc61939faa5f1d5d6f9f80a263c51c",
        },
        ships_extra_py=True,
    ),
    "SYSPIN/tts_vits_coquiai_KannadaMale": _Pin(
        revision="9c88e05883ec8f52ecb7823d6967c69491f4f152",
        files={
            "chars.txt": "e37962426bb6bb0ad57e1cb371ff1db9d766e9f14bc86df11525dec7418b1917",
            "kn_male_vits_30hrs.pt": "b2a7e16b3509df1be518e0616bcb3ce6eb9c4e59d7fafed075ee57426befdef9",
        },
        ships_extra_py=True,
    ),
    "SYSPIN/tts_vits_coquiai_MarathiFemale": _Pin(
        revision="04512ff4b9d5058cba410939b6531992af6a97b4",
        files={
            "chars.txt": "372c5f2e592468cfa0c5e513adf5c4d6aba2454a11d6d18f88bd0de0921a68c0",
            "mr_female_vits_30hrs.pt": "5da210ece20c171ee09f8969d1755fb475a43c2c3166a4a088f36b9aa828dbb7",
        },
        ships_extra_py=True,
    ),
    "SYSPIN/tts_vits_coquiai_MarathiMale": _Pin(
        revision="d4e65e961d3f36dc3932042591e94d9ca255a944",
        files={
            "chars.txt": "2d0a0a8b14a4b4151f4954b771a191b5a1d91fa045627e69f9d6878f63b947f3",
            "mr_male_vits_30hrs.pt": "8de04cf53233c6018c4a1d86cad6bbabc61b14fe8601128a0b6a0ca9573d8d64",
        },
        ships_extra_py=True,
    ),
    "SYSPIN/tts_vits_coquiai_BengaliFemale": _Pin(
        revision="4aa58c497f71a19abe906a5845eb66930b40ceb2",
        files={
            "chars.txt": "d95857355b5e6ee95fb219e07efdef2abf63791338da57d401f909637140f4dd",
            "bn_female_vits_30hrs.pt": "53208e056050bb485df9192a0d444d3fa72eefe15b2c04840e9a500e4ac1bbf4",
        },
        ships_extra_py=False,
    ),
    "SYSPIN/tts_vits_coquiai_BengaliMale": _Pin(
        revision="93d425c03945d7c05192bf40dc6edb38ea881aec",
        files={
            "chars.txt": "2f9f01f53b17e777945baad65a4cb7301cfc1ae3e39cdb1be78838928e58af63",
            "bn_male_vits_30hrs.pt": "c9d8d52f0bc33ef01d733eef36fb00f1e17192b8c86123a0ccf84a24dbb80d0e",
        },
        ships_extra_py=True,
    ),
}


def pin_for(repo_id: str) -> _Pin:
    try:
        return SYSPIN_MANIFEST[repo_id]
    except KeyError:
        raise KeyError(
            f"{repo_id} is not pinned in app/providers/tts/syspin_manifest.py. Add its revision and the "
            f"sha256 of chars.txt and its weights before using it -- unpinned voices do not load."
        ) from None


def supply_chain_warnings(repo_ids) -> list[str]:
    """Human-readable problems with the pinned releases of these voices."""
    out = []
    for repo in sorted(set(repo_ids)):
        pin = SYSPIN_MANIFEST.get(repo)
        if pin is None:
            out.append(f"{repo}: not pinned -- it will refuse to load")
        elif not pin.ships_extra_py:
            out.append(f"{repo}: release ships no extra.py; the byte-identical file is borrowed from "
                       f"{EXTRA_PY_DONORS[0]} (sha256 {EXTRA_PY_SHA256[:12]}...). Upstream fix pending.")
    return out
