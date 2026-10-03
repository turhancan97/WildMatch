# wm-video environment

Separate conda environment for the explainer video (created 2026-10-03 with the user's
approval, so the shared `ex-reid` evaluation environment stays untouched). Rebuild with:

```bash
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda create -y -n wm-video python=3.12
conda activate wm-video
# ManimPango builds from source and needs pangocairo with its full pkg-config chain.
conda install -y -c conda-forge ffmpeg pango cairo pkg-config harfbuzz fribidi glib fontconfig freetype
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
pip install manim==0.21.0 kokoro==0.9.4 soundfile "misaki[en]"
```

Notes from the first install:

- `espeak-ng` is not on conda-forge under any name; Kokoro's phonemiser (`misaki`) ships its
  own espeak-ng library, so no system package is needed.
- Without `harfbuzz` the `pangocairo` pkg-config check fails and `manimpango` refuses to build.
- Kokoro downloads the `hexgrad/Kokoro-82M` weights and the spaCy `en_core_web_sm` model on
  first use (about 330 MB and 13 MB); both land in the user's Hugging Face and pip caches.
- Smoke test (2026-10-03): `manim -ql` rendered a two-second scene; Kokoro voice `af_heart` at
  `speed=0.95` spoke a 12-word sentence in 5.4 s (134 words per minute).

Rendering uses the CPU only; a 1080p 30 fps render of the full explainer takes minutes, not
hours.
