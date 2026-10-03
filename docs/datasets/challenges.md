# Data challenges

What to look for: the benchmark photographs were taken by camera traps, divers and
field teams, not for computer vision. Every image below stays in the database and query
sets of all methods; nothing was removed or cleaned.
{ .wm-lead }

The paper's figure shows one or two raw examples per problem. This page shows the
confirmed examples behind it, grouped into two kinds. **Dataset noise** is content no
re-identification method can use: frames without an animal, saturated flash frames,
corrupted or blurred photos, or an animal too small to carry its markings. **Inherent
difficulties** are valid photographs of identifiable animals that look different from
daylight close-ups: night and infrared frames, and animals partly hidden by vegetation or
by the handler's hand. Both groups were found with the image-quality audit described at
the end of the page and confirmed by eye; every photo here is the raw source file, never
the masked model input.

## Dataset noise

### Overexposure

Infrared and flash frames in which the sensor saturates. In CzechLynx the audit finds
1,560 images whose masked foreground is more than 30 % pure white, mostly from one camera
site; the animal's pattern is gone, and the masked input is a nearly uniform white shape.
LeopardID2022 has 162 such frames, Turtle a few flash shots of the shell.

<div id="wm-challenge-overexposure" class="wm-widget wm-challenge" data-category="overexposure">Loading examples…</div>

### Empty frame

Camera-trap frames whose mask segments a stick, a log, a branch or a patch of glare,
while the lynx has already left the frame or never entered it. The CzechLynx masks are
provided with the dataset, so these frames carry a valid-looking mask around no animal.
Several come from one trap that repeated the same frame across a sequence.

<div id="wm-challenge-empty_frame" class="wm-widget wm-challenge" data-category="empty_frame">Loading examples…</div>

### Insufficient detail

The animal is a few dozen pixels at model input size. In ZindiTurtleRecall many such
photos are of the handler and the identification tag, with the turtle incidental, and one
file is stored rotated by 90 degrees without an orientation tag, so models see it
sideways. Whale sharks photographed from the surface or against the light show no spot
pattern.

<div id="wm-challenge-insufficient_detail" class="wm-widget wm-challenge" data-category="insufficient_detail">Loading examples…</div>

### Corruption

Sensor or encoding failures: colour banding and posterised blocks. Rare, and found only in
CzechLynx.

<div id="wm-challenge-corruption" class="wm-widget wm-challenge" data-category="corruption">Loading examples…</div>

### Blur

Motion blur in night frames and out-of-focus flash close-ups. The audit ranks sharpness
within each dataset and flags the bottom 2 %, so the threshold differs per dataset; the
examples here are the ones a human agreed are blurred. Blurred Salamander queries are
rarely retrieved (Top-1 0.02 against 0.33 for the rest, five queries), so blur does cost
accuracy where the pattern is fine-grained.

<div id="wm-challenge-blur" class="wm-widget wm-challenge" data-category="blur">Loading examples…</div>

## Inherent difficulties

### Night and infrared frames

Camera traps record most activity at night: greyscale or purple-cast infrared frames and
very dark colour frames. The animal is identifiable, and flagged Hyena night queries are
not less accurate than the rest, so these are a property of the domain rather than noise.
They are, however, far from the daylight photographs that pretrained matchers were trained
on.

<div id="wm-challenge-night_infrared" class="wm-widget wm-challenge" data-category="night_infrared">Loading examples…</div>

### Occlusion

Vegetation in front of the coat in LeopardID2022 and CzechLynx, and the handler's finger
in SalamanderID2025, where it splits the segmentation mask into several fragments
(62 images). Finger-split Salamander queries are not less accurate (Top-1 0.34 against
0.32), so occlusion is a difficulty the matchers cope with, not harmful noise.

<div id="wm-challenge-occlusion" class="wm-widget wm-challenge" data-category="occlusion">Loading examples…</div>

## Problems visible only in the masks

These are described without images, because a masked input could be read as our error
when the mask came with the dataset or was computed by us:

- **Provider mask failures.** The official SeaStarReID2023 masked files keep only a speck
  of foreground for 64 images whose raw photos are valid full-frame close-ups. WhaleSharkID
  shows the same pattern on 49 images flagged as tiny foreground, most of them sharp spot
  close-ups.
- **Masks on the wrong object.** CzechLynx masks that segment branches, sticks, vegetation
  or a static corner glare repeated across a sequence (34 fragmented masks, and the empty
  frames above).
- **Finger-split masks.** SalamanderID2025 masks computed with SAM 3 split into several
  instances where the handler's finger crosses the body; all instances are merged into one
  mask, so the animal is kept.
- **Dark animals undercounted.** For pre-masked WildlifeReID-10k files the audit measures
  the foreground with a brightness threshold, so very dark animals, such as night hyenas and
  infrared leopards, can be flagged as tiny or empty. Those flags were rejected by eye and
  are not shown above.

## What the audit found

The audit measures the exact model input of every image (pre-masked files, or the raw
photo with the dataset's mask applied) and flags candidates by fixed rules; the flags are
review rankings, not labels. The table also joins each query's Top-1 rate over all
completed runs of all methods, so it shows whether flagged queries are less accurate.

<div id="wm-image-quality-table" class="wm-widget">Loading the audit summary…</div>

<small>Raw photographs from CzechLynx (Picek et al.), WildlifeReID-10k and SalamanderID2025
(AnimalCLEF 2025), the latter shown with the dataset team's permission.</small>

[Back to datasets](index.md){ .md-button }
