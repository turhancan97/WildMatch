"""Pair mining (slot; filled when ``rdd-parallel-benchmark`` is merged in).

Planned interface: given a dataset from the registry and its cached matcher features,
write a pair index (anchors with mined positives and hard negatives, and the pretrained
matcher's scores) under ``paths.output_root``. Matcher fine-tuning reads that index.
"""
