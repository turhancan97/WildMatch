"""Matcher fine-tuning (slot; filled when ``lynx-finetuning`` is merged in).

Planned interface: given a dataset from the registry and a mined pair index, fine-tune a
matcher with the shared recipe and write epoch directories with a protocol JSON under
``paths.checkpoint_root/<dataset>/<matcher>/<recipe>/epoch_<N>/``, the layout the
evaluation's custom-checkpoint loader already reads.
"""
