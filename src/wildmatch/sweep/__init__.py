"""Evaluation sweeps: method x checkpoint x candidate-budget grids over registry datasets.

A sweep spec (``conf/sweep/<name>.yaml`` or any YAML file) names registry datasets, candidate
budgets and method rows. :mod:`.spec` turns it into the task table, :mod:`.manifest` freezes
the table, the probe config tree and checkpoint SHA-256 identities into an immutable
submission, and :mod:`.runner` runs one task from that submission (locally or as a Slurm
array element) with per-task log records (:mod:`.logs`).

This replaces the bash launchers ``probe-parallel-wildlife.sh`` and
``probe-parallel-czechlynx.sh`` (2026-10-04). Submissions they created stay runnable: the
manifest schema and task fields are unchanged.
"""
