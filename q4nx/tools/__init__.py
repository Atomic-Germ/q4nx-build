"""Conversion workflow tooling: verify, compare, and diagnose Q4NX output.

These tools operate on the output of ``convert.py`` (a directory containing
``model.q4nx`` and/or ``vision_weight.q4nx``/``audio_weight.q4nx``, plus
``config.json`` and tokenizer assets). They are exposed through the
``q4nx-build`` console script (see ``q4nx.tools.cli``).
"""
