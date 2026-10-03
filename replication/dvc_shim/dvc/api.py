"""Minimal stand-in for dvc.api.params_show, so METR's stages run without installing DVC.

DVC merges params.yaml with the extra param files declared in dvc.yaml (here fig_params/figs.yaml);
stage filtering is irrelevant because callers index into the keys they need.
"""

import pathlib

import yaml


def params_show(*_args, **_kwargs):
    params = {}
    for f in ["params.yaml", "fig_params/figs.yaml"]:
        p = pathlib.Path.cwd() / f
        if p.exists():
            params.update(yaml.safe_load(p.read_text()) or {})
    return params
