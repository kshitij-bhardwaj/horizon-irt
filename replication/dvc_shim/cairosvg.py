"""Stub: cairosvg is only used to rasterise a logo on plots; metrics stages never call it."""


def svg2png(*_a, **_k):
    raise RuntimeError("cairosvg stub: plot logo rendering is disabled in this replication env")
