import json
from pathlib import Path
from data.detector_geometry import make_rack_geometry

def test_shared_json_is_single_source():
    p=Path(__file__).parents[1]/"geometry"/"detector_geometry.json"
    d=json.loads(p.read_text())
    assert [x["count"] for x in d["layers"]]==[16,9]
    assert d["hodoscope"]["channels_per_hodoscope"]==25

def test_three_hodoscopes_75_sensitive_pieces():
    g=make_rack_geometry()
    assert len(g.bars)==75
    assert [b.channel_id for b in g.bars]==list(range(75))
    assert all(b.is_sensitive for b in g.bars)

def test_edge_halves_are_explicit_json_polygons():
    g=make_rack_geometry()
    h=g.hodoscopes[0]
    assert [b.bar_id for b in h.bottom_layer if b.is_half_end]==[0,15]
    assert [b.bar_id for b in h.top_layer if b.is_half_end]==[0,8]
    assert all(len(b.cross_section_vertices_m())==3 for b in h.bars)
