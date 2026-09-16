from data.detector_geometry import make_rack_geometry


def test_geometry():
    g = make_rack_geometry()
    assert len(g.bars) == 75

    for h in g.hodoscopes:
        assert len(h.bars) == 25
        assert len(h.bottom_layer) == 16
        assert len(h.top_layer) == 9
        assert h.bottom_layer[0].is_half_end
        assert h.bottom_layer[-1].is_half_end
        assert h.top_layer[0].is_half_end
        assert h.top_layer[-1].is_half_end
        assert all(b.is_sensitive for b in h.bars)
        assert [b.channel_id for b in h.bars] == list(
            range(h.hodoscope_id * 25, (h.hodoscope_id + 1) * 25)
        )
