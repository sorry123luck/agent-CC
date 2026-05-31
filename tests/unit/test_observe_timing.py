from src.integration.observe_timing import attach_observe_timing


class Canvas:
    def __init__(self):
        self.artifacts = {}


def test_attach_observe_timing_records_stage_seconds_and_slowest_stage():
    canvas = Canvas()

    attach_observe_timing(
        canvas,
        {
            "1_perception_analyze": 1.23456,
            "4_resolve_page_model": 0.25,
            "11_cache_put": 2.0,
            "TOTAL": 3.5,
        },
    )

    timing = canvas.artifacts["observe_timing"]
    assert timing["stages"]["1_perception_analyze"] == 1.235
    assert timing["stages"]["11_cache_put"] == 2.0
    assert timing["total_seconds"] == 3.5
    assert timing["slowest_stage"] == {
        "name": "11_cache_put",
        "seconds": 2.0,
    }
