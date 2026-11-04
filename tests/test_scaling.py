from src.phase2.scaling.executor import map_parallel


def test_map_parallel_preserves_order():
    items = list(range(8))
    out = map_parallel(items, lambda x: x * 2, max_workers=4)
    assert out == [0, 2, 4, 6, 8, 10, 12, 14]
