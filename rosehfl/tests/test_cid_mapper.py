"""Tests for _CidMapper — maps Flower CIDs to partition indices."""
import pytest
from rosehfl._cid_mapper import CidMapper


class TestCidMapperRegistration:
    def test_register_from_metrics_stores_node_id(self):
        mapper = CidMapper(num_nodes=30)
        mapper.register_from_metrics("abc123", {"node_id": 5})
        assert mapper.resolve("abc123") == 5

    def test_register_ignores_invalid_node_id(self):
        mapper = CidMapper(num_nodes=30)
        mapper.register_from_metrics("abc", {"node_id": -1})
        mapper.register_from_metrics("def", {"node_id": 99})
        mapper.register_from_metrics("ghi", {})
        assert "abc" not in mapper.cid_to_node_id
        assert "def" not in mapper.cid_to_node_id
        assert "ghi" not in mapper.cid_to_node_id

    def test_register_overwrites_previous(self):
        mapper = CidMapper(num_nodes=30)
        mapper.register_from_metrics("abc", {"node_id": 3})
        mapper.register_from_metrics("abc", {"node_id": 7})
        assert mapper.resolve("abc") == 7


class TestCidMapperSortOrderFallback:
    def test_build_sort_order_enables_fallback(self):
        mapper = CidMapper(num_nodes=4)

        class FakeClient:
            def __init__(self, cid):
                self.cid = cid

        clients = [FakeClient("zzz"), FakeClient("aaa"), FakeClient("mmm"), FakeClient("bbb")]
        mapper.build_sort_order(clients)
        assert mapper.resolve("aaa") == 0
        assert mapper.resolve("bbb") == 1
        assert mapper.resolve("mmm") == 2
        assert mapper.resolve("zzz") == 3

    def test_resolve_raises_before_build_sort_order(self):
        mapper = CidMapper(num_nodes=4)
        with pytest.raises(ValueError, match="not in sort-order map"):
            mapper.resolve("unknown_cid")

    def test_resolve_raises_for_unregistered_cid_without_sort_order(self):
        mapper = CidMapper(num_nodes=4)

        class FakeClient:
            def __init__(self, cid):
                self.cid = cid

        mapper.build_sort_order([FakeClient("aaa"), FakeClient("bbb")])
        with pytest.raises(ValueError, match="not in sort-order map"):
            mapper.resolve("zzz")


class TestCidMapperMetricsOverride:
    def test_metrics_registration_overrides_sort_order(self):
        mapper = CidMapper(num_nodes=4)

        class FakeClient:
            def __init__(self, cid):
                self.cid = cid

        mapper.build_sort_order([FakeClient("aaa"), FakeClient("bbb")])
        assert mapper.resolve("aaa") == 0
        mapper.register_from_metrics("aaa", {"node_id": 3})
        assert mapper.resolve("aaa") == 3


class TestCidMapperIntegerCids:
    def test_integer_cid_works_via_metrics(self):
        mapper = CidMapper(num_nodes=30)
        mapper.register_from_metrics("5", {"node_id": 5})
        assert mapper.resolve("5") == 5

    def test_integer_cid_works_via_sort_order(self):
        mapper = CidMapper(num_nodes=4)

        class FakeClient:
            def __init__(self, cid):
                self.cid = cid

        mapper.build_sort_order([FakeClient("3"), FakeClient("1"), FakeClient("0"), FakeClient("2")])
        assert mapper.resolve("0") == 0
        assert mapper.resolve("3") == 3


class TestCidMapperCheckpoint:
    def test_to_from_checkpoint_roundtrip(self):
        mapper = CidMapper(num_nodes=30)
        mapper.register_from_metrics("abc", {"node_id": 5})
        mapper.register_from_metrics("def", {"node_id": 10})
        state = mapper.to_checkpoint()
        mapper2 = CidMapper(num_nodes=30)
        mapper2.from_checkpoint(state)
        assert mapper2.resolve("abc") == 5
        assert mapper2.resolve("def") == 10
