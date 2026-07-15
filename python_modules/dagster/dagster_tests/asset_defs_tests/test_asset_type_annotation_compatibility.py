import pytest
from dagster import DagsterInvalidDefinitionError, Definitions, asset
from dagster._core.definitions.assets.graph.asset_type_compatibility import is_annotation_compatible


def test_compatible_numeric_widening_does_not_raise():
    @asset
    def upstream() -> int:
        return 1

    @asset
    def downstream(upstream: float) -> int:
        return int(upstream) + 1

    Definitions(assets=[upstream, downstream]).resolve_asset_graph()


def test_incompatible_types_raise():
    @asset
    def upstream() -> int:
        return 1

    @asset
    def downstream(upstream: str) -> int:
        return len(upstream)

    with pytest.raises(DagsterInvalidDefinitionError, match="not compatible"):
        Definitions(assets=[upstream, downstream]).resolve_asset_graph()


def test_toggle_env_var_disables_check(monkeypatch):
    monkeypatch.setenv("DAGSTER_STATIC_ASSET_TYPE_CHECK", "0")

    @asset
    def upstream() -> int:
        return 1

    @asset
    def downstream(upstream: str) -> int:
        return len(upstream)

    Definitions(assets=[upstream, downstream]).resolve_asset_graph()


def test_any_is_always_compatible():
    from typing import Any

    assert is_annotation_compatible(int, Any)
    assert is_annotation_compatible(Any, str)


def test_optional_consumer_accepts_none_and_inner_type():
    assert is_annotation_compatible(int, int | None)
    assert is_annotation_compatible(type(None), int | None)
    assert not is_annotation_compatible(type(None), int)


def test_generic_container_args_checked_recursively():

    assert is_annotation_compatible(list[int], list[float])
    assert not is_annotation_compatible(list[int], list[str])


def test_subclass_relationship_is_compatible():
    class Base:
        pass

    class Child(Base):
        pass

    assert is_annotation_compatible(Child, Base)
    assert not is_annotation_compatible(Base, Child)


def test_no_type_annotations_is_compatible():
    @asset
    def upstream():
        return 1

    @asset
    def downstream(upstream) -> int:
        return upstream + 1

    Definitions(assets=[upstream, downstream]).resolve_asset_graph()
