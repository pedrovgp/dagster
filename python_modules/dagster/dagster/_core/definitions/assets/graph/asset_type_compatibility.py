"""Static, definition-time validation that an asset's declared return type annotation is
compatible with the parameter type annotation of each downstream asset that depends on it.

This mirrors the assignability rules a type checker (mypy/pyright) would apply if the same
producer/consumer values were connected via a normal function call, since Dagster wires asset
dependencies together by name (parameter name -> upstream AssetKey) rather than by an actual
Python reference that mypy/pyright could see.

Toggle: set the DAGSTER_STATIC_ASSET_TYPE_CHECK environment variable to "0" to disable. Enabled
by default.
"""

import functools
import inspect
import os
import types
import typing as t
from pathlib import Path

from dagster._core.definitions.asset_key import AssetKey
from dagster._core.errors import DagsterInvalidDefinitionError

if t.TYPE_CHECKING:
    from dagster._core.definitions.assets.definition.assets_definition import AssetsDefinition

ENV_VAR_NAME = "DAGSTER_STATIC_ASSET_TYPE_CHECK"

_NUMERIC_TOWER: t.Sequence[type] = (bool, int, float, complex)


def is_static_asset_type_check_enabled() -> bool:
    return os.getenv(ENV_VAR_NAME, "1") not in ("0", "false", "False")


@functools.lru_cache(maxsize=1)
def _strict_optional_setting() -> bool:
    """Mirrors mypy/pyright's strict-optional behavior by reading the same config keys they read,
    so that this check is neither stricter nor looser about `None`/`Optional` handling than
    whatever type checker the user already has configured for their project.

    mypy: [tool.mypy] strict_optional (default True since mypy 0.600)
    pyright: [tool.pyright] strictParameterNoneValue (default True)
    """
    pyproject_path = _find_pyproject_toml()
    if pyproject_path is None:
        return True

    try:
        import tomli  # defer for perf

        with open(pyproject_path, "rb") as f:
            data = tomli.load(f)
    except Exception:
        return True

    mypy_setting = data.get("tool", {}).get("mypy", {}).get("strict_optional")
    if isinstance(mypy_setting, bool):
        return mypy_setting

    pyright_setting = data.get("tool", {}).get("pyright", {}).get("strictParameterNoneValue")
    if isinstance(pyright_setting, bool):
        return pyright_setting

    return True


def _find_pyproject_toml() -> Path | None:
    directory = Path.cwd()
    for candidate in (directory, *directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists():
            return pyproject
    return None


def _is_any(annotation: object) -> bool:
    return annotation is t.Any or annotation is inspect.Parameter.empty


def _is_none_type(annotation: object) -> bool:
    return annotation is type(None)


def _union_args(annotation: object) -> tuple[object, ...] | None:
    origin = t.get_origin(annotation)
    if origin is t.Union or origin is types.UnionType:
        return t.get_args(annotation)
    return None


def is_annotation_compatible(producer: object, consumer: object) -> bool:
    """Returns whether a value annotated with `producer` can flow into a parameter annotated
    with `consumer`, using the same assignability rules a static type checker applies:
    Any is compatible with everything, Optional/Union are widened/narrowed per PEP 484, the
    numeric tower (bool -> int -> float -> complex) is respected, and otherwise nominal subtyping
    (issubclass) is used. Unrecognized/complex typing constructs fail open (treated as
    compatible) rather than raising false positives on constructs we don't model.
    """
    if _is_any(producer) or _is_any(consumer):
        return True

    if consumer is object:
        return True

    strict_optional = _strict_optional_setting()

    if _is_none_type(producer):
        if not strict_optional or _is_none_type(consumer):
            return True
        consumer_union_members = _union_args(consumer)
        return consumer_union_members is not None and any(
            _is_none_type(member) for member in consumer_union_members
        )

    consumer_union = _union_args(consumer)
    if consumer_union is not None:
        return any(is_annotation_compatible(producer, member) for member in consumer_union)

    producer_union = _union_args(producer)
    if producer_union is not None:
        return all(is_annotation_compatible(member, consumer) for member in producer_union)

    if producer in _NUMERIC_TOWER and consumer in _NUMERIC_TOWER:
        return _NUMERIC_TOWER.index(producer) <= _NUMERIC_TOWER.index(consumer)

    producer_origin = t.get_origin(producer)
    consumer_origin = t.get_origin(consumer)
    if producer_origin is not None or consumer_origin is not None:
        if producer_origin != consumer_origin:
            return True  # fail open: don't model container-supertype relationships
        producer_args = t.get_args(producer)
        consumer_args = t.get_args(consumer)
        if len(producer_args) != len(consumer_args):
            return True  # fail open
        return all(
            is_annotation_compatible(p_arg, c_arg)
            for p_arg, c_arg in zip(producer_args, consumer_args)
        )

    if inspect.isclass(producer) and inspect.isclass(consumer):
        return issubclass(producer, consumer)

    # Unrecognized typing construct (TypeVar, Protocol, ForwardRef, etc.) -- fail open.
    return True


def validate_asset_graph_type_annotations(
    asset_nodes_by_key: t.Mapping[AssetKey, "t.Any"],
) -> None:
    """Walks every resolved asset dependency edge and validates that the producing asset's
    output type annotation is compatible with the consuming asset's input type annotation for
    that edge. Raises DagsterInvalidDefinitionError on the first incompatible edge found.

    This is a no-op if DAGSTER_STATIC_ASSET_TYPE_CHECK=0 is set in the environment.
    """
    if not is_static_asset_type_check_enabled():
        return

    for key, node in asset_nodes_by_key.items():
        assets_def: AssetsDefinition = node.assets_def
        for parent_key in node.parent_keys:
            input_name = assets_def.input_names_by_node_key.get(parent_key)
            if input_name is None or not assets_def.node_def.has_input(input_name):
                # Not every input name recorded on the AssetsDefinition maps onto an input on
                # its underlying NodeDefinition (e.g. graph-backed or subsetted assets, where the
                # node's input names can diverge). Fail open rather than block materialization.
                continue

            input_def = assets_def.node_def.input_def_named(input_name)
            if input_def.dagster_type.is_nothing:
                continue
            consumer_type = input_def.dagster_type.typing_type

            parent_node = asset_nodes_by_key.get(parent_key)
            if parent_node is None:
                continue
            parent_assets_def: AssetsDefinition = parent_node.assets_def
            output_name = next(
                (
                    name
                    for name, ak in parent_assets_def.node_keys_by_output_name.items()
                    if ak == parent_key
                ),
                None,
            )
            if output_name is None or not parent_assets_def.node_def.has_output(output_name):
                continue
            output_def = parent_assets_def.node_def.output_def_named(output_name)
            producer_type = output_def.dagster_type.typing_type

            if not is_annotation_compatible(producer_type, consumer_type):
                raise DagsterInvalidDefinitionError(
                    f"Asset '{key.to_user_string()}' has input '{input_name}' annotated as"
                    f" '{getattr(consumer_type, '__name__', consumer_type)}', which is not"
                    f" compatible with the return type annotation"
                    f" '{getattr(producer_type, '__name__', producer_type)}' of its upstream"
                    f" asset '{parent_key.to_user_string()}'. Set"
                    f" {ENV_VAR_NAME}=0 to disable this check."
                )
