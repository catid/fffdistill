#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextPatch:
    name: str
    path: Path
    expected_version_package: str
    expected_versions: set[str]
    original: str
    replacement: str
    marker: str


def _package_root(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"Could not locate installed module {module_name!r}")
    return Path(next(iter(spec.submodule_search_locations)))


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required package {package!r} is not installed") from exc


def _apply_text_patch(patch: TextPatch) -> bool:
    installed_version = _version(patch.expected_version_package)
    if installed_version not in patch.expected_versions:
        expected = ", ".join(sorted(patch.expected_versions))
        raise RuntimeError(
            f"{patch.name}: refusing to patch {patch.expected_version_package} "
            f"version {installed_version}; expected one of {expected}"
        )
    text = patch.path.read_text(encoding="utf-8")
    if patch.marker in text:
        print(f"{patch.name}: already patched ({patch.path})")
        return False
    if patch.original not in text:
        raise RuntimeError(f"{patch.name}: target hunk not found in {patch.path}")
    patch.path.write_text(text.replace(patch.original, patch.replacement), encoding="utf-8")
    print(f"{patch.name}: patched {patch.path}")
    return True


def _patches() -> list[TextPatch]:
    tvm_ffi_root = _package_root("tvm_ffi")
    mamba_root = _package_root("mamba_ssm")
    tilelang_root = _package_root("tilelang")
    support_path = (
        tilelang_root
        / "3rdparty"
        / "tvm"
        / "python"
        / "tvm"
        / "runtime"
        / "support.py"
    )
    return [
        TextPatch(
            name="tvm_ffi_skip_unsettable_type_dunders",
            path=tvm_ffi_root / "registry.py",
            expected_version_package="apache-tvm-ffi",
            expected_versions={"0.1.12"},
            marker="_UNSETTABLE_TYPE_DUNDERS",
            original='''def _add_class_attrs(type_cls: type, type_info: TypeInfo) -> type:
    for field in type_info.fields:
        name = field.name
        if name not in type_cls.__dict__:  # skip attributes defined directly on this class
            setattr(type_cls, name, field.as_property(type_cls))
''',
            replacement='''_UNSETTABLE_TYPE_DUNDERS = frozenset({"__dict__", "__weakref__"})


def _add_class_attrs(type_cls: type, type_info: TypeInfo) -> type:
    for field in type_info.fields:
        name = field.name
        if name in _UNSETTABLE_TYPE_DUNDERS:
            continue
        if name not in type_cls.__dict__:  # skip attributes defined directly on this class
            setattr(type_cls, name, field.as_property(type_cls))
''',
        ),
        TextPatch(
            name="mamba3_optional_cute_step_import",
            path=mamba_root / "modules" / "mamba3.py",
            expected_version_package="mamba-ssm",
            expected_versions={"2.3.2.post1"},
            marker="except Exception as exc:  # optional Cute inference dependency",
            original=(
                '''try:
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
except ImportError:'''
                + "    \n"
                + '''    mamba3_step_fn = None
'''
            ),
            replacement='''try:
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
except Exception as exc:  # optional Cute inference dependency
    mamba3_step_fn = None
    mamba3_step_import_error = exc
else:
    mamba3_step_import_error = None
''',
        ),
        TextPatch(
            name="tilelang_tvm_derived_object_python_state",
            path=support_path,
            expected_version_package="tilelang",
            expected_versions={"0.1.8"},
            marker='__slots__ = ("_inst", "__weakref__")',
            original='''    class TVMDerivedObject(metadata["cls"]):  # type: ignore
        """The derived object to avoid cyclic dependency."""

        _cls = cls
        _type = "TVMDerivedObject"
''',
            replacement='''    class TVMDerivedObject(metadata["cls"]):  # type: ignore
        """The derived object to avoid cyclic dependency."""

        __slots__ = ("_inst", "__weakref__")
        _cls = cls
        _type = "TVMDerivedObject"
''',
        ),
        TextPatch(
            name="tilelang_tvm_derived_object_setattr",
            path=support_path,
            expected_version_package="tilelang",
            expected_versions={"0.1.8"},
            marker='inst = object.__getattribute__(self, "_inst")',
            original='''        def __setattr__(self, name, value):
            if name not in ["_inst", "key", "handle"]:
                self._inst.__setattr__(name, value)
            else:
                super(TVMDerivedObject, self).__setattr__(name, value)
''',
            replacement='''        def __setattr__(self, name, value):
            if name == "_inst":
                object.__setattr__(self, name, value)
                return
            if name in ["key", "handle"]:
                super(TVMDerivedObject, self).__setattr__(name, value)
                return
            try:
                inst = object.__getattribute__(self, "_inst")
            except AttributeError:
                object.__setattr__(self, name, value)
            else:
                inst.__setattr__(name, value)
''',
        ),
    ]


def main() -> int:
    changed = False
    for patch in _patches():
        changed = _apply_text_patch(patch) or changed
    if changed:
        print("vendor patches applied")
    else:
        print("vendor patches already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
