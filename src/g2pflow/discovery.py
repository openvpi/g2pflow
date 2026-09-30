"""Discover decorator-registered components in built-in and external paths."""

import importlib
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType


_plugin_paths: list[Path] = []
_loaded_files: dict[Path, ModuleType] = {}
_EXTERNAL_NAMESPACE = "g2pflow._plugins"


def _link_module(name: str, module: ModuleType) -> None:
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    setattr(sys.modules[parent], child, module)


def _namespace(name: str, paths: list[str]) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = paths
    module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = paths
    _link_module(name, module)


def _remember_imports() -> None:
    # Relative imports may have already loaded another discovery candidate.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("g2pflow.") or module is None:
            continue
        filename = getattr(module, "__file__", None)
        if filename:
            _loaded_files.setdefault(Path(filename).resolve(), module)


def _discover_package(package: str, directory: Path, *, builtin: bool = False) -> None:
    for entry in sorted(directory.iterdir(), key=lambda path: path.name):
        if entry.is_file() and entry.suffix == ".py":
            if entry.stem == "__init__" or (builtin and entry.stem == "base"):
                continue
            source, name = entry, entry.stem
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            source, name = entry / "__init__.py", entry.name
        else:
            continue
        _remember_imports()
        fullname = f"{package}.{name}"
        existing = _loaded_files.get(source.resolve())
        if existing is not None:
            _link_module(fullname, existing)
        else:
            module = importlib.import_module(fullname)
            _loaded_files[source.resolve()] = module


def register_plugin_paths(*paths: str | Path) -> None:
    """Register directories in order and immediately discover their modules.

    Paths are resolved from the current working directory. A directory can
    contain both converters and preprocessors. Repeated calls do not reload
    previously imported modules. Each spawned process must register the same
    paths in the same order before unpickling external component instances.
    """
    resolved = [Path(path).expanduser().resolve(strict=True) for path in paths]
    for directory in resolved:
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))
    for directory in resolved:
        if directory not in _plugin_paths:
            _plugin_paths.append(directory)
    discover_plugins()


def discover_plugins() -> None:
    """Find new modules in built-in and registered external directories.

    Discovery is single-level: import each Python file and each package's
    initializer. Packages control their own submodule imports. Import errors
    propagate and already imported modules retain normal Python import state.
    """
    importlib.invalidate_caches()
    root = Path(__file__).parent
    for name in ("preprocessors", "converters"):
        _discover_package(f"g2pflow.{name}", root / name, builtin=True)
    if not _plugin_paths:
        return
    _namespace(_EXTERNAL_NAMESPACE, [])
    for index, directory in enumerate(_plugin_paths):
        package = f"{_EXTERNAL_NAMESPACE}.p{index}"
        _namespace(package, [str(directory)])
        _discover_package(package, directory)
