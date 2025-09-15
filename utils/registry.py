from __future__ import annotations
from typing import Callable, List, Optional
import pkgutil, importlib

# In-memory registry for user tools (via decorator or add_tool)
_PHENO_TOOLS: List[Callable] = []

def pheno_tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to mark a function as a PhenoAssistant tool."""
    def _wrap(func: Callable):
        setattr(func, "_pheno_name", name or func.__name__)
        setattr(func, "_pheno_desc", description or (func.__doc__ or ""))
        _PHENO_TOOLS.append(func)
        return func
    return _wrap

def add_tool(func: Callable, name: Optional[str] = None, description: Optional[str] = None):
    """Programmatically add a Python function as a tool (no decorator needed)."""
    setattr(func, "_pheno_name", name or getattr(func, "_pheno_name", func.__name__))
    setattr(func, "_pheno_desc", description or getattr(func, "_pheno_desc", func.__doc__ or ""))
    _PHENO_TOOLS.append(func)

# we need to work on this so that it only registers the tools that have the decorator
def auto_discover_tools():
    """Import every module under ./functions so decorated tools self-register."""
    try:
        import functions  # existing tools folder
    except ImportError:
        return
    for mod in pkgutil.iter_modules(functions.__path__, functions.__name__ + "."):
        importlib.import_module(mod.name)

def register_all_tools(manager, executor):
    """Register all collected tools with the manager + executor."""
    from autogen import register_function
    for func in _PHENO_TOOLS:
        register_function(
            func,
            caller=manager,
            executor=executor,
            name=getattr(func, "_pheno_name", func.__name__),
            description=getattr(func, "_pheno_desc", func.__doc__ or ""),
        )