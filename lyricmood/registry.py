"""The shared tool registry.

A tool is an async function `(ctx, args: InputModel) -> OutputModel`
decorated with `@tool`. Its docstring becomes the description the LLM sees,
and the pydantic models become the JSON schemas. The MCP server and the
LiteLLM chat both read this registry, so a tool is defined exactly once.

Adding a tool: create any module in `lyricmood/tools/` and decorate a function
with `@tool`. Modules there are discovered automatically.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel

if TYPE_CHECKING:
    from lyricmood.context import AppContext

ToolFn = Callable[["AppContext", Any], Awaitable[BaseModel]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: ToolFn

    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    def openai_schema(self) -> dict[str, Any]:
        """Function-calling schema in the OpenAI format LiteLLM accepts."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(self, fn: ToolFn | None = None, *, name: str | None = None):
        """Decorator registering `fn`; models are read from its type hints."""

        def register(fn: ToolFn) -> ToolFn:
            hints = get_type_hints(fn)
            params = list(inspect.signature(fn).parameters)
            if len(params) != 2:
                raise TypeError(f"{fn.__name__}: a tool takes exactly (ctx, args)")
            input_model, output_model = hints.get(params[1]), hints.get("return")
            for model in (input_model, output_model):
                if not (isinstance(model, type) and issubclass(model, BaseModel)):
                    raise TypeError(f"{fn.__name__}: args and return must be pydantic models")
            spec = ToolSpec(
                name=name or fn.__name__,
                description=inspect.cleandoc(fn.__doc__ or ""),
                input_model=input_model,
                output_model=output_model,
                fn=fn,
            )
            if spec.name in self._tools:
                raise ValueError(f"tool {spec.name!r} registered twice")
            self._tools[spec.name] = spec
            return fn

        return register(fn) if fn is not None else register

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}; known: {', '.join(self._tools)}") from None

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    async def call(self, ctx: "AppContext", name: str, args: dict[str, Any]) -> BaseModel:
        spec = self.get(name)
        return await spec.fn(ctx, spec.input_model.model_validate(args))

    async def call_json(self, ctx: "AppContext", name: str, args: dict[str, Any]) -> str:
        """Compact JSON for an LLM: no nulls, no whitespace."""
        result = await self.call(ctx, name, args)
        return result.model_dump_json(exclude_none=True)


registry = ToolRegistry()
tool = registry.tool

_loaded = False


def load_tools() -> ToolRegistry:
    """Import every module in `lyricmood.tools` so their `@tool`s register."""
    global _loaded
    if not _loaded:
        import lyricmood.tools as pkg

        for mod in pkgutil.iter_modules(pkg.__path__):
            if not mod.name.startswith("_"):
                importlib.import_module(f"{pkg.__name__}.{mod.name}")
        _loaded = True
    return registry
