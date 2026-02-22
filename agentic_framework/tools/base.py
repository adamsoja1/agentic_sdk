from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints, overload


_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class BaseTool:
    name: str
    description: str
    func: Callable[..., Any]

    def execute(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def to_openai_schema(self) -> dict[str, Any]:
        sig = inspect.signature(self.func)
        try:
            hints = get_type_hints(self.func)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            py_type = hints.get(param_name, None)
            json_type = _PY_TO_JSON_TYPE.get(py_type, "string")
            properties[param_name] = {"type": json_type}

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }



def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str = "",
) -> BaseTool | Callable[[Callable], BaseTool]:

    def decorator(f: Callable) -> BaseTool:
        return BaseTool(
            name=name or f.__name__,
            description=description or (f.__doc__ or "").strip(),
            func=f,
        )

    if func is not None:
        return decorator(func)

    return decorator