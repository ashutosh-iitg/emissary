"""A deliberately flawed unit-conversion agent: the thing `improve` works on.

Two seeded faults, one in prose and one in code: the tool description does not
say which unit names it accepts, and the mile factor is wrong.
"""

import emissary

FACTORS_TO_METRES = {"m": 1.0, "km": 1000.0, "mi": 1600.0, "ft": 0.3048}


def convert(value: float, from_unit: str, to_unit: str) -> dict:
    metres = value * FACTORS_TO_METRES[from_unit]
    return {"value": metres / FACTORS_TO_METRES[to_unit], "unit": to_unit}


AGENT = emissary.Agent(
    name="converter",
    instructions="Answer unit-conversion questions with the convert tool. Reply with the number.",
    tools=(
        emissary.Tool(
            name="convert",
            description="Convert a length.",
            input_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
                "additionalProperties": False,
            },
            execute=convert,
            api_scope="none",
        ),
    ),
)
