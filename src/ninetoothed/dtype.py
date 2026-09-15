int8 = "i8"
int16 = "i16"
int32 = "i32"
int64 = "i64"

uint8 = "u8"
uint16 = "u16"
uint32 = "u32"
uint64 = "u64"

float16 = "fp16"
bfloat16 = "bf16"
float32 = "fp32"
float64 = "fp64"


def normalize_dtype(dtype: str | None) -> str | None:
    """Resolve public dtype aliases to backend-neutral type names."""
    if dtype is None:
        return None

    name = dtype.strip().strip("'\"").rsplit(".", 1)[-1]
    aliases = {
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
        "u64": "uint64",
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
        "fp64": "float64",
        "float": "float32",
    }

    return aliases.get(name, name)
