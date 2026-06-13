from __future__ import annotations

import argparse
from dataclasses import dataclass

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class VastBootstrapConfig:
    model: str = DEFAULT_MODEL
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None


def build_vllm_start_command(config: VastBootstrapConfig) -> str:
    parts = [
        "python -m vllm.entrypoints.openai.api_server",
        "--host 0.0.0.0",
        f"--port {config.port}",
        f"--model {shell_quote(config.model)}",
        f"--tensor-parallel-size {config.tensor_parallel_size}",
        f"--gpu-memory-utilization {config.gpu_memory_utilization}",
    ]
    if config.max_model_len is not None:
        parts.append(f"--max-model-len {config.max_model_len}")
    return " ".join(parts)


def build_bootstrap_script(config: VastBootstrapConfig) -> str:
    command = build_vllm_start_command(config)
    return f"""#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install --upgrade vllm

cat > /tmp/start-vllm.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
{command}
EOF
chmod +x /tmp/start-vllm.sh

exec /tmp/start-vllm.sh
"""


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a Vast.ai vLLM bootstrap script.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        build_bootstrap_script(
            VastBootstrapConfig(
                model=args.model,
                port=args.port,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            )
        )
    )


if __name__ == "__main__":
    main()
