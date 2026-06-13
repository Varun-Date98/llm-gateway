from __future__ import annotations

import os

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PORT = 8000


def vllm_command(model: str = DEFAULT_MODEL, port: int = DEFAULT_PORT) -> list[str]:
    return [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--model",
        model,
    ]


try:
    import modal
except ModuleNotFoundError:
    modal = None


if modal is not None:
    app = modal.App("llm-gateway-vllm")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("vllm")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    )

    @app.function(
        image=image,
        gpu=os.getenv("MODAL_GPU", "A10G"),
        timeout=60 * 60,
        scaledown_window=60,
    )
    @modal.web_server(DEFAULT_PORT)
    def serve() -> None:
        import subprocess

        subprocess.Popen(vllm_command(os.getenv("VLLM_MODEL", DEFAULT_MODEL))).wait()
else:
    app = None


if __name__ == "__main__":
    print(" ".join(vllm_command(os.getenv("VLLM_MODEL", DEFAULT_MODEL))))
