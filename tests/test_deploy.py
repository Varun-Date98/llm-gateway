from deploy.modal_app import DEFAULT_MODEL, vllm_command
from deploy.vastai_bootstrap import (
    VastBootstrapConfig,
    build_bootstrap_script,
    build_vllm_start_command,
)


def test_modal_vllm_command_contains_openai_server_entrypoint() -> None:
    command = vllm_command("test/model", port=9000)

    assert command[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--model" in command
    assert "test/model" in command
    assert "9000" in command


def test_vastai_start_command_quotes_model_and_sets_vllm_flags() -> None:
    command = build_vllm_start_command(
        VastBootstrapConfig(model="org/model with space", port=9000, max_model_len=4096)
    )

    assert "vllm.entrypoints.openai.api_server" in command
    assert "--model 'org/model with space'" in command
    assert "--port 9000" in command
    assert "--max-model-len 4096" in command


def test_vastai_bootstrap_script_installs_vllm() -> None:
    script = build_bootstrap_script(VastBootstrapConfig(model=DEFAULT_MODEL))

    assert "pip install --upgrade vllm" in script
    assert "exec /tmp/start-vllm.sh" in script
