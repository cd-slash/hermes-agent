from types import SimpleNamespace

from plugins.memory.mem0 import Mem0MemoryProvider, _load_config


def test_load_config_reads_host_from_provider_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"mem0": {"host": "https://config.mem0.test"}}},
    )

    config = _load_config()

    assert config["api_key"] == "test-key"
    assert config["host"] == "https://config.mem0.test"


def test_load_config_prefers_provider_config_host_over_legacy_mem0_json(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"mem0": {"host": "https://config.mem0.test"}}},
    )
    (tmp_path / "mem0.json").write_text(
        '{"host": "https://file.mem0.test"}',
        encoding="utf-8",
    )

    config = _load_config()

    assert config["host"] == "https://config.mem0.test"


def test_load_config_falls_back_to_legacy_mem0_json_host_when_config_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    (tmp_path / "mem0.json").write_text(
        '{"host": "https://file.mem0.test"}',
        encoding="utf-8",
    )

    config = _load_config()

    assert config["host"] == "https://file.mem0.test"


def test_load_config_leaves_host_empty_without_provider_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})

    config = _load_config()

    assert config["api_key"] == "test-key"
    assert config["host"] == ""


def test_load_config_coerces_rerank_string_to_bool(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"mem0": {"rerank": "false"}}},
    )

    config = _load_config()

    assert config["rerank"] is False


def test_get_client_passes_host_to_memory_client(monkeypatch):
    captured = {}

    class FakeMemoryClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "mem0",
        SimpleNamespace(MemoryClient=FakeMemoryClient),
    )

    provider = Mem0MemoryProvider()
    provider._api_key = "test-key"
    provider._host = "https://host.mem0.test"

    client = provider._get_client()

    assert isinstance(client, FakeMemoryClient)
    assert captured == {
        "api_key": "test-key",
        "host": "https://host.mem0.test",
    }


def test_initialize_and_prefetch_use_configured_host(monkeypatch, tmp_path):
    captured = {}

    class FakeMemoryClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def search(self, **kwargs):
            return [{"memory": "prefetched fact"}]

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"mem0": {"host": "https://host.mem0.test"}}},
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "mem0",
        SimpleNamespace(MemoryClient=FakeMemoryClient),
    )

    provider = Mem0MemoryProvider()
    provider.initialize("session")
    provider.queue_prefetch("preferences")
    assert provider._prefetch_thread is not None
    provider._prefetch_thread.join(timeout=2)

    assert captured == {
        "api_key": "test-key",
        "host": "https://host.mem0.test",
    }
    assert "prefetched fact" in provider.prefetch("preferences")


def test_save_config_writes_provider_config_to_config_yaml(monkeypatch, tmp_path):
    provider = Mem0MemoryProvider()
    saved = {}

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "mem0"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.update(cfg))

    provider.save_config({"host": "https://config.mem0.test"}, tmp_path)

    assert saved["memory"]["provider"] == "mem0"
    assert saved["memory"]["mem0"]["host"] == "https://config.mem0.test"
    assert not (tmp_path / "mem0.json").exists()


def test_get_config_schema_includes_host():
    provider = Mem0MemoryProvider()
    schema = {item["key"]: item for item in provider.get_config_schema()}

    assert set(schema) >= {"api_key", "host"}
    assert "env_var" not in schema["host"]
