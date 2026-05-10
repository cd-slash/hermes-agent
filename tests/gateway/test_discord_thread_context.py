"""Tests for Discord native thread-context backfill.

When Hermes is mentioned in an existing Discord thread for the first time,
it should prepend recent native thread messages once so the model sees the
conversation context before the persisted Hermes transcript takes over.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.Client = MagicMock
        discord_mod.File = MagicMock
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.ui = SimpleNamespace(
            View=object,
            button=lambda *a, **k: (lambda fn: fn),
            Button=object,
        )
        discord_mod.ButtonStyle = SimpleNamespace(
            success=1,
            primary=2,
            secondary=2,
            danger=3,
            green=1,
            grey=2,
            blurple=2,
            red=3,
        )
        discord_mod.Color = SimpleNamespace(
            orange=lambda: 1,
            green=lambda: 2,
            blue=lambda: 3,
            red=lambda: 4,
            purple=lambda: 5,
        )
        discord_mod.Interaction = object
        discord_mod.Embed = MagicMock
        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            autocomplete=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
        )

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)

    discord_mod = sys.modules["discord"]
    if not hasattr(discord_mod, "MessageType"):
        discord_mod.MessageType = SimpleNamespace(default=0, reply=1)


_ensure_discord_mock()

import gateway.platforms.discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class FakeThread:
    def __init__(self, channel_id: int = 456, name: str = "thread", parent=None, guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name=guild_name, id=777)
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999, name="HermesBot", display_name="HermesBot"))
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    return adapter


def _history_iter(messages):
    async def _history(*, limit=None, before=None, oldest_first=False):
        seq = list(messages)
        if not oldest_first:
            seq = list(reversed(seq))
        if limit is not None:
            seq = seq[:limit]
        for msg in seq:
            yield msg

    return _history


async def _unexpected_history(*, limit=None, before=None, oldest_first=False):
    raise AssertionError("thread history should not be fetched")
    yield  # pragma: no cover - keeps this an async generator


def _make_session_store(
    entries=None,
    *,
    group_sessions_per_user=True,
    thread_sessions_per_user=False,
    should_reset_return=None,
):
    store = MagicMock()
    store._entries = entries or {}
    store._ensure_loaded = MagicMock()
    store._should_reset = MagicMock(return_value=should_reset_return)
    store.config = SimpleNamespace(
        group_sessions_per_user=group_sessions_per_user,
        thread_sessions_per_user=thread_sessions_per_user,
    )
    return store


def _make_message(*, channel, author, content: str, msg_id: int, mentions=None):
    return SimpleNamespace(
        id=msg_id,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        type=discord_platform.discord.MessageType.default,
        guild=getattr(channel, "guild", None),
    )


@pytest.mark.asyncio
async def test_existing_discord_thread_prepends_native_thread_history(adapter, monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)

    thread = FakeThread(channel_id=456, name="planning")
    bot_user = adapter._client.user

    alice = SimpleNamespace(id=101, display_name="Alice", name="Alice", bot=False)
    bob = SimpleNamespace(id=202, display_name="Bob", name="Bob", bot=False)
    hermes = SimpleNamespace(id=999, display_name="HermesBot", name="HermesBot", bot=True)
    user = SimpleNamespace(id=303, display_name="Carol", name="Carol", bot=False)

    prior_1 = _make_message(channel=thread, author=alice, content="first note", msg_id=1)
    prior_2 = _make_message(channel=thread, author=bob, content="<@999> more details", msg_id=2)
    prior_3 = _make_message(channel=thread, author=hermes, content="old bot reply", msg_id=3)
    thread.history = _history_iter([prior_1, prior_2, prior_3])

    message = _make_message(
        channel=thread,
        author=user,
        content="<@999> what's next?",
        msg_id=4,
        mentions=[bot_user],
    )

    adapter._session_store = _make_session_store(entries={})

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "thread"
    assert event.source.chat_id == "456"
    assert event.source.thread_id == "456"
    assert event.text.startswith(
        "[Thread context — prior messages in this Discord thread (not yet in conversation history):]"
    )
    assert "Alice: first note" in event.text
    assert "Bob: more details" in event.text
    assert "HermesBot: old bot reply" not in event.text
    assert event.text.rstrip().endswith("what's next?")


@pytest.mark.asyncio
async def test_existing_session_skips_native_thread_backfill(adapter, monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)

    thread = FakeThread(channel_id=456, name="planning")
    thread.history = _unexpected_history
    bot_user = adapter._client.user
    user = SimpleNamespace(id=303, display_name="Carol", name="Carol", bot=False)
    message = _make_message(
        channel=thread,
        author=user,
        content="<@999> what's next?",
        msg_id=4,
        mentions=[bot_user],
    )

    adapter._session_store = _make_session_store(
        entries={"agent:main:discord:thread:456:456": MagicMock()}
    )

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "what's next?"


@pytest.mark.asyncio
async def test_expiring_discord_thread_session_still_backfills_native_history(adapter, monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)

    thread = FakeThread(channel_id=456, name="planning")
    bot_user = adapter._client.user
    alice = SimpleNamespace(id=101, display_name="Alice", name="Alice", bot=False)
    user = SimpleNamespace(id=303, display_name="Carol", name="Carol", bot=False)
    prior = _make_message(channel=thread, author=alice, content="first note", msg_id=1)
    thread.history = _history_iter([prior])

    message = _make_message(
        channel=thread,
        author=user,
        content="<@999> what's next?",
        msg_id=4,
        mentions=[bot_user],
    )

    adapter._session_store = _make_session_store(
        entries={
            "agent:main:discord:thread:456:456": SimpleNamespace(
                resume_pending=False,
                suspended=False,
            )
        },
        should_reset_return="idle",
    )

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "Alice: first note" in event.text
    assert event.text.rstrip().endswith("what's next?")


@pytest.mark.asyncio
async def test_discord_thread_backfill_does_not_break_commands(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    thread = FakeThread(channel_id=456, name="planning")
    thread.history = _unexpected_history
    user = SimpleNamespace(id=303, display_name="Carol", name="Carol", bot=False)
    message = _make_message(
        channel=thread,
        author=user,
        content="/status",
        msg_id=4,
        mentions=[],
    )

    adapter._session_store = _make_session_store(entries={})

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "/status"
    assert event.get_command() == "status"
