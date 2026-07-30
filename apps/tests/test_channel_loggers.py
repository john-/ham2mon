import asyncio
from unittest.mock import MagicMock

import pytest
from channel_loggers import (
    ActivityLogger,
    ActivityParams,
    FixedField,
    JsonToServer,
    NoOp,
)
from frequency_manager import ChannelMessage, TransmissionRecord


@pytest.mark.asyncio
async def test_fixed_field_logger(tmp_path):
    log_file = tmp_path / "channel.log"
    params = ActivityParams(type="fixed-field", dest=str(log_file), interval=0)
    logger = FixedField(params)

    # 1. Message WITH matched CTCSS
    msg1 = ChannelMessage(
        state="on",
        rf=145.5,
        bb=0,
        channel=0,
        priority=1,
        classification="V",
        matched_ctcss=100.0,
        file="test1.wav"
    )
    await logger.log(msg1)

    # 2. Message WITHOUT matched CTCSS
    msg2 = ChannelMessage(
        state="off",
        rf=145.5,
        bb=0,
        channel=0,
        priority=None,
        classification=None,
        matched_ctcss=None,
        file="test2.wav"
    )
    await logger.log(msg2)

    assert log_file.exists()
    lines = log_file.read_text().splitlines()
    assert len(lines) == 2

    # Verify formatting of msg1: matched_ctcss '100.0  ' should be present right before the filename
    line1 = lines[0]
    # Check that we can find the tone and filename in the correct order/formatting
    assert "100.0  " in line1
    assert "test1.wav" in line1

    # Verify formatting of msg2: matched_ctcss is empty/omitted
    line2 = lines[1]
    # In line 2, the matched CTCSS column should be empty (7 spaces)
    # The file part is test2.wav
    assert "test2.wav" in line2
    # Ensure "100.0" is not in line2
    assert "100.0" not in line2


@pytest.mark.asyncio
async def test_json_to_server_logger():
    params = ActivityParams(type="json-server", dest="http://example.com/log", interval=0)
    logger = JsonToServer(params)

    # Mock requests post directly on the instance's requests object
    mock_post = MagicMock()
    logger.requests.post = mock_post

    msg = ChannelMessage(
        state="on",
        rf=145.5,
        bb=0,
        channel=0,
        priority=1,
        classification="V",
        matched_ctcss=141.3,
        file="test.wav"
    )
    await logger.log(msg)

    # Verify remote call was made with expected JSON body
    mock_post.assert_called_once()
    called_args, called_kwargs = mock_post.call_args
    assert called_args[0] == "http://example.com/log"

    posted_json = called_kwargs["json"]
    assert posted_json["matched_ctcss"] == 141.3
    assert posted_json["state"] == "on"
    assert posted_json["rf"] == 145.5
    assert posted_json["file"] == "test.wav"


class SpyLogger(ActivityLogger):
    def __init__(self, params: ActivityParams, get_ctcss=None) -> None:
        super().__init__(params, get_ctcss=get_ctcss)
        self.interval = params.interval
        self.logged_messages = []

    async def log(self, msg: ChannelMessage | None,
                  record: TransmissionRecord | None = None) -> None:
        if msg is None:
            return
        await super().log(msg, record)
        self.logged_messages.append(msg)
        self.handle_channel_state(msg)


def test_activity_logger_factory():
    # 1. fixed-field logger
    params_ff = ActivityParams(type="fixed-field", dest="file.log", interval=0)
    assert isinstance(ActivityLogger.get_logger(params_ff), FixedField)

    # 2. json-server logger
    params_js = ActivityParams(type="json-server", dest="http://example.com", interval=0)
    assert isinstance(ActivityLogger.get_logger(params_js), JsonToServer)

    # 3. debug (which is now removed/unsupported) -> falls back to NoOp
    params_debug = ActivityParams(type="debug", dest="", interval=0)
    assert isinstance(ActivityLogger.get_logger(params_debug), NoOp)

    # 4. none -> NoOp
    params_none = ActivityParams(type="none", dest="", interval=0)
    assert isinstance(ActivityLogger.get_logger(params_none), NoOp)


@pytest.mark.asyncio
async def test_channel_state_handling_keyerror_and_leak_prevention():
    params = ActivityParams(type="test", dest="", interval=5)
    logger = SpyLogger(params)

    # 1. Sending an 'off' message without an 'on' message first (no active task)
    # This should not raise a KeyError.
    msg_off = ChannelMessage(state="off", rf=145.5, bb=0, channel=1)
    await logger.log(msg_off)
    assert 1 not in logger.log_task

    # 2. Sending an 'on' message should start a task
    msg_on = ChannelMessage(state="on", rf=145.5, bb=0, channel=2)
    await logger.log(msg_on)
    assert 2 in logger.log_task
    task = logger.log_task[2]
    assert not task.done()

    # 3. Sending an 'off' message should cancel the task and delete it from dict
    msg_off_2 = ChannelMessage(state="off", rf=145.5, bb=0, channel=2)
    await logger.log(msg_off_2)
    assert 2 not in logger.log_task
    # Yield control to let cancellation take effect in the event loop
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_channel_state_active_logging():
    # Set a very low timeout (10ms) for testing
    params = ActivityParams(type="test", dest="", interval=0.01)
    mock_get_ctcss = MagicMock(return_value=100.0)
    logger = SpyLogger(params, get_ctcss=mock_get_ctcss)

    # Send 'on' message to start active logging
    msg_on = ChannelMessage(state="on", rf=145.5, bb=0, channel=3)
    await logger.log(msg_on)

    # Wait long enough for at least 2 active logging ticks to occur
    await asyncio.sleep(0.035)

    # Send 'off' message to terminate active logging
    msg_off = ChannelMessage(state="off", rf=145.5, bb=0, channel=3)
    await logger.log(msg_off)

    # Verify that 'act' (active) messages were logged periodically
    act_messages = [m for m in logger.logged_messages if m.state == "act"]
    assert len(act_messages) >= 2
    for m in act_messages:
        assert m.rf == 145.5
        assert m.channel == 3
        assert m.matched_ctcss == 100.0

    mock_get_ctcss.assert_called_with(0)
