import pytest
from unittest.mock import MagicMock
from channel_loggers import ChannelLogParams, FixedField, JsonToServer
from frequency_manager import ChannelMessage

@pytest.mark.asyncio
async def test_fixed_field_logger(tmp_path):
    log_file = tmp_path / "channel.log"
    params = ChannelLogParams(type="fixed-field", target=str(log_file), timeout=0)
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
    params = ChannelLogParams(type="json-server", target="http://example.com/log", timeout=0)
    logger = JsonToServer(params)

    # Mock requests post directly on the instance's requests object
    mock_post = MagicMock()
    setattr(logger.requests, "post", mock_post)

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
