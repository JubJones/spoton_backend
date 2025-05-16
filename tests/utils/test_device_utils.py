"""
Unit tests for device utility functions in app.utils.device_utils.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock

from app.utils.device_utils import get_selected_device, get_boxmot_device_string

# Test cases for get_selected_device
@pytest.mark.parametrize(
    "requested_device_str, cuda_available, mps_available, mps_built, expected_device_type, expected_log_msgs",
    [
        ("auto", True, False, False, "cuda", ["Auto-selected CUDA device"]),
        ("auto", False, True, True, "mps", ["Auto-selected MPS device"]),
        ("auto", False, False, False, "cpu", ["Auto-selected CPU device"]),
        ("cuda", True, False, False, "cuda", ["Selected CUDA device"]),
        ("cuda:0", True, False, False, "cuda", ["Selected CUDA device"]),
        ("cuda", False, True, True, "cpu", ["CUDA requested but not available. Falling back.", "Auto-selected CPU device"]), # Falls back to auto logic
        ("mps", False, True, True, "mps", ["Selected MPS device"]),
        ("mps", True, False, False, "cpu", ["MPS requested but not available/built. Falling back.", "Auto-selected CPU device"]), # Falls back to auto
        ("cpu", True, True, True, "cpu", ["Selected CPU device."]),
        ("unknown_device", False, False, False, "cpu", ["Unknown device requested", "Auto-selected CPU device."]),
    ],
)
def test_get_selected_device(
    mocker,
    requested_device_str: str,
    cuda_available: bool,
    mps_available: bool,
    mps_built: bool,
    expected_device_type: str,
    expected_log_msgs: list[str],
):
    """
    Tests get_selected_device with various scenarios and mocked hardware availability.
    """
    mocker.patch("torch.cuda.is_available", return_value=cuda_available)
    mocker.patch("torch.cuda.get_device_name", return_value="Mocked CUDA GPU") # For log message

    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available = MagicMock(return_value=mps_available)
    mock_mps_backend.is_built = MagicMock(return_value=mps_built)
    mocker.patch("torch.backends.mps", mock_mps_backend, create=True) # create=True for when mps might not exist

    # Mock tensor creation to avoid actual hardware interaction for tests
    mocker.patch("torch.tensor", return_value=MagicMock())

    mock_logger_info = mocker.patch("app.utils.device_utils.logger.info")
    mock_logger_warning = mocker.patch("app.utils.device_utils.logger.warning")

    device = get_selected_device(requested_device_str)

    assert device.type == expected_device_type

    # Check if all expected log messages appeared
    for msg_part in expected_log_msgs:
        found_msg = False
        # Check info logs
        for call_args in mock_logger_info.call_args_list:
            if msg_part in call_args[0][0]:
                found_msg = True
                break
        # Check warning logs if not found in info
        if not found_msg:
            for call_args in mock_logger_warning.call_args_list:
                if msg_part in call_args[0][0]:
                    found_msg = True
                    break
        assert found_msg, f"Expected log message part '{msg_part}' not found."


# Test cases for get_boxmot_device_string
@pytest.mark.parametrize(
    "input_torch_device, expected_string",
    [
        (torch.device("cuda:0"), "0"),
        (torch.device("cuda:1"), "1"),
        (torch.device("cuda"), "0"), # Default index for cuda if not specified
        (torch.device("mps"), "mps"),
        (torch.device("cpu"), "cpu"),
    ],
)
def test_get_boxmot_device_string_valid_devices(input_torch_device, expected_string):
    """
    Tests get_boxmot_device_string with valid torch.device objects.
    """
    assert get_boxmot_device_string(input_torch_device) == expected_string

def test_get_boxmot_device_string_invalid_input_type(mocker):
    """
    Tests get_boxmot_device_string with an invalid input type that cannot be converted.
    """
    mock_logger_error = mocker.patch("app.utils.device_utils.logger.error")
    mock_logger_warning = mocker.patch("app.utils.device_utils.logger.warning")

    # Test with a type that torch.device() would reject if forced conversion
    assert get_boxmot_device_string(None) == "cpu" # type: ignore
    mock_logger_error.assert_called_once()
    assert "Could not convert 'None' to torch.device" in mock_logger_error.call_args[0][0]

    mock_logger_error.reset_mock()
    # Test with an unsupported torch device type after hypothetical conversion
    mock_unsupported_device = MagicMock(spec=torch.device)
    mock_unsupported_device.type = "xla"
    mocker.patch("torch.device", return_value=mock_unsupported_device) # If it was a string initially

    assert get_boxmot_device_string("xla_device_string") == "cpu"
    mock_logger_warning.assert_any_call(
        "Input to get_boxmot_device_string is not torch.device: <class 'str'>. Trying to convert."
    )
    mock_logger_warning.assert_any_call(
        "Unsupported torch.device type 'xla' for BoxMOT. Defaulting to 'cpu' string."
    )