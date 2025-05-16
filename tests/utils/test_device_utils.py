"""
Unit tests for device utility functions in app.utils.device_utils.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock, call 

from app.utils.device_utils import get_selected_device, get_boxmot_device_string

# Test cases for get_selected_device
@pytest.mark.parametrize(
    "requested_device_str, cuda_available, mps_available, mps_built, expected_device_type, expected_log_parts",
    [
        ("auto", True, False, False, "cuda", ["Auto-selected CUDA device"]),
        ("auto", False, True, True, "mps", ["Auto-selected MPS device"]),
        ("auto", False, False, False, "cpu", ["Auto-selected CPU device."]),
        ("cuda", True, False, False, "cuda", ["Selected CUDA device"]),
        ("cuda:0", True, False, False, "cuda", ["Selected CUDA device"]),
        ("cuda", False, True, True, "cpu", ["CUDA requested but not available. Falling back.", "Device selection failed unexpectedly. Defaulting to CPU."]),
        ("cuda", False, False, False, "cpu", ["CUDA requested but not available. Falling back.", "Device selection failed unexpectedly. Defaulting to CPU."]),
        ("mps", False, True, True, "mps", ["Selected MPS device"]),
        ("mps", True, False, False, "cpu", ["MPS requested but not available/built. Falling back.", "Device selection failed unexpectedly. Defaulting to CPU."]),
        ("mps", False, False, False, "cpu", ["MPS requested but not available/built. Falling back.", "Device selection failed unexpectedly. Defaulting to CPU."]),
        ("cpu", True, True, True, "cpu", ["Selected CPU device."]),
        ("unknown_device", False, False, False, "cpu", ["Unknown device requested", "Falling back to CPU."]),
    ],
)
def test_get_selected_device(
    mocker,
    requested_device_str: str,
    cuda_available: bool,
    mps_available: bool,
    mps_built: bool,
    expected_device_type: str,
    expected_log_parts: list[str],
):
    """
    Tests get_selected_device with various scenarios and mocked hardware availability.
    """
    mocker.patch("torch.cuda.is_available", return_value=cuda_available)
    mocker.patch("torch.cuda.get_device_name", return_value="Mocked CUDA GPU")

    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available = MagicMock(return_value=mps_available)
    mock_mps_backend.is_built = MagicMock(return_value=mps_built)
    mocker.patch("torch.backends.mps", mock_mps_backend, create=True)

    mocker.patch("torch.tensor", return_value=MagicMock())

    mock_logger_info = mocker.patch("app.utils.device_utils.logger.info")
    mock_logger_warning = mocker.patch("app.utils.device_utils.logger.warning")
    mock_logger_error = mocker.patch("app.utils.device_utils.logger.error") 

    device = get_selected_device(requested_device_str)

    assert device.type == expected_device_type

    all_actual_log_calls_info = [call_args[0][0] for call_args in mock_logger_info.call_args_list]
    all_actual_log_calls_warning = [call_args[0][0] for call_args in mock_logger_warning.call_args_list]
    all_actual_log_calls_error = [call_args[0][0] for call_args in mock_logger_error.call_args_list]
    
    combined_logs = all_actual_log_calls_info + all_actual_log_calls_warning + all_actual_log_calls_error

    for msg_part in expected_log_parts:
        found_msg = any(msg_part in log_call for log_call in combined_logs)
        assert found_msg, f"Expected log message part '{msg_part}' not found in logs: {combined_logs}"


# Test cases for get_boxmot_device_string
@pytest.mark.parametrize(
    "input_torch_device, expected_string",
    [
        (torch.device("cuda:0"), "0"),
        (torch.device("cuda:1"), "1"),
        (torch.device("cuda"), "0"), 
        (torch.device("mps"), "mps"),
        (torch.device("cpu"), "cpu"),
    ],
)
def test_get_boxmot_device_string_valid_devices(input_torch_device, expected_string):
    """
    Tests get_boxmot_device_string with valid torch.device objects.
    """
    assert get_boxmot_device_string(input_torch_device) == expected_string

