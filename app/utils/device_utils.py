import logging
import torch

logger = logging.getLogger(__name__)

def get_selected_device(requested_device: str = "auto") -> torch.device:
    """
    Gets the torch.device based on the requested setting and availability.
    Prioritizes CUDA > MPS > CPU for "auto".
    """
    req_device_lower = requested_device.lower()
    logger.info(f"--- Determining Device (Requested: '{requested_device}') ---")

    selected_device = None

    if req_device_lower.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                device = torch.device(req_device_lower)
                _ = torch.tensor([1.0], device=device) # Test allocation
                selected_device = device
                logger.info(f"Selected CUDA device: {device} ({torch.cuda.get_device_name(device)})")
            except Exception as e:
                logger.warning(f"Requested CUDA device '{requested_device}' not valid or test failed ({e}). Falling back.")
        else:
            logger.warning("CUDA requested but not available. Falling back.")

    elif req_device_lower == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                device = torch.device("mps")
                _ = torch.tensor([1.0], device=device) # Test allocation
                selected_device = device
                logger.info(f"Selected MPS device: {device}")
            except Exception as e:
                logger.warning(f"MPS available but test failed ({e}). Falling back.")
        else:
            logger.warning("MPS requested but not available/built. Falling back.")

    elif req_device_lower == "cpu":
        selected_device = torch.device("cpu")
        logger.info("Selected CPU device.")

    elif req_device_lower == "auto":
        logger.info("Attempting auto-detection: CUDA > MPS > CPU")
        # Try CUDA
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                _ = torch.tensor([1.0], device=device)
                selected_device = device
                logger.info(f"Auto-selected CUDA device: {device} ({torch.cuda.get_device_name(device)})")
            except Exception as e:
                logger.warning(f"CUDA available but failed test ({e}). Checking MPS.")
        # Try MPS if CUDA failed or unavailable
        if selected_device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                device = torch.device("mps")
                _ = torch.tensor([1.0], device=device)
                selected_device = device
                logger.info(f"Auto-selected MPS device: {device}")
            except Exception as e:
                logger.warning(f"MPS available but test failed ({e}). Falling back to CPU.")
        # Fallback to CPU
        if selected_device is None:
            selected_device = torch.device("cpu")
            logger.info(f"Auto-selected CPU device.")
    else:
        logger.warning(f"Unknown device requested: '{requested_device}'. Falling back to CPU.")
        selected_device = torch.device("cpu")

    # Final fallback if something went wrong
    if selected_device is None:
        logger.error("Device selection failed unexpectedly. Defaulting to CPU.")
        selected_device = torch.device("cpu")

    return selected_device