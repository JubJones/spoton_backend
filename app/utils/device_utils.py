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
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                _ = torch.tensor([1.0], device=device)
                selected_device = device
                logger.info(f"Auto-selected CUDA device: {device} ({torch.cuda.get_device_name(device)})")
            except Exception as e:
                logger.warning(f"CUDA available but failed test ({e}). Checking MPS.")
        
        if selected_device is None and hasattr(torch.backends, 'mps') and \
           torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                device = torch.device("mps")
                _ = torch.tensor([1.0], device=device)
                selected_device = device
                logger.info(f"Auto-selected MPS device: {device}")
            except Exception as e:
                logger.warning(f"MPS available but test failed ({e}). Falling back to CPU.")
        
        if selected_device is None:
            selected_device = torch.device("cpu")
            logger.info(f"Auto-selected CPU device.")
    else:
        logger.warning(f"Unknown device requested: '{requested_device}'. Falling back to CPU.")
        selected_device = torch.device("cpu")

    if selected_device is None: # Should not happen with fallbacks, but as a final safety.
        logger.error("Device selection failed unexpectedly. Defaulting to CPU.")
        selected_device = torch.device("cpu")
        
    return selected_device


def get_boxmot_device_string(device: torch.device) -> str:
    """
    Converts a torch.device object into a string specifier suitable for BoxMOT.
    Example: torch.device('cuda:0') -> '0', torch.device('cpu') -> 'cpu'
    """
    if not isinstance(device, torch.device):
        logger.warning(f"Input to get_boxmot_device_string is not torch.device: {type(device)}. Trying to convert.")
        try:
            device = torch.device(str(device)) # Attempt conversion
        except Exception:
            logger.error(f"Could not convert '{device}' to torch.device. Defaulting to 'cpu' string.")
            return 'cpu'

    device_type = device.type
    if device_type == 'cuda':
        # BoxMOT usually expects the device index string, e.g., '0', '1'
        return str(device.index if device.index is not None else 0)
    elif device_type == 'mps':
        return 'mps'
    elif device_type == 'cpu':
        return 'cpu'
    else:
        logger.warning(f"Unsupported torch.device type '{device_type}' for BoxMOT. Defaulting to 'cpu' string.")
        return 'cpu'