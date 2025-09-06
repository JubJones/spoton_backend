# SpotOn Backend Code Style and Conventions

## Code Formatting and Linting

### Tool Configuration
- **ruff** - Modern Python linter and formatter (replaces black, isort, flake8)
- **Line length**: 88 characters
- **Target Python version**: 3.9+
- **Quote style**: Double quotes
- **Indentation**: 4 spaces (no tabs)

### Ruff Configuration
```toml
[tool.ruff]
line-length = 88
target-version = "py39"
select = ["E", "W", "F", "I", "UP", "C90", "N", "D", "ANN", "BLE", "B", "A", "RUF"]
ignore = ["D203", "D212", "ANN401"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

[tool.ruff.lint.pydocstyle]
convention = "google"
```

## Documentation Standards

### Docstring Convention
- **Google Style** docstrings for all functions, classes, and modules
- All public APIs must have comprehensive docstrings
- Include parameter types, return types, and examples where helpful

### Example Docstring
```python
def process_frame(frame: np.ndarray, camera_id: str) -> Dict[str, Any]:
    """Process video frame for person detection and tracking.
    
    Args:
        frame: Input video frame as numpy array
        camera_id: Unique identifier for the camera
        
    Returns:
        Dictionary containing detection results with bounding boxes,
        confidence scores, and tracking IDs.
        
    Raises:
        ValueError: If frame dimensions are invalid
        RuntimeError: If AI model is not loaded
    """
```

## Type Hints

### Requirements
- **Mandatory** type hints for all function parameters and return values
- Use `from typing import` for complex types
- Use `Optional[T]` for nullable values
- Use `Union[T, U]` for multiple types
- Use `Dict[str, Any]` for flexible dictionaries

### Examples
```python
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

async def get_tracking_data(
    task_id: str,
    camera_ids: List[str],
    start_time: Optional[datetime] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get tracking data for specified cameras."""
```

## Naming Conventions

### Variables and Functions
- **snake_case** for variables, functions, and module names
- Descriptive names that clearly indicate purpose
- Avoid single-letter variables except for loop counters

```python
# Good
person_tracker = BoxMOTTracker()
camera_frame_processor = MultiCameraFrameProcessor()
global_person_id = generate_unique_id()

# Bad  
pt = BoxMOTTracker()
cfp = MultiCameraFrameProcessor()
gid = generate_unique_id()
```

### Classes
- **PascalCase** for class names
- Use descriptive names that indicate the class purpose
- Inherit from appropriate base classes

```python
class PersonDetectionService:
    """Service for detecting persons in video frames."""

class CameraTrackerFactory:
    """Factory for creating camera-specific tracker instances."""
```

### Constants
- **UPPER_SNAKE_CASE** for constants
- Define module-level constants at the top of files

```python
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
MAX_TRACKING_HISTORY = 1000
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov"]
```

### File and Directory Names
- **snake_case** for Python files
- **lowercase** for directories
- Use descriptive names that indicate content

```
app/
├── domains/
│   ├── detection/
│   ├── reid/
│   └── mapping/
├── services/
│   ├── pipeline_orchestrator_service.py
│   ├── video_data_manager_service.py
│   └── notification_service.py
```

## Code Organization

### Import Organization
1. Standard library imports
2. Third-party library imports  
3. Local application imports
4. Blank lines between each group

```python
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.models.detectors import RTDETRDetector
from app.services.notification_service import NotificationService
```

### Class Organization
1. Class docstring
2. Class-level constants
3. `__init__` method
4. Public methods (alphabetically)
5. Private methods (prefixed with `_`)
6. Static methods and class methods last

### Function Organization
- Keep functions focused and single-purpose
- Maximum function length: ~50 lines (exceptions for complex algorithms)
- Use early returns to reduce nesting
- Extract complex logic into separate functions

## Error Handling

### Exception Handling
- Use specific exception types, not bare `except:`
- Log errors with appropriate levels (ERROR, WARNING, INFO)
- Provide meaningful error messages
- Use custom exceptions for domain-specific errors

```python
import logging

logger = logging.getLogger(__name__)

async def process_video_frame(frame_data: bytes) -> Dict[str, Any]:
    """Process video frame with comprehensive error handling."""
    try:
        # Decode frame
        frame = decode_frame(frame_data)
        
        # Run detection
        detections = await detector.detect(frame)
        
        return {"detections": detections, "status": "success"}
        
    except InvalidFrameError as e:
        logger.warning(f"Invalid frame data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except ModelNotLoadedError as e:
        logger.error(f"AI model not loaded: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    except Exception as e:
        logger.error(f"Unexpected error processing frame: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Logging Standards
- Use structured logging with clear levels
- Include context information in log messages
- Use f-strings for string formatting in logs
- Configure logger at module level

```python
import logging

logger = logging.getLogger(__name__)

# Good logging examples
logger.info(f"Starting processing task: {task_id}")
logger.warning(f"Low confidence detection: {confidence:.2f} < {threshold:.2f}")
logger.error(f"Failed to load model weights from {model_path}: {error}")
```

## Async/Await Patterns

### Async Function Guidelines
- Use `async def` for I/O-bound operations
- Use `await` for async function calls
- Use `asyncio.gather()` for concurrent operations
- Proper exception handling in async contexts

```python
async def process_multiple_cameras(camera_ids: List[str]) -> Dict[str, Any]:
    """Process frames from multiple cameras concurrently."""
    tasks = [
        process_camera_frame(camera_id) 
        for camera_id in camera_ids
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        camera_id: result 
        for camera_id, result in zip(camera_ids, results)
        if not isinstance(result, Exception)
    }
```

## Testing Conventions

### Test File Organization
- Test files in `tests/` directory
- Mirror source structure: `tests/services/test_detection_service.py`
- Use descriptive test function names: `test_detection_service_handles_invalid_frame`

### Test Markers
- Use pytest markers for test categorization
- Available markers: `unit`, `integration`, `security`, `performance`, `gpu`, `slow`

```python
import pytest

@pytest.mark.unit
async def test_person_detector_initialization():
    """Test that PersonDetector initializes correctly."""

@pytest.mark.integration
@pytest.mark.database
async def test_tracking_data_storage():
    """Test that tracking data is properly stored in database."""
```

## Configuration Management

### Environment Variables
- Use Pydantic Settings for configuration
- Document all environment variables
- Provide sensible defaults
- Validate configuration at startup

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    app_name: str = "SpotOn Backend"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    
    # Database settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "spoton"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

## Performance Guidelines

### Async Best Practices
- Use connection pooling for database operations
- Implement proper caching strategies
- Use background tasks for heavy operations
- Monitor memory usage and cleanup resources

### Resource Management
- Use context managers for file operations
- Close WebSocket connections properly
- Implement proper cleanup in exception handlers
- Monitor AI model memory usage