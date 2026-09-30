__all__ = ["run_inference"]

__version__ = "1.0.2"


def __getattr__(name):
    """Load video and YOLO dependencies only when inference is requested."""
    if name == "run_inference":
        from .inference import run_inference

        return run_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
