"""
OpenTelemetry tracing configuration for Sequence FX Forecasting Toolkit.

Provides centralized tracing setup for all components including data loading,
feature engineering, model training, and evaluation.

Usage:
    from utils.tracing import setup_tracing, get_tracer

    # Setup tracing once at app startup
    setup_tracing(service_name="sequence-training")

    # Get tracer for your module
    tracer = get_tracer(__name__)

    # Use in your code
    with tracer.start_as_current_span("training_epoch") as span:
        span.set_attribute("epoch", epoch_num)
        # Your training code here
"""

from __future__ import annotations

from typing import Any

# Optional OpenTelemetry imports - gracefully degrade if not installed
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.torch import TorchInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    # OpenTelemetry not installed - use no-op implementations
    OTEL_AVAILABLE = False
    TracerProvider = type(None)  # Placeholder type for annotations
    trace = None


class NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str):
        """No-op context manager."""
        return NoOpSpan()


class NoOpSpan:
    """No-op span when OpenTelemetry is not available."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_attribute(self, key, value):
        pass

    def end(self):
        pass


def setup_tracing(
        service_name: str = "sequence",
        otlp_endpoint: str = "http://localhost:4318",
        environment: str = "development",
) -> TracerProvider | None:
    """
    Initialize OpenTelemetry tracing for the application.

    Args:
        service_name: Service name for tracing (default: "sequence")
        otlp_endpoint: OTLP collector HTTP endpoint (default: localhost:4318 for AI Toolkit)
        environment: Environment name (development, staging, production)

    Returns:
        Configured TracerProvider instance, or None if OpenTelemetry is not available

    Example:
        >>> setup_tracing(service_name="sequence-training")
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("training") as span:
        ...     span.set_attribute("epochs", 10)
    """
    if not OTEL_AVAILABLE:
        return None

    # Create OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        timeout=10,
    )

    # Create tracer provider with batch processor
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument PyTorch
    TorchInstrumentor().instrument()

    # Auto-instrument logging
    LoggingInstrumentor().instrument()

    # Set global service attributes
    resource_attributes = {
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": environment,
    }

    for key, value in resource_attributes.items():
        trace.get_current_span().set_attribute(key, value)

    return tracer_provider


def get_tracer(module_name: str):
    """
    Get a tracer instance for the given module.

    Args:
        module_name: Module name (typically __name__)

    Returns:
        Tracer instance for the module, or NoOpTracer if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        return NoOpTracer()
    return trace.get_tracer(module_name)


class TracingContext:
    """Context manager for tracing named spans with attributes."""

    def __init__(self, tracer: Any | None, span_name: str, attributes: dict | None = None):
        """
        Initialize tracing context. Works gracefully with None tracer.

        Args:
            tracer: Tracer instance (or None to skip tracing)
            span_name: Name of the span
            attributes: Optional dict of attributes to set on span
        """
        self.tracer = tracer
        self.span_name = span_name
        self.attributes = attributes or {}
        self.span = None

    def __enter__(self):
        if self.tracer is None:
            return None
        self.span = self.tracer.start_as_current_span(self.span_name)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span is None:
            return
        if exc_type is not None:
            self.span.set_attribute("error.type", exc_type.__name__)
            self.span.set_attribute("error.message", str(exc_val))
        self.span.end()


def trace_function(tracer: Any | None, func_name: str = None):
    """
    Decorator to automatically trace a function's execution.

    Args:
        tracer: Tracer instance
        func_name: Optional custom span name (defaults to function name)

    Example:
        >>> tracer = get_tracer(__name__)
        >>> @trace_function(tracer)
        ... def my_function(x, y):
        ...     return x + y
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = func_name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("function", func.__name__)
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("result.status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        return wrapper

    return decorator


# Convenience functions for common tracing scenarios

def trace_training_epoch(tracer: Any, epoch: int, num_epochs: int):
    """Context manager for tracing a training epoch."""
    return TracingContext(tracer, "training_epoch", {
        "epoch": epoch,
        "total_epochs": num_epochs,
    })


def trace_batch_processing(tracer: Any, batch_idx: int, batch_size: int):
    """Context manager for tracing batch processing."""
    return TracingContext(tracer, "batch_processing", {
        "batch_index": batch_idx,
        "batch_size": batch_size,
    })


def trace_validation(tracer: Any, val_loss: float, val_metric: float):
    """Context manager for tracing validation."""
    return TracingContext(tracer, "validation", {
        "val_loss": val_loss,
        "val_metric": val_metric,
    })


def trace_data_loading(tracer: Any, pair: str, num_samples: int):
    """Context manager for tracing data loading."""
    return TracingContext(tracer, "data_loading", {
        "pair": pair,
        "num_samples": num_samples,
    })


def trace_feature_engineering(tracer: Any, pair: str, feature_count: int):
    """Context manager for tracing feature engineering."""
    return TracingContext(tracer, "feature_engineering", {
        "pair": pair,
        "feature_count": feature_count,
    })
