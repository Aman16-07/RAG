"""Central OpenTelemetry configuration for the multimodal RAG system.

Initialises:
  - TracerProvider with span processors
  - MeterProvider for application metrics
  - Console exporter (default) or OTLP exporter

Environment variables:
  ``OTEL_EXPORTER``             – ``console`` | ``otlp`` (default: ``console``)
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` – collector endpoint (default: ``http://localhost:4317``)
  ``OTEL_SERVICE_NAME``         – logical service name (default: ``rag-system``)
"""
from __future__ import annotations

import os

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_INITIALISED = False


def get_tracer(name: str = "rag-system") -> trace.Tracer:
    """Return a tracer scoped to *name*."""
    return trace.get_tracer(name)


def get_meter(name: str = "rag-system") -> metrics.Meter:
    """Return a meter scoped to *name*."""
    return metrics.get_meter(name)


def init_telemetry() -> None:
    """Initialise the OpenTelemetry SDK (idempotent).

    Call once at application startup (e.g. top of ``app.py``).
    """
    global _INITIALISED
    if _INITIALISED:
        return

    service_name = os.environ.get("OTEL_SERVICE_NAME", "rag-system")
    exporter_type = os.environ.get("OTEL_EXPORTER", "console").lower()

    resource = Resource.create({"service.name": service_name})

    # ── Traces ────────────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)

    if exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
        except ImportError:
            tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
    else:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )

    trace.set_tracer_provider(tracer_provider)

    # ── Metrics ───────────────────────────────────────────────────────
    if exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
            )
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=endpoint),
                export_interval_millis=10_000,
            )
        except ImportError:
            metric_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(), export_interval_millis=30_000
            )
    else:
        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=30_000
        )

    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    _INITIALISED = True
