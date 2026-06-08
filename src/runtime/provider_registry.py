"""Provider registry — process-wide singleton for Vision Runtime providers."""

from __future__ import annotations

import logging
from typing import Any

from src.runtime.provider_metadata import ProviderMetadata

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Process-wide singleton provider registry.

    Providers register themselves at startup (typically in ``api_server.lifespan``).
    ``HealthMonitor`` and future ``VisionRuntimeManager`` query the registry
    to discover available providers and their metadata.

    The registry does **not** start, stop, or invoke providers — it is a
    passive data structure.
    """

    _instance: ProviderRegistry | None = None

    def __new__(cls) -> ProviderRegistry:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._providers: dict[str, ProviderMetadata] = {}
            cls._instance = inst
        return cls._instance

    # -- write ----------------------------------------------------------------

    def register(self, metadata: ProviderMetadata) -> None:
        """Register (or overwrite) a provider."""
        old = self._providers.get(metadata.provider_id)
        self._providers[metadata.provider_id] = metadata
        if old is None:
            logger.info("Provider registered: %s (%s/%s)", metadata.provider_id, metadata.provider_type, metadata.mode)
        else:
            logger.info("Provider updated: %s", metadata.provider_id)

    def update_endpoint(self, provider_id: str, endpoint: str) -> None:
        """Update the endpoint of an already-registered provider.

        Returns without action if the provider is not registered.
        """
        old = self._providers.get(provider_id)
        if old is None:
            return
        self._providers[provider_id] = ProviderMetadata(
            provider_id=old.provider_id,
            provider_type=old.provider_type,
            mode=old.mode,
            capabilities=old.capabilities,
            default_timeout_seconds=old.default_timeout_seconds,
            endpoint=endpoint,
            process_info=old.process_info,
            model_dependency=old.model_dependency,
            health_check=old.health_check,
        )
        logger.info("Provider endpoint updated: %s → %s", provider_id, endpoint)

    # -- read -----------------------------------------------------------------

    def get(self, provider_id: str) -> ProviderMetadata | None:
        """Return metadata for *provider_id*, or ``None`` if not registered."""
        return self._providers.get(provider_id)

    def list_all(self) -> list[ProviderMetadata]:
        """Return all registered providers in registration order."""
        return list(self._providers.values())

    def list_by_type(self, provider_type: str) -> list[ProviderMetadata]:
        """Return providers matching *provider_type*."""
        return [p for p in self._providers.values() if p.provider_type == provider_type]

    def provider_ids(self) -> list[str]:
        """Return all registered provider IDs."""
        return list(self._providers.keys())

    # -- debug / test ---------------------------------------------------------

    def clear(self) -> None:
        """Remove all registered providers. **Test-only** — not for production."""
        self._providers.clear()

    def __len__(self) -> int:
        return len(self._providers)

    def __repr__(self) -> str:
        ids = ", ".join(self._providers.keys())
        return f"<ProviderRegistry [{ids}]>"


def get_registry() -> ProviderRegistry:
    """Return the global singleton registry."""
    return ProviderRegistry()
