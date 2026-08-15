"""How each provider proves who it is (ADR-0021).

Authentication used to be two fields on `Provider` — an env var name and a
required flag — which assumed every backend holds one API key in the
environment. Vertex authenticates with Application Default Credentials and
addresses models by project and region instead, so the assumption had to
become a collaborator rather than grow a special case.

Every method here answers **without a network call**. `key_present` is
consulted before a request is spent, so a credential check that reaches out
would defeat the reason it is called at all.
"""

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Credential(Protocol):
    def available(self) -> bool:
        """Whether this credential can be obtained now, locally."""
        ...

    def token(self) -> str | None:
        """The bearer value, where the SDK needs it passed explicitly."""
        ...

    def describe(self) -> str:
        """What an operator must configure — for error messages only."""
        ...


@dataclass(frozen=True)
class ApiKey:
    env_var: str

    def available(self) -> bool:
        return bool(os.environ.get(self.env_var))

    def token(self) -> str | None:
        return os.environ.get(self.env_var) or None

    def describe(self) -> str:
        return self.env_var


@dataclass(frozen=True)
class Unauthenticated:
    """A backend that needs no credential — a local vLLM server, by default.

    `env_var` stays optional so a deployment that puts auth in front of it can
    still supply a key, which is then sent like any other provider's.
    """

    env_var: str | None = None

    def available(self) -> bool:
        return True

    def token(self) -> str | None:
        return os.environ.get(self.env_var) if self.env_var else None

    def describe(self) -> str:
        return f"{self.env_var} (optional)" if self.env_var else "no credential required"


@dataclass(frozen=True)
class GoogleADC:
    """Application Default Credentials plus a GCP project and region.

    Reports unavailable rather than raising on any failure. `google.auth` may
    be missing entirely, and a credential probe is not the place to discover
    that — the caller asked a yes/no question before spending a request.
    """

    project_env: str
    location_env: str
    default_location: str = "global"

    def available(self) -> bool:
        if not self.project():
            return False
        try:
            import google.auth

            credentials, _ = google.auth.default()
            return credentials is not None
        # Deliberately blanket: `google.auth` may be absent entirely, and ADC
        # discovery raises several unrelated types. A yes/no probe asked before
        # a request is spent must answer, never propagate.
        except Exception:  # noqa: BLE001
            return False

    def token(self) -> str | None:
        # The Google SDK resolves and refreshes ADC itself; handing it a
        # string here would bypass refresh and expire mid-run.
        return None

    def project(self) -> str | None:
        return os.environ.get(self.project_env) or None

    def location(self) -> str:
        return os.environ.get(self.location_env) or self.default_location

    def describe(self) -> str:
        return f"{self.project_env} and Google application default credentials"


__all__ = ["ApiKey", "Credential", "GoogleADC", "Unauthenticated"]
