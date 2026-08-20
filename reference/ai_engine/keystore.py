"""OS-keychain-backed storage for user-supplied API keys.

A shipped desktop app must not keep credentials in a dotenv file: plaintext on
disk gets swept into crash dumps, synced by backup tools, and shown in
screen-shares. `.env` is a development affordance and nothing more.

In the production Tauri build this logic lives on the Rust side (the `keyring`
crate) so that the webview can never read key material at all. This module is
the Python equivalent used by the CV sidecar's test harness and by CLI tooling.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    import keyring  # macOS Keychain / Windows Credential Manager / Secret Service
    from keyring.errors import KeyringError
except ImportError:  # pragma: no cover
    keyring = None
    KeyringError = Exception

SERVICE = "com.yourorg.gameplayanalyzer"

# Shape checks only. They catch paste errors early; they are not authentication.
KEY_PATTERNS = {
    "google": re.compile(r"^AIza[0-9A-Za-z_\-]{30,}$"),
    "openai": re.compile(r"^sk-[0-9A-Za-z_\-]{20,}$"),
    "anthropic": re.compile(r"^sk-ant-[0-9A-Za-z_\-]{20,}$"),
}

# Apply this at the logging sink, not at each call site: someone will eventually
# debug-print a whole request object, and you want that caught structurally.
SECRET_SCRUB = re.compile(r"\b(sk-ant-|sk-|AIza)[A-Za-z0-9_\-]{16,}")


def scrub(text: str) -> str:
    return SECRET_SCRUB.sub(r"\1[REDACTED]", text)


@dataclass(frozen=True)
class KeyFingerprint:
    """The only representation of a key that is ever allowed into the UI layer."""

    provider: str
    last4: str
    sha256_prefix: str
    stored_at: str

    def display(self) -> str:
        return f"{self.provider}: ••••{self.last4}"


class KeyStoreUnavailable(RuntimeError):
    """No Secret Service / keychain daemon is reachable.

    Common on headless Linux and minimal window managers. Surface this to the
    user and offer an explicitly-consented encrypted file instead. Never fall
    back to silent plaintext.
    """


def save_key(provider: str, api_key: str) -> KeyFingerprint:
    if provider not in KEY_PATTERNS:
        raise ValueError(f"unknown provider: {provider}")
    api_key = api_key.strip()
    if not KEY_PATTERNS[provider].match(api_key):
        raise ValueError(f"that does not look like a {provider} API key")
    if keyring is None:
        raise KeyStoreUnavailable("the `keyring` package is not installed")
    try:
        keyring.set_password(SERVICE, f"provider:{provider}", api_key)
    except KeyringError as exc:
        raise KeyStoreUnavailable(str(exc)) from exc
    return fingerprint(provider, api_key)


def load_key(provider: str) -> str | None:
    """Resolve a key: keychain first, then env var for development only."""
    if keyring is not None:
        try:
            if (stored := keyring.get_password(SERVICE, f"provider:{provider}")):
                return stored
        except KeyringError:
            pass  # fall through to env
    return os.environ.get(
        {"google": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[provider]
    )


def delete_key(provider: str) -> None:
    """Remove the local copy.

    The caller must also show the vendor's revocation URL: deleting locally
    does not revoke anything, and users reasonably assume it does.
    """
    if keyring is not None:
        try:
            keyring.delete_password(SERVICE, f"provider:{provider}")
        except KeyringError:
            pass


def fingerprint(provider: str, api_key: str) -> KeyFingerprint:
    return KeyFingerprint(
        provider=provider,
        last4=api_key[-4:],
        sha256_prefix=hashlib.sha256(api_key.encode()).hexdigest()[:12],
        stored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
