from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

try:  # pragma: no cover - optional dependency
    from nacl import public, encoding, exceptions
except ImportError:  # pragma: no cover
    public = None  # type: ignore
    encoding = None  # type: ignore
    exceptions = None  # type: ignore


class CryptoUnavailableError(RuntimeError):
    """Raised when PyNaCl is not available for secure aggregation."""


def _require_crypto() -> None:
    if public is None:
        raise CryptoUnavailableError(
            "PyNaCl is required for secure aggregation. Install it via `pip install pynacl`."
        )


@dataclass
class CryptoContext:
    """
    Holds key material for a client participating in secure aggregation.
    """

    client_id: str
    private_key: public.PrivateKey
    public_key: public.PublicKey
    peer_public_keys: Dict[str, public.PublicKey] = field(default_factory=dict)

    @classmethod
    def generate(cls, client_id: str) -> "CryptoContext":
        _require_crypto()
        private = public.PrivateKey.generate()
        return cls(client_id=client_id, private_key=private, public_key=private.public_key)

    def public_key_hex(self) -> str:
        _require_crypto()
        return self.public_key.encode(encoder=encoding.HexEncoder).decode("ascii")

    def register_peers(self, peer_map: Dict[str, str]) -> None:
        """
        Load peer public keys expressed in hex.
        """
        _require_crypto()
        self.peer_public_keys = {
            peer_id: public.PublicKey(peer_hex, encoder=encoding.HexEncoder)
            for peer_id, peer_hex in peer_map.items()
            if peer_id != self.client_id
        }

    def derive_shared_key(self, peer_id: str) -> bytes:
        _require_crypto()
        peer_key = self.peer_public_keys.get(peer_id)
        if peer_key is None:
            raise KeyError(f"Peer public key for {peer_id} not registered")
        box = public.Box(self.private_key, peer_key)
        return box.shared_key()
