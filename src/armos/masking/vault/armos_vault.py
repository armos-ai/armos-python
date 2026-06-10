# SPDX-License-Identifier: Apache-2.0
"""
ArmosVault — client-side encrypted vault backed by the Armos proxy.

Values are encrypted locally before being sent to the proxy; the proxy
only ever stores opaque ciphertext.  Decryption always happens on the
client using the caller's api_key.
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from .base import BaseVault


# ---------------------------------------------------------------------------
# Crypto helpers
# ---------------------------------------------------------------------------

def _derive_key(api_key: str, token: str) -> bytes:
    """Derive a 32-byte AES key from api_key + token via HKDF-SHA256."""
    import hmac as hmac_mod
    import hashlib
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    salt = hmac_mod.new(b"armos-salt", token.encode(), hashlib.sha256).digest()
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"armos-vault-v1",
    )
    return hkdf.derive(api_key.encode())


def _encrypt_value(plaintext: str, key: bytes) -> str:
    """Encrypt *plaintext* with AES-GCM; return nonce+ciphertext as hex."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return (nonce + ciphertext).hex()


def _decrypt_value(encoded: str, key: bytes) -> str:
    """Decrypt a hex nonce+ciphertext produced by _encrypt_value."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    buf = bytes.fromhex(encoded)
    nonce, ciphertext = buf[:12], buf[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None).decode()


# ---------------------------------------------------------------------------
# ArmosVault
# ---------------------------------------------------------------------------

class ArmosVault(BaseVault):
    """
    Client-side encrypted vault that stores ciphertext on the Armos proxy.

    The proxy never sees plaintext — all encryption and decryption is
    performed locally using keys derived from the caller's api_key.

    Args:
        api_key:  Armos API key (used for auth and key derivation).
        base_url: Base URL of the Armos proxy (default: https://proxy.armos.dev).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://proxy.armos.dev",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # Local cache: token -> plaintext (avoids re-encrypting already-stored tokens)
        self._local_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Armos proxy unreachable at {url}. "
                f"Check your network or base_url. Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # BaseVault interface
    # ------------------------------------------------------------------

    def store(self, real_value: str, entity_type: str) -> str:
        """
        Encrypt *real_value* client-side, upload ciphertext to the proxy,
        and return the token.
        """
        # Deduplicate: same normalised value → same token (BaseVault logic)
        token = self.generate_token(real_value, entity_type)

        if token in self._local_cache:
            return token

        key = _derive_key(self._api_key, token)
        ciphertext = _encrypt_value(real_value, key)

        self._post("/v1/vault/store", {"token": token, "ciphertext": ciphertext})

        # Cache locally so subsequent calls skip the network round-trip
        self._local_cache[token] = real_value
        return token

    def retrieve(self, token: str) -> Optional[str]:
        """
        Return the real value for *token*, decrypting it client-side.
        Returns ``None`` if the token is unknown.
        """
        if token in self._local_cache:
            return self._local_cache[token]

        try:
            response = self._post("/v1/vault/retrieve", {"tokens": [token]})
        except ConnectionError:
            return None

        results: dict = response.get("results", {})
        ciphertext = results.get(token)
        if ciphertext is None:
            return None

        key = _derive_key(self._api_key, token)
        plaintext = _decrypt_value(ciphertext, key)
        self._local_cache[token] = plaintext
        return plaintext

    def clear(self) -> None:
        """Clear the local session cache. Remote entries expire via TTL."""
        self._local_cache.clear()
