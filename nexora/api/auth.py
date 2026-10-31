"""Optional "Sign in with Google" for the local web app (OAuth 2.0 + PKCE, OpenID Connect).

Only login talks to Google; all AI inference stays on this computer.
Off by default. To enable you need your own OAuth client from Google Cloud
Console (see README) and at least one allowed e-mail address.

Emergency switch: set the environment variable NEXORA_DISABLE_AUTH=1 (or set
auth.enabled to false in runtime/settings.json) if you lock yourself out.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
import urllib.request

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
COOKIE = "nexora_session"
MASK = "********"
PUBLIC_PREFIXES = ("/auth/", "/static/", "/login")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class GoogleAuth:
    def __init__(self, app):
        self.app = app
        self._states: dict[str, tuple[str, float]] = {}

    # ----------------------------------------------------------- config
    @property
    def cfg(self) -> dict:
        return self.app.cfg().get("auth", {})

    def enabled(self) -> bool:
        if os.environ.get("NEXORA_DISABLE_AUTH") == "1":
            return False
        c = self.cfg
        return bool(c.get("enabled") and c.get("google_client_id") and c.get("allowed_emails"))

    def _key(self) -> bytes:
        p = self.app.paths.root / "secret.key"
        if not p.exists():
            p.write_text(secrets.token_hex(32))
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass
        return bytes.fromhex(p.read_text().strip())

    # ---------------------------------------------------------- sessions
    def make_session(self, email: str) -> str:
        payload = _b64(json.dumps({"email": email, "exp": time.time() +
                                   self.cfg.get("session_hours", 168) * 3600}).encode())
        sig = _b64(hmac.new(self._key(), payload.encode(), hashlib.sha256).digest())
        return f"{payload}.{sig}"

    def read_session(self, token: str | None) -> str | None:
        if not token or "." not in token:
            return None
        payload, sig = token.rsplit(".", 1)
        good = _b64(hmac.new(self._key(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        try:
            data = json.loads(_unb64(payload))
        except (ValueError, json.JSONDecodeError):
            return None
        if data.get("exp", 0) < time.time():
            return None
        if data.get("email", "").lower() not in self.allowed():
            return None  # removed from the allow-list -> session no longer valid
        return data["email"]

    def allowed(self) -> set[str]:
        return {e.strip().lower() for e in self.cfg.get("allowed_emails", []) if e.strip()}

    # ------------------------------------------------------------- OAuth
    def login_url(self) -> str:
        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        now = time.time()
        self._states = {k: v for k, v in self._states.items() if now - v[1] < 600}
        self._states[state] = (verifier, now)
        challenge = _b64(hashlib.sha256(verifier.encode()).digest())
        q = {"client_id": self.cfg["google_client_id"], "redirect_uri": self.cfg["redirect_uri"],
             "response_type": "code", "scope": "openid email profile", "state": state,
             "code_challenge": challenge, "code_challenge_method": "S256", "prompt": "select_account"}
        return f"{GOOGLE_AUTH}?{urllib.parse.urlencode(q)}"

    def exchange_code(self, code: str, verifier: str) -> dict:
        """POST to Google's token endpoint (the only network call NEXORA makes)."""
        body = urllib.parse.urlencode({
            "code": code, "client_id": self.cfg["google_client_id"],
            "client_secret": self.cfg.get("google_client_secret", ""),
            "redirect_uri": self.cfg["redirect_uri"], "grant_type": "authorization_code",
            "code_verifier": verifier}).encode()
        req = urllib.request.Request(GOOGLE_TOKEN, data=body, method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())

    def finish(self, code: str, state: str) -> str:
        """Validate the callback and return the signed-in e-mail (raises ValueError)."""
        entry = self._states.pop(state, None)
        if not entry or time.time() - entry[1] > 600:
            raise ValueError("login link expired or invalid - please try again")
        tokens = self.exchange_code(code, entry[0])
        idt = tokens.get("id_token")
        if not idt or idt.count(".") != 2:
            raise ValueError("Google did not return an identity token")
        # The ID token came straight from Google's token endpoint over TLS, so
        # (per OpenID Connect Core 3.1.3.7) its claims can be checked directly.
        claims = json.loads(_unb64(idt.split(".")[1]))
        if claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
            raise ValueError("token not issued by Google")
        if claims.get("aud") != self.cfg["google_client_id"]:
            raise ValueError("token issued for a different application")
        if claims.get("exp", 0) < time.time():
            raise ValueError("token expired")
        if not claims.get("email_verified"):
            raise ValueError("Google account e-mail is not verified")
        email = claims.get("email", "").lower()
        if email not in self.allowed():
            raise ValueError(f"{email} is not on the list of allowed accounts")
        self.app.db.audit(email, "login", method="google")
        return email
