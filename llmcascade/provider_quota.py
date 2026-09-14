"""Parse provider-reported remaining quota from HTTP headers / key APIs."""

from __future__ import annotations

from typing import Any


def _to_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return max(0, int(float(str(raw).split(";")[0].strip().rstrip("s"))))
    except (TypeError, ValueError):
        return None


def parse_quota_headers(
    headers: dict[str, str] | None,
    *,
    provider: str = "",
) -> dict[str, int]:
    """Return any of rpd/rpm/tpm the provider actually sent. Empty if unknown."""
    if not headers:
        return {}
    h = {str(k).lower(): str(v) for k, v in headers.items()}
    out: dict[str, int] = {}

    rpd = _to_int(h.get("x-ratelimit-remaining-requests-day"))
    rpm = _to_int(h.get("x-ratelimit-remaining-requests-minute"))
    tpm = _to_int(h.get("x-ratelimit-remaining-tokens"))
    rem_req = _to_int(h.get("x-ratelimit-remaining-requests"))
    or_rem = _to_int(h.get("x-ratelimit-remaining"))
    or_lim = _to_int(h.get("x-ratelimit-limit"))

    rpd_limit = _to_int(h.get("x-ratelimit-limit-requests-day"))
    rpm_limit = _to_int(h.get("x-ratelimit-limit-requests-minute"))
    tpm_limit = _to_int(h.get("x-ratelimit-limit-tokens"))
    lim_req = _to_int(h.get("x-ratelimit-limit-requests"))

    if provider == "groq":
        # Groq: remaining-requests / limit-requests are RPD; tokens are TPM.
        if rpd is None and rem_req is not None:
            rpd = rem_req
        if rpd_limit is None and lim_req is not None:
            rpd_limit = lim_req
    else:
        if rpm is None and rem_req is not None and rpd is None:
            rpm = rem_req
        if rpm_limit is None and lim_req is not None and rpd_limit is None:
            rpm_limit = lim_req

    if or_rem is not None:
        if or_lim is not None and or_lim >= 40:
            rpd = or_rem if rpd is None else rpd
        elif rpm is None:
            rpm = or_rem

    if rpd is not None:
        out["rpd"] = rpd
    if rpm is not None:
        out["rpm"] = rpm
    if tpm is not None:
        out["tpm"] = tpm
    if rpd_limit is not None:
        out["rpd_limit"] = rpd_limit
    if rpm_limit is not None:
        out["rpm_limit"] = rpm_limit
    if tpm_limit is not None:
        out["tpm_limit"] = tpm_limit
    return out


def parse_openrouter_key(payload: Any) -> dict[str, int]:
    """Best-effort remaining from GET /api/v1/key. Often credits only."""
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict):
        return {}
    out: dict[str, int] = {}
    rl = data.get("rate_limit")
    if isinstance(rl, dict):
        rem = _to_int(rl.get("remaining"))
        req = _to_int(rl.get("requests"))
        interval = str(rl.get("interval") or "").lower()
        if rem is not None:
            if "d" in interval or (req is not None and req >= 40):
                out["rpd"] = rem
            else:
                out["rpm"] = rem
    return out
