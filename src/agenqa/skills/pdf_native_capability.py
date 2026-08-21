from __future__ import annotations

import os
from typing import Any


_AIMUX_PDF_NATIVE_MODELS = {
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-6",
    "claude-sonnet-4-6-20260217",
}


def model_supports_pdf_native(*, api_channel: Any, model_name: Any) -> bool:
    model = str(model_name or "").strip().lower()
    if not model:
        return False
    if "gemini" in model:
        return True

    channel = str(api_channel or "").strip().lower()
    if channel == "aimux" and model in _AIMUX_PDF_NATIVE_MODELS:
        return True
    return False


def should_attach_pdf_natively(*, api_channel: Any, model_name: Any) -> bool:
    if os.getenv("SCICLONE_ALLOW_PDF_ATTACHMENT_ANY_MODEL", "").strip() == "1":
        return True
    return model_supports_pdf_native(api_channel=api_channel, model_name=model_name)


__all__ = ["model_supports_pdf_native", "should_attach_pdf_natively"]
