import re

from detect_secrets.plugins.base import RegexBasedDetector


class HuggingFaceTokenDetector(RegexBasedDetector):
    secret_type = "HuggingFace User Access Tokens"  # pragma: allowlist secret
    denylist = [
        re.compile(r"hf_[A-Za-z]{34}"),
    ]
