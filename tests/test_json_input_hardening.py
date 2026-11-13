"""Defense-depth checks for every JSON entry point that accepts untrusted input.

The token decode path normalizes interpreter-level JSON failures
(``ValueError`` for malformed or truncated documents, ``RecursionError`` for
pathologically nested ones) into :class:`jwt.exceptions.DecodeError`. The
key-side entry points -- ``PyJWK.from_json()``, ``PyJWKSet.from_json()``,
``PyJWKClient.fetch_data()``, and the per-algorithm ``from_jwk()``
JSON-string paths -- must behave the same way rather than leaking ``json``'s
exceptions to the caller.

The cases below are table-driven over those entry points, so a regression
that reopens the asymmetry shows up as one failing row rather than as a
silently untested path.
"""

import contextlib
import json
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any
from unittest import mock

import pytest

from jwt import PyJWK, PyJWKClient, PyJWKSet
from jwt.algorithms import HMACAlgorithm
from jwt.api_jws import PyJWS
from jwt.exceptions import (
    DecodeError,
    InvalidKeyError,
    PyJWKClientError,
    PyJWKError,
    PyJWKSetError,
    PyJWTError,
)
from jwt.utils import base64url_encode

# Upper bound for the calibration probe. If the parser still hasn't exhausted
# the recursion limit at this depth it is not doing bounded recursion at all,
# and calibration should fail loudly rather than pick a bogus threshold.
_PROBE_CEILING = 1 << 16


def _json_overflows_at(depth: int) -> bool:
    """Return whether ``json.loads`` raises ``RecursionError`` at *depth*."""
    try:
        json.loads("[" * depth + "]" * depth)
    except RecursionError:
        return True
    except ValueError:
        return False
    return False


@lru_cache(maxsize=1)
def measured_recursion_threshold() -> int:
    """Measure the smallest nesting depth where ``json.loads`` overflows.

    The threshold belongs to the interpreter and its current stack depth,
    not to this library, so it is calibrated at test time instead of hard
    coded. The returned value is the smallest depth observed to raise
    ``RecursionError``.
    """
    if not _json_overflows_at(_PROBE_CEILING):
        pytest.fail(
            "json.loads did not exhaust the recursion limit at depth "
            f"{_PROBE_CEILING}; the threshold cannot be calibrated"
        )

    # Exponential probe to bracket the threshold, then binary search.
    lower, upper = 1, 2
    while not _json_overflows_at(upper):
        lower, upper = upper, upper * 2
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if _json_overflows_at(middle):
            upper = middle
        else:
            lower = middle
    return upper


def deeply_nested_json() -> str:
    """A JSON document nested with a large margin past the measured threshold.

    The margin absorbs the small difference between the C stack available
    where the boundary was measured and the one available at each entry point.
    """
    depth = measured_recursion_threshold() * 4
    return "[" * depth + "]" * depth


@contextlib.contextmanager
def mocked_http_response(body: bytes) -> Iterator[None]:
    """Serve *body* from a mocked ``urllib.request.build_opener``."""
    with mock.patch("urllib.request.build_opener") as patched:
        opener = mock.Mock()
        patched.return_value = opener
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        # A truthy __exit__ would suppress the exception under test.
        response.__exit__ = mock.Mock(return_value=False)
        # Real urllib responses hand back bytes, not str.
        response.read.return_value = body
        opener.open.return_value = response
        yield


def _fetch_jwks_from_endpoint(body: str) -> Any:
    """Exercise the endpoint-pull entry point against a mocked HTTP response."""
    with mocked_http_response(body.encode("utf-8")):
        client = PyJWKClient("https://example.test/.well-known/jwks.json")
        return client.fetch_data()


def _load_jwk_set_from_endpoint(body: str) -> Any:
    """Exercise ``get_jwk_set()``, the documented route to the endpoint."""
    with mocked_http_response(body.encode("utf-8")):
        client = PyJWKClient("https://example.test/.well-known/jwks.json")
        return client.get_jwk_set()


def _decode_token_with_header(body: str) -> Any:
    """Exercise the token path with *body* as the protected header."""
    # ``_decode_base64url_segment`` rejects a segment whose length is 1 mod 4,
    # which would raise a padding DecodeError before the JSON is ever parsed.
    # Trailing whitespace is legal JSON, so pad with it to keep the payload
    # focused on JSON handling.
    while len(body) % 4 == 1:
        body += " "
    header_segment = base64url_encode(body.encode("utf-8")).decode("utf-8")
    # Payload "e30" is "{}"; the empty signature segment keeps the compact
    # form parseable, so only the header JSON is under test.
    return PyJWS().get_unverified_header(f"{header_segment}.e30.")


def _hmac_from_jwk(jwk: str) -> Any:
    """Exercise the algorithm-level ``from_jwk`` JSON-string path."""
    return HMACAlgorithm(HMACAlgorithm.SHA256).from_jwk(jwk)


# (entry point, error type raised for malformed JSON, error type raised for
# valid JSON that is not a JSON object). The token header is first so the
# table also locks parity between the already-hardened path and the rest.
ENTRY_POINTS = [
    pytest.param(
        _decode_token_with_header,
        DecodeError,
        DecodeError,
        id="token-header",
    ),
    pytest.param(
        PyJWK.from_json,
        PyJWKError,
        InvalidKeyError,
        id="PyJWK.from_json",
    ),
    pytest.param(
        PyJWKSet.from_json,
        PyJWKSetError,
        PyJWKSetError,
        id="PyJWKSet.from_json",
    ),
    pytest.param(
        _fetch_jwks_from_endpoint,
        PyJWKClientError,
        PyJWKClientError,
        id="PyJWKClient.fetch_data",
    ),
    pytest.param(
        _load_jwk_set_from_endpoint,
        PyJWKClientError,
        PyJWKClientError,
        id="PyJWKClient.get_jwk_set",
    ),
    pytest.param(
        _hmac_from_jwk,
        InvalidKeyError,
        InvalidKeyError,
        id="HMACAlgorithm.from_jwk",
    ),
]

MALFORMED_SHAPES = [
    pytest.param(deeply_nested_json, RecursionError, id="deeply-nested"),
    pytest.param(
        lambda: '{"kty": "RSA", "n": }',
        ValueError,
        id="invalid-syntax",
    ),
    pytest.param(
        lambda: '{"keys": [{"kty": "RSA", "n": "AQAB"',
        ValueError,
        id="truncated",
    ),
]


def test_recursion_threshold_is_measured_not_assumed() -> None:
    threshold = measured_recursion_threshold()

    assert threshold > 1
    # The calibrated value is real: a document with a large margin past it
    # still overflows in a fresh call. The exact boundary is not re-asserted
    # because the remaining C stack depends on the caller's frame depth.
    with pytest.raises(RecursionError):
        json.loads(deeply_nested_json())


@pytest.mark.parametrize("entry_point, error_cls, _not_an_object", ENTRY_POINTS)
@pytest.mark.parametrize("shape, cause_cls", MALFORMED_SHAPES)
def test_json_entry_points_normalize_malformed_input(
    entry_point: Callable[[str], Any],
    error_cls: type[PyJWTError],
    _not_an_object: type[PyJWTError],
    shape: Callable[[], str],
    cause_cls: type[BaseException],
) -> None:
    payload = shape()

    with pytest.raises(error_cls) as exc_info:
        entry_point(payload)

    # The caller sees the library's own error type, never json's.
    assert isinstance(exc_info.value, PyJWTError)
    assert not isinstance(exc_info.value, (ValueError, RecursionError))
    # The original failure is preserved for debugging, exactly as the token
    # path chains it into DecodeError.
    assert isinstance(exc_info.value.__cause__, cause_cls)


@pytest.mark.parametrize("entry_point, _malformed, not_an_object", ENTRY_POINTS)
def test_json_entry_points_normalize_valid_json_that_is_not_an_object(
    entry_point: Callable[[str], Any],
    _malformed: type[PyJWTError],
    not_an_object: type[PyJWTError],
) -> None:
    # Valid JSON that isn't a JWK/JWKS/header object used to escape as
    # AttributeError. Each entry point reports it in its own vocabulary, but
    # always as a PyJWTError.
    with pytest.raises(not_an_object):
        entry_point("[]")
