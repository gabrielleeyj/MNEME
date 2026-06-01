from __future__ import annotations

import pytest

from mneme.llm.json_io import JSONExtractionError, extract_json_object


def test_parses_plain_json_object():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_strips_json_code_fence():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_strips_bare_code_fence():
    assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_recovers_object_embedded_in_prose():
    raw = 'Here you go:\n{"a": 1}\nHope that helps!'
    assert extract_json_object(raw) == {"a": 1}


def test_no_object_raises():
    with pytest.raises(JSONExtractionError):
        extract_json_object("not json at all")


def test_unparseable_braces_raise():
    with pytest.raises(JSONExtractionError):
        extract_json_object("{ this is not : valid }")
