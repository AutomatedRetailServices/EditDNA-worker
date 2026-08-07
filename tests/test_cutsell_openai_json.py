import pytest

from cutsell_worker.openai_json import parse_json_object


def test_parse_plain_json_object():
    assert parse_json_object('{"clips": []}') == {"clips": []}


def test_parse_markdown_fenced_json_object():
    assert parse_json_object('```json\n{"clips": []}\n```') == {"clips": []}


def test_parse_small_surrounding_prose_only_when_object_is_valid():
    assert parse_json_object('Result:\n{"ranked": []}\nDone.') == {"ranked": []}


def test_rejects_non_object_json():
    with pytest.raises(ValueError):
        parse_json_object('[1, 2, 3]')


def test_rejects_output_without_json_object():
    with pytest.raises(Exception):
        parse_json_object('not json')
