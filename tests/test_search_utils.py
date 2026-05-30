"""Unit tests for search utility functions."""

import pytest

from retain.search import (
    _format_sparse,
    _format_vector,
    _query_is_identifier_heavy,
)


class TestFormatVector:
    def test_normal(self):
        assert _format_vector([0.5, -0.3, 0.1]) == "[0.5,-0.3,0.1]"

    def test_empty_list(self):
        assert _format_vector([]) == "[]"

    def test_single_element(self):
        assert _format_vector([0.75]) == "[0.75]"


class TestFormatSparse:
    def test_normal(self):
        result = _format_sparse({"indices": [1, 3, 5], "values": [0.7, 0.3, 0.9]})
        assert result == "{1:0.7,3:0.3,5:0.9}/30522"

    def test_empty(self):
        result = _format_sparse({"indices": [], "values": []})
        assert result == "{}/30522"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            _format_sparse({"indices": [1, 2], "values": [0.5]})

    def test_missing_indices_key(self):
        with pytest.raises(ValueError, match="length mismatch"):
            _format_sparse({"values": [0.5]})

    def test_missing_values_key(self):
        with pytest.raises(ValueError, match="length mismatch"):
            _format_sparse({"indices": [1]})


class TestQueryIdentifierHeavy:
    def test_digits_true(self):
        assert _query_is_identifier_heavy("order 12345") is True

    def test_semantic_false(self):
        assert _query_is_identifier_heavy("billing problem refund") is False

    def test_short_digits_false(self):
        assert _query_is_identifier_heavy("ref 12") is False

    def test_phone_number_true(self):
        assert _query_is_identifier_heavy("phone 555-123-4567") is True

    def test_empty_string_false(self):
        assert _query_is_identifier_heavy("") is False

    def test_account_number_true(self):
        assert _query_is_identifier_heavy("account 987654321") is True
