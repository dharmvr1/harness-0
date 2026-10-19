"""HTTP header rules: value encoding and x-mcp-header. Fixtures marked SPEC are copied from the specification."""
from email.message import Message

import pytest
from ce_agent.mcp.headers import (check_param_headers, decode_header_value, encode_header_value, header_values_for_call,
                                  mirrored_parameters)

# SPEC: the "Encoding examples" table of the Streamable HTTP page
SPEC_TABLE = [
    ("us-west1", "us-west1"),                                                       # plain ASCII
    ("Hello, 世界", "=?base64?SGVsbG8sIOS4lueVjA==?="),                              # non-ASCII
    (" padded ", "=?base64?IHBhZGRlZCA=?="),                                         # leading/trailing spaces
    ("line1\nline2", "=?base64?bGluZTEKbGluZTI=?="),                                 # contains a newline
    ("=?base64?literal?=", "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="),                   # looks like the sentinel
]


def headers(**values):
    message = Message()
    for name, value in values.items():
        message[name.replace("_", "-")] = value
    return message


# ---------- value encoding ----------
@pytest.mark.parametrize("original, encoded", SPEC_TABLE)
def test_encoding_matches_the_spec_table_and_round_trips(original, encoded):
    assert encode_header_value(original) == encoded
    assert decode_header_value(encoded) == original

def test_plain_values_are_left_alone_including_inner_spaces_and_tabs():
    for value in ("a b", "x\ty", "~!@#$%^&*()", "", "5"):
        assert encode_header_value(value) == value

def test_leading_or_trailing_tab_forces_base64():
    assert encode_header_value("x\t").startswith("=?base64?") and encode_header_value("\tx").startswith("=?base64?")

def test_control_characters_force_base64():
    for value in ("a\x00b", "a\x7fb", "a\rb"):
        assert encode_header_value(value).startswith("=?base64?")
        assert decode_header_value(encode_header_value(value)) == value

def test_decoding_a_plain_value_returns_it_unchanged():
    assert decode_header_value("us-west1") == "us-west1"
    assert decode_header_value("=?base64?not closed") == "=?base64?not closed"      # not the sentinel form

@pytest.mark.parametrize("bad", ["=?base64?***?=", "=?base64?abc?=", "=?base64?/w==?="])    # not base64 / bad padding / invalid UTF-8
def test_bad_base64_is_rejected(bad):
    with pytest.raises(ValueError):
        decode_header_value(bad)

def test_the_sentinel_markers_are_case_sensitive():
    assert decode_header_value("=?BASE64?aGk=?=") == "=?BASE64?aGk=?="


# ---------- x-mcp-header: which parameters are mirrored ----------
SPEC_TOOL = {                                                                           # SPEC: execute_sql example
    "type": "object",
    "properties": {"region": {"type": "string", "description": "The region", "x-mcp-header": "Region"},
                   "query": {"type": "string", "description": "The SQL query"}},
    "required": ["region", "query"],
}

def test_the_spec_example_mirrors_the_region():
    assert mirrored_parameters(SPEC_TOOL) == [("Region", ("region",), "string")]

def test_a_schema_without_annotations_mirrors_nothing():
    assert mirrored_parameters({"type": "object", "properties": {"a": {"type": "string"}}}) == []
    assert mirrored_parameters({"type": "object", "additionalProperties": False}) == []

def test_nested_properties_are_reachable():
    schema = {"type": "object", "properties": {"opts": {"type": "object", "properties": {
        "zone": {"type": "string", "x-mcp-header": "Zone"}}}}}
    assert mirrored_parameters(schema) == [("Zone", ("opts", "zone"), "string")]

def prop(**kw):
    return {"type": "object", "properties": {"p": kw}}

@pytest.mark.parametrize("schema, why", [
    (prop(type="string", **{"x-mcp-header": ""}), "empty"),
    (prop(type="string", **{"x-mcp-header": "Bad Name"}), "space"),
    (prop(type="string", **{"x-mcp-header": "a:b"}), "colon"),
    (prop(type="string", **{"x-mcp-header": "A\nB"}), "newline"),
    (prop(type="string", **{"x-mcp-header": 5}), "not a string"),
    (prop(type="number", **{"x-mcp-header": "N"}), "number is not allowed"),
    (prop(type="array", **{"x-mcp-header": "A"}), "array"),
    (prop(type="object", **{"x-mcp-header": "O"}), "object"),
    (prop(**{"x-mcp-header": "NoType"}), "no type at all"),
    ({"type": "object", "properties": {
        "a": {"type": "string", "x-mcp-header": "Same"}, "b": {"type": "string", "x-mcp-header": "SAME"}}}, "duplicate ignoring case"),
    ({"type": "object", "properties": {"list": {"type": "array", "items": {"type": "string", "x-mcp-header": "X"}}}}, "through items"),
    ({"type": "object", "oneOf": [{"properties": {"a": {"type": "string", "x-mcp-header": "X"}}}]}, "through oneOf"),
    ({"type": "object", "anyOf": [{"properties": {"a": {"type": "string", "x-mcp-header": "X"}}}]}, "through anyOf"),
    ({"type": "object", "allOf": [{"properties": {"a": {"type": "string", "x-mcp-header": "X"}}}]}, "through allOf"),
    ({"type": "object", "$defs": {"d": {"type": "string", "x-mcp-header": "X"}}, "properties": {"a": {"$ref": "#/$defs/d"}}}, "through $ref"),
    ({"type": "object", "if": {"properties": {"a": {"type": "string", "x-mcp-header": "X"}}}}, "through if"),
])
def test_annotations_that_break_the_rules_reject_the_tool(schema, why):
    with pytest.raises(ValueError):
        mirrored_parameters(schema)


# ---------- the headers a client adds ----------
def test_values_become_headers_of_the_right_form():
    schema = {"type": "object", "properties": {
        "s": {"type": "string", "x-mcp-header": "S"}, "i": {"type": "integer", "x-mcp-header": "I"},
        "b": {"type": "boolean", "x-mcp-header": "B"}, "n": {"type": "integer", "x-mcp-header": "N"}}}
    assert header_values_for_call(schema, {"s": "us-west1", "i": -7, "b": True, "n": 0}) == {
        "Mcp-Param-S": "us-west1", "Mcp-Param-I": "-7", "Mcp-Param-B": "true", "Mcp-Param-N": "0"}
    assert header_values_for_call(schema, {"b": False})["Mcp-Param-B"] == "false"

def test_null_and_missing_parameters_are_omitted():
    assert header_values_for_call(SPEC_TOOL, {"region": None, "query": "x"}) == {}
    assert header_values_for_call(SPEC_TOOL, {"query": "x"}) == {}

def test_the_spec_example_call():                                                      # SPEC: Mcp-Param-Region: us-west1
    assert header_values_for_call(SPEC_TOOL, {"region": "us-west1", "query": "SELECT * FROM users"}) == {"Mcp-Param-Region": "us-west1"}

def test_unsafe_values_are_base64_encoded():
    assert header_values_for_call(SPEC_TOOL, {"region": "Hello, 世界"}) == {"Mcp-Param-Region": "=?base64?SGVsbG8sIOS4lueVjA==?="}

def test_nested_values_are_found_by_path():
    schema = {"type": "object", "properties": {"opts": {"type": "object", "properties": {"zone": {"type": "string", "x-mcp-header": "Zone"}}}}}
    assert header_values_for_call(schema, {"opts": {"zone": "a1"}}) == {"Mcp-Param-Zone": "a1"}
    assert header_values_for_call(schema, {"opts": {}}) == {} and header_values_for_call(schema, {}) == {}

def test_integers_outside_the_safe_range_cannot_be_sent():
    schema = {"type": "object", "properties": {"i": {"type": "integer", "x-mcp-header": "I"}}}
    assert header_values_for_call(schema, {"i": 2**53 - 1})
    with pytest.raises(ValueError, match="safe range"):
        header_values_for_call(schema, {"i": 2**53})

def test_a_value_that_cannot_be_mirrored_is_an_error():
    with pytest.raises(ValueError, match="cannot mirror"):
        header_values_for_call(SPEC_TOOL, {"region": ["a"]})


# ---------- what a server checks ----------
def test_matching_headers_pass():
    assert check_param_headers(SPEC_TOOL, {"region": "us-west1", "query": "q"}, headers(Mcp_Param_Region="us-west1")) is None

def test_a_missing_header_for_a_value_in_the_body_is_rejected():                       # SPEC: "Server MUST reject"
    assert "Missing required header Mcp-Param-Region" in check_param_headers(SPEC_TOOL, {"region": "x", "query": "q"}, headers())

def test_a_different_value_is_rejected():
    assert "does not match" in check_param_headers(SPEC_TOOL, {"region": "eu", "query": "q"}, headers(Mcp_Param_Region="us"))

def test_header_names_are_case_insensitive():
    assert check_param_headers(SPEC_TOOL, {"region": "x"}, headers(mcp_param_region="x")) is None

def test_encoded_values_are_decoded_before_comparing():
    body = {"region": "Hello, 世界"}
    assert check_param_headers(SPEC_TOOL, body, headers(Mcp_Param_Region="=?base64?SGVsbG8sIOS4lueVjA==?=")) is None
    assert "not valid" in check_param_headers(SPEC_TOOL, body, headers(Mcp_Param_Region="=?base64?***?="))

def test_null_or_absent_values_need_no_header_and_extra_headers_are_ignored():
    assert check_param_headers(SPEC_TOOL, {"region": None}, headers()) is None
    assert check_param_headers(SPEC_TOOL, {}, headers()) is None
    assert check_param_headers(SPEC_TOOL, {"query": "q"}, headers(Mcp_Param_Other="whatever")) is None

def test_integers_compare_as_numbers_not_text():                                       # SPEC: 42.0 and 42 are equal
    schema = {"type": "object", "properties": {"i": {"type": "integer", "x-mcp-header": "I"}}}
    assert check_param_headers(schema, {"i": 42}, headers(Mcp_Param_I="42.0")) is None
    assert check_param_headers(schema, {"i": 42.0}, headers(Mcp_Param_I="42")) is None
    assert check_param_headers(schema, {"i": 42}, headers(Mcp_Param_I="43"))
    assert check_param_headers(schema, {"i": 42}, headers(Mcp_Param_I="forty-two"))

def test_booleans_compare_as_lowercase_words():
    schema = {"type": "object", "properties": {"b": {"type": "boolean", "x-mcp-header": "B"}}}
    assert check_param_headers(schema, {"b": True}, headers(Mcp_Param_B="true")) is None
    assert check_param_headers(schema, {"b": True}, headers(Mcp_Param_B="True"))
    assert check_param_headers(schema, {"b": False}, headers(Mcp_Param_B="false")) is None

def test_a_tool_without_annotations_needs_no_headers():
    assert check_param_headers({"type": "object", "properties": {"a": {"type": "string"}}}, {"a": "x"}, headers()) is None
