"""
HTTP header rules of the Streamable HTTP transport, as pure functions (no I/O).

The transport copies some fields of the JSON-RPC body into HTTP headers, so that load balancers and gateways can
route requests without parsing the body: Mcp-Method, Mcp-Name, and Mcp-Param-* for tool parameters that the tool
definition marks with "x-mcp-header". The body stays the source of truth, and servers MUST reject a request whose
headers and body disagree (otherwise a proxy and the server could act on different values: request smuggling).

Spec: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#request-metadata
"""

import base64
import binascii
import re

SENTINEL = re.compile(r"^=\?base64\?(.*)\?=$", re.S)
TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")        # RFC 9110 "token": what a header name may contain
SAFE_INTEGER = 2**53 - 1                                       # integers must survive a round trip through a double
PRIMITIVE_TYPES = ("string", "integer", "boolean")             # "number" is not allowed
JSON_SCHEMA_ANNOTATION = "x-mcp-header"
PARAM_PREFIX = "Mcp-Param-"


# ----------------------------------------------------------------------
# value encoding
# ----------------------------------------------------------------------
def _is_plain(value):
    """Can this go in a header as it is? Only visible ASCII, space and tab, and no space/tab at either end."""
    return value == value.strip(" \t") and all(c == "\t" or 0x20 <= ord(c) <= 0x7E for c in value)


def encode_header_value(value):
    """Plain text when that is safe, otherwise =?base64?<utf-8 as base64>?=.
    A plain value that happens to look like the base64 form is encoded too, so decoding is never ambiguous."""
    if _is_plain(value) and not SENTINEL.match(value):
        return value
    return "=?base64?" + base64.b64encode(value.encode("utf-8")).decode("ascii") + "?="


def decode_header_value(header):
    """The inverse of encode_header_value. Raises ValueError for a malformed base64 form."""
    match = SENTINEL.match(header)
    if not match:
        return header
    try:
        return base64.b64decode(match.group(1), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        raise ValueError("invalid base64 header value")


# ----------------------------------------------------------------------
# x-mcp-header: tool parameters that are mirrored into headers
# ----------------------------------------------------------------------
def _annotated_nodes(node, found):
    """Every dict anywhere in a JSON structure that carries an x-mcp-header key."""
    if isinstance(node, dict):
        if JSON_SCHEMA_ANNOTATION in node:
            found.add(id(node))
        for value in node.values():
            _annotated_nodes(value, found)
    elif isinstance(node, list):
        for value in node:
            _annotated_nodes(value, found)


def mirrored_parameters(input_schema):
    """The parameters of a tool that must be mirrored into headers: a list of (header name, path, JSON type).

    Raises ValueError when the annotations break the rules; a client must then reject the whole tool.
    The rules: the value is a non-empty header token, unique ignoring case; only on string / integer / boolean
    properties; and only on properties reachable from the schema root through "properties" alone (not through
    items, oneOf/anyOf/allOf/not, if/then/else or $ref)."""
    reached, result, seen_names = set(), [], set()

    def walk(schema, path):
        for name, sub in (schema.get("properties") or {}).items():
            if not isinstance(sub, dict):
                continue
            if JSON_SCHEMA_ANNOTATION in sub:
                reached.add(id(sub))
                header = sub[JSON_SCHEMA_ANNOTATION]
                if not isinstance(header, str) or not header or not TOKEN.match(header):
                    raise ValueError(f"invalid x-mcp-header value {header!r} on '{name}': must be a non-empty header token")
                if header.lower() in seen_names:
                    raise ValueError(f"x-mcp-header '{header}' is used twice (names are compared ignoring case)")
                seen_names.add(header.lower())
                if sub.get("type") not in PRIMITIVE_TYPES:
                    raise ValueError(f"x-mcp-header on '{name}': only string, integer and boolean parameters can be mirrored")
                result.append((header, path + (name,), sub["type"]))
            walk(sub, path + (name,))

    walk(input_schema, ())
    everywhere = set()
    _annotated_nodes(input_schema, everywhere)
    if everywhere - reached:
        raise ValueError("x-mcp-header appears where it cannot be reached through plain 'properties'")
    return result


def _lookup(arguments, path):
    """The value at this property path, or None when it is missing or null."""
    value = arguments
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def header_values_for_call(input_schema, arguments):
    """The Mcp-Param-* headers a client must add to a tools/call. Missing and null parameters are left out."""
    headers = {}
    for header, path, json_type in mirrored_parameters(input_schema):
        value = _lookup(arguments, path)
        if value is None:
            continue
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, int):
            if abs(value) > SAFE_INTEGER:
                raise ValueError(f"'{header}': integer {value} is outside the safe range for a header")
            text = str(value)
        elif isinstance(value, str):
            text = value
        else:
            raise ValueError(f"'{header}': cannot mirror a value of type {type(value).__name__}")
        headers[PARAM_PREFIX + header] = encode_header_value(text)
    return headers


def check_param_headers(input_schema, arguments, headers):
    """Server side: do the Mcp-Param-* headers agree with the arguments in the body? Returns a problem text or None.
    `headers` needs a case-insensitive .get(). Integers are compared as numbers, so 42 and 42.0 agree."""
    try:
        parameters = mirrored_parameters(input_schema)
    except ValueError:
        return None                       # a tool definition we would not have published ourselves; nothing to enforce
    for header, path, json_type in parameters:
        value = _lookup(arguments, path)
        raw = headers.get(PARAM_PREFIX + header)
        if value is None:
            continue                      # null or absent: the client omits the header and we do not expect it
        if raw is None:
            return f"Missing required header {PARAM_PREFIX}{header}"
        try:
            decoded = decode_header_value(raw)
        except ValueError:
            return f"{PARAM_PREFIX}{header} is not valid"
        if json_type == "integer":
            try:
                same = float(decoded) == float(value)
            except ValueError:
                same = False
        elif json_type == "boolean":
            same = decoded == ("true" if value is True else "false")
        else:
            same = decoded == value
        if not same:
            return f"Header {PARAM_PREFIX}{header} does not match the value in the request body"
    return None
