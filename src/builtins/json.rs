use super::*;
use crate::save_gc_context;

// ============================================================================
// JSON serialization builtins
// ============================================================================

/// Maximum nesting depth for JSON encode/decode to prevent stack overflow
const JSON_MAX_DEPTH: usize = 512;

/// JSON encoding/decoding error types
#[derive(Debug)]
pub enum JsonError {
    /// Nesting depth exceeded maximum
    DepthExceeded { depth: usize, max: usize },
    /// Value cannot be encoded to JSON
    UnencodableType { type_name: String },
    /// Invalid JSON syntax during parsing
    ParseError { position: usize, message: String },
    /// Unexpected end of input
    UnexpectedEof { position: usize },
    /// Invalid number format
    InvalidNumber { position: usize, value: String },
    /// Invalid escape sequence
    InvalidEscape { position: usize, sequence: String },
    /// Unterminated string
    UnterminatedString { position: usize },
    /// Expected specific character
    Expected {
        position: usize,
        expected: char,
        found: Option<char>,
    },
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonError::DepthExceeded { depth, max } => {
                write!(f, "JSON nesting depth {} exceeds maximum of {}", depth, max)
            }
            JsonError::UnencodableType { type_name } => {
                write!(f, "Cannot encode {} to JSON", type_name)
            }
            JsonError::ParseError { position, message } => {
                write!(f, "JSON parse error at position {}: {}", position, message)
            }
            JsonError::UnexpectedEof { position } => {
                write!(f, "Unexpected end of JSON input at position {}", position)
            }
            JsonError::InvalidNumber { position, value } => {
                write!(f, "Invalid number '{}' at position {}", value, position)
            }
            JsonError::InvalidEscape { position, sequence } => {
                write!(
                    f,
                    "Invalid escape sequence '{}' at position {}",
                    sequence, position
                )
            }
            JsonError::UnterminatedString { position } => {
                write!(f, "Unterminated string starting at position {}", position)
            }
            JsonError::Expected {
                position,
                expected,
                found,
            } => match found {
                Some(c) => write!(
                    f,
                    "Expected '{}' at position {}, found '{}'",
                    expected, position, c
                ),
                None => write!(
                    f,
                    "Expected '{}' at position {}, found end of input",
                    expected, position
                ),
            },
        }
    }
}

/// Escape a string for JSON output according to RFC 8259
pub fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + s.len() / 8);
    for ch in s.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\u{0008}' => result.push_str("\\b"), // backspace
            '\u{000C}' => result.push_str("\\f"), // form feed
            c if c.is_control() => {
                // Escape other control characters as \uXXXX
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

/// Format a float for JSON output with minimal representation that preserves precision.
/// Uses Rust's built-in formatting which implements Grisu3/Dragon4 hybrid algorithm.
pub fn format_float_for_json(value: f64) -> String {
    if value.is_nan() || value.is_infinite() {
        // JSON doesn't support NaN or Infinity - these must be handled by caller
        return "null".to_string();
    }

    // Check if the value is an integer (no fractional part)
    if value.fract() == 0.0 && value.abs() < (i64::MAX as f64) {
        // Format as integer-like but ensure it's still recognized as a float
        // by including .0 when needed
        let int_val = value as i64;
        let formatted = format!("{}", int_val);
        // If parsing back gives same value, use compact form
        if formatted
            .parse::<f64>()
            .map(|v| v == value)
            .unwrap_or(false)
        {
            return format!("{}.0", int_val);
        }
    }

    // Use debug formatting for full precision, then clean up
    let formatted = format!("{:?}", value);

    // Ensure the result parses back to the same value
    if formatted
        .parse::<f64>()
        .map(|v| v == value)
        .unwrap_or(false)
    {
        formatted
    } else {
        // Fall back to maximum precision
        format!("{:.17}", value)
    }
}

/// Convert a Beagle value to JSON string representation.
/// Returns an error for unencodable types or depth exceeded.
pub fn value_to_json(runtime: &Runtime, value: usize, depth: usize) -> Result<String, JsonError> {
    use crate::collections::{GcHandle, PersistentMap, PersistentVec};

    if depth > JSON_MAX_DEPTH {
        return Err(JsonError::DepthExceeded {
            depth,
            max: JSON_MAX_DEPTH,
        });
    }

    let tag = BuiltInTypes::get_kind(value);
    match tag {
        BuiltInTypes::Null => Ok("null".to_string()),
        BuiltInTypes::Int => Ok(BuiltInTypes::untag_isize(value as isize).to_string()),
        BuiltInTypes::Float => {
            let ptr = BuiltInTypes::untag(value) as *const f64;
            let float_value = unsafe { *ptr.add(1) };
            if float_value.is_nan() {
                Err(JsonError::UnencodableType {
                    type_name: "NaN".to_string(),
                })
            } else if float_value.is_infinite() {
                Err(JsonError::UnencodableType {
                    type_name: if float_value.is_sign_positive() {
                        "Infinity"
                    } else {
                        "-Infinity"
                    }
                    .to_string(),
                })
            } else {
                Ok(format_float_for_json(float_value))
            }
        }
        BuiltInTypes::String => {
            let string = runtime.get_string_literal(value);
            Ok(format!("\"{}\"", escape_json_string(&string)))
        }
        BuiltInTypes::Bool => {
            let bool_value = BuiltInTypes::untag(value);
            Ok(if bool_value == 0 { "false" } else { "true" }.to_string())
        }
        BuiltInTypes::Function => Err(JsonError::UnencodableType {
            type_name: "function".to_string(),
        }),
        BuiltInTypes::Closure => Err(JsonError::UnencodableType {
            type_name: "closure".to_string(),
        }),
        BuiltInTypes::HeapObject => {
            let object = HeapObject::from_tagged(value);
            let header = object.get_header();

            match header.type_id {
                1 => {
                    // Raw array
                    let fields = object.get_fields();
                    let mut parts: Vec<String> = Vec::with_capacity(fields.len());
                    for field in fields {
                        parts.push(value_to_json(runtime, *field, depth + 1)?);
                    }
                    Ok(format!("[{}]", parts.join(",")))
                }
                2 | 34 | 35 => {
                    // HeapObject string (flat, slice, or cons)
                    let bytes = runtime.get_string_bytes_vec(value);
                    let string = std::str::from_utf8(&bytes).unwrap_or("");
                    Ok(format!("\"{}\"", escape_json_string(string)))
                }
                3 => {
                    // Keyword - encode as string (without the leading colon)
                    let bytes = object.get_keyword_bytes();
                    let keyword_text = std::str::from_utf8(bytes).unwrap_or("");
                    Ok(format!("\"{}\"", escape_json_string(keyword_text)))
                }
                20 => {
                    // PersistentVector
                    let vec_handle = GcHandle::from_tagged(value);
                    let count = PersistentVec::count(vec_handle);
                    let mut parts: Vec<String> = Vec::with_capacity(count);
                    for i in 0..count {
                        let elem = PersistentVec::get(vec_handle, i);
                        parts.push(value_to_json(runtime, elem, depth + 1)?);
                    }
                    Ok(format!("[{}]", parts.join(",")))
                }
                22 => {
                    // PersistentMap
                    let map_handle = GcHandle::from_tagged(value);
                    let count = PersistentMap::count(map_handle);
                    if count == 0 {
                        Ok("{}".to_string())
                    } else {
                        let entries = runtime.get_map_entries_for_json(map_handle);
                        let mut parts: Vec<String> = Vec::with_capacity(entries.len());
                        for (k, v) in entries {
                            let key_str = value_to_json_key(runtime, k)?;
                            let val_str = value_to_json(runtime, v, depth + 1)?;
                            parts.push(format!("{}:{}", key_str, val_str));
                        }
                        Ok(format!("{{{}}}", parts.join(",")))
                    }
                }
                0 => {
                    // Struct - encode as JSON object with field names
                    let struct_id = object.get_struct_id();
                    if let Some(struct_value) = runtime.get_struct_by_id(struct_id) {
                        let fields = object.get_fields();
                        let mut parts: Vec<String> = Vec::with_capacity(struct_value.fields.len());
                        for (i, field_name) in struct_value.fields.iter().enumerate() {
                            if i < fields.len() {
                                let key_str = format!("\"{}\"", escape_json_string(field_name));
                                let val_str = value_to_json(runtime, fields[i], depth + 1)?;
                                parts.push(format!("{}:{}", key_str, val_str));
                            }
                        }
                        Ok(format!("{{{}}}", parts.join(",")))
                    } else {
                        Err(JsonError::UnencodableType {
                            type_name: "unknown struct".to_string(),
                        })
                    }
                }
                4 => Err(JsonError::UnencodableType {
                    type_name: "closure".to_string(),
                }),
                5 => Err(JsonError::UnencodableType {
                    type_name: "atom".to_string(),
                }),
                _ => Err(JsonError::UnencodableType {
                    type_name: format!("heap object type {}", header.type_id),
                }),
            }
        }
    }
}

/// Convert a value to a JSON object key (must be a string per JSON spec).
/// Returns an error if the key cannot be converted to a string.
pub fn value_to_json_key(runtime: &Runtime, value: usize) -> Result<String, JsonError> {
    let tag = BuiltInTypes::get_kind(value);
    match tag {
        BuiltInTypes::String => {
            let string = runtime.get_string_literal(value);
            Ok(format!("\"{}\"", escape_json_string(&string)))
        }
        BuiltInTypes::HeapObject => {
            let object = HeapObject::from_tagged(value);
            let header = object.get_header();
            match header.type_id {
                2 | 34 | 35 => {
                    // HeapObject string (flat, slice, or cons)
                    let bytes = runtime.get_string_bytes_vec(value);
                    let string = std::str::from_utf8(&bytes).unwrap_or("");
                    Ok(format!("\"{}\"", escape_json_string(string)))
                }
                3 => {
                    // Keyword - use as string key (without colon prefix)
                    let bytes = object.get_keyword_bytes();
                    let keyword_text = std::str::from_utf8(bytes).unwrap_or("");
                    Ok(format!("\"{}\"", escape_json_string(keyword_text)))
                }
                _ => Err(JsonError::UnencodableType {
                    type_name: "non-string map key".to_string(),
                }),
            }
        }
        BuiltInTypes::Int => {
            // Convert integer to string key (common in JavaScript-style objects)
            Ok(format!("\"{}\"", BuiltInTypes::untag_isize(value as isize)))
        }
        _ => Err(JsonError::UnencodableType {
            type_name: format!("map key of type {:?}", tag),
        }),
    }
}

/// json-encode builtin: Convert any Beagle value to a JSON string.
/// Throws JsonError if the value cannot be encoded.
pub extern "C" fn json_encode(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    trace!("handler", "json_encode called");
    let runtime = get_runtime().get_mut();

    match value_to_json(runtime, value, 0) {
        Ok(json_string) => match runtime.allocate_string(stack_pointer, json_string) {
            Ok(ptr) => ptr.into(),
            Err(_) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate JSON string - out of memory".to_string(),
                );
            },
        },
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "JsonError", e.to_string());
        },
    }
}

/// JSON parser state
pub struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        JsonParser {
            bytes: input.as_bytes(),
            pos: 0,
            depth: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        if self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), JsonError> {
        match self.peek() {
            Some(b) if b == expected => {
                self.advance();
                Ok(())
            }
            Some(b) => Err(JsonError::Expected {
                position: self.pos,
                expected: expected as char,
                found: Some(b as char),
            }),
            None => Err(JsonError::Expected {
                position: self.pos,
                expected: expected as char,
                found: None,
            }),
        }
    }

    fn parse_value(
        &mut self,
        runtime: &mut Runtime,
        stack_pointer: usize,
    ) -> Result<usize, JsonError> {
        use crate::collections::{GcHandle, PersistentMap, PersistentVec};

        self.depth += 1;
        if self.depth > JSON_MAX_DEPTH {
            return Err(JsonError::DepthExceeded {
                depth: self.depth,
                max: JSON_MAX_DEPTH,
            });
        }

        self.skip_whitespace();

        let result = match self.peek() {
            None => Err(JsonError::UnexpectedEof { position: self.pos }),
            Some(b'n') => self.parse_null(),
            Some(b't') => self.parse_true(),
            Some(b'f') => self.parse_false(),
            Some(b'"') => self.parse_string(runtime, stack_pointer),
            Some(b'[') => {
                self.advance();
                self.skip_whitespace();

                let empty_vec = PersistentVec::empty(runtime, stack_pointer).map_err(|e| {
                    JsonError::ParseError {
                        position: self.pos,
                        message: e.to_string(),
                    }
                })?;
                let mut vec_id = runtime.register_temporary_root(empty_vec.as_tagged());

                let mut first = true;
                loop {
                    self.skip_whitespace();
                    match self.peek() {
                        Some(b']') => {
                            self.advance();
                            break;
                        }
                        None => return Err(JsonError::UnexpectedEof { position: self.pos }),
                        _ => {
                            if !first {
                                self.expect(b',')?;
                                self.skip_whitespace();
                            }
                            first = false;

                            let elem = self.parse_value(runtime, stack_pointer)?;
                            let elem_id = runtime.register_temporary_root(elem);
                            let vec = runtime.peek_temporary_root(vec_id);
                            let elem = runtime.peek_temporary_root(elem_id);
                            runtime.unregister_temporary_root(elem_id);

                            let new_vec = PersistentVec::push(
                                runtime,
                                stack_pointer,
                                GcHandle::from_tagged(vec),
                                elem,
                            )
                            .map_err(|e| JsonError::ParseError {
                                position: self.pos,
                                message: e.to_string(),
                            })?;

                            runtime.unregister_temporary_root(vec_id);
                            vec_id = runtime.register_temporary_root(new_vec.as_tagged());
                        }
                    }
                }
                Ok(runtime.unregister_temporary_root(vec_id))
            }
            Some(b'{') => {
                self.advance();
                self.skip_whitespace();

                let empty_map = PersistentMap::empty(runtime, stack_pointer).map_err(|e| {
                    JsonError::ParseError {
                        position: self.pos,
                        message: e.to_string(),
                    }
                })?;
                let mut map_id = runtime.register_temporary_root(empty_map.as_tagged());

                let mut first = true;
                loop {
                    self.skip_whitespace();
                    match self.peek() {
                        Some(b'}') => {
                            self.advance();
                            break;
                        }
                        None => return Err(JsonError::UnexpectedEof { position: self.pos }),
                        _ => {
                            if !first {
                                self.expect(b',')?;
                                self.skip_whitespace();
                            }
                            first = false;

                            // Parse key (must be a string)
                            if self.peek() != Some(b'"') {
                                return Err(JsonError::ParseError {
                                    position: self.pos,
                                    message: "Object key must be a string".to_string(),
                                });
                            }
                            let key = self.parse_string(runtime, stack_pointer)?;
                            let key_id = runtime.register_temporary_root(key);

                            self.skip_whitespace();
                            self.expect(b':')?;

                            // Parse value
                            let value = self.parse_value(runtime, stack_pointer)?;
                            let value_id = runtime.register_temporary_root(value);

                            // Insert into map
                            let map = runtime.peek_temporary_root(map_id);
                            let key = runtime.peek_temporary_root(key_id);
                            let value = runtime.peek_temporary_root(value_id);
                            runtime.unregister_temporary_root(value_id);
                            runtime.unregister_temporary_root(key_id);

                            let new_map =
                                PersistentMap::assoc(runtime, stack_pointer, map, key, value)
                                    .map_err(|e| JsonError::ParseError {
                                        position: self.pos,
                                        message: e.to_string(),
                                    })?;

                            runtime.unregister_temporary_root(map_id);
                            map_id = runtime.register_temporary_root(new_map.as_tagged());
                        }
                    }
                }
                Ok(runtime.unregister_temporary_root(map_id))
            }
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(runtime, stack_pointer),
            Some(c) => Err(JsonError::ParseError {
                position: self.pos,
                message: format!("Unexpected character '{}'", c as char),
            }),
        };

        self.depth -= 1;
        result
    }

    fn parse_null(&mut self) -> Result<usize, JsonError> {
        let start = self.pos;
        if self.bytes.len() >= self.pos + 4 && &self.bytes[self.pos..self.pos + 4] == b"null" {
            self.pos += 4;
            Ok(BuiltInTypes::null_value() as usize)
        } else {
            Err(JsonError::ParseError {
                position: start,
                message: "Expected 'null'".to_string(),
            })
        }
    }

    fn parse_true(&mut self) -> Result<usize, JsonError> {
        let start = self.pos;
        if self.bytes.len() >= self.pos + 4 && &self.bytes[self.pos..self.pos + 4] == b"true" {
            self.pos += 4;
            Ok(BuiltInTypes::Bool.tag(1) as usize)
        } else {
            Err(JsonError::ParseError {
                position: start,
                message: "Expected 'true'".to_string(),
            })
        }
    }

    fn parse_false(&mut self) -> Result<usize, JsonError> {
        let start = self.pos;
        if self.bytes.len() >= self.pos + 5 && &self.bytes[self.pos..self.pos + 5] == b"false" {
            self.pos += 5;
            Ok(BuiltInTypes::Bool.tag(0) as usize)
        } else {
            Err(JsonError::ParseError {
                position: start,
                message: "Expected 'false'".to_string(),
            })
        }
    }

    fn parse_string(
        &mut self,
        runtime: &mut Runtime,
        stack_pointer: usize,
    ) -> Result<usize, JsonError> {
        let start = self.pos;
        self.expect(b'"')?;

        let mut string = String::new();
        loop {
            match self.advance() {
                None => return Err(JsonError::UnterminatedString { position: start }),
                Some(b'"') => break,
                Some(b'\\') => {
                    let escape_pos = self.pos - 1;
                    match self.advance() {
                        None => return Err(JsonError::UnterminatedString { position: start }),
                        Some(b'"') => string.push('"'),
                        Some(b'\\') => string.push('\\'),
                        Some(b'/') => string.push('/'),
                        Some(b'b') => string.push('\u{0008}'),
                        Some(b'f') => string.push('\u{000C}'),
                        Some(b'n') => string.push('\n'),
                        Some(b'r') => string.push('\r'),
                        Some(b't') => string.push('\t'),
                        Some(b'u') => {
                            // Parse 4 hex digits
                            if self.pos + 4 > self.bytes.len() {
                                return Err(JsonError::InvalidEscape {
                                    position: escape_pos,
                                    sequence: "\\u (incomplete)".to_string(),
                                });
                            }
                            let hex = &self.bytes[self.pos..self.pos + 4];
                            let hex_str =
                                std::str::from_utf8(hex).map_err(|_| JsonError::InvalidEscape {
                                    position: escape_pos,
                                    sequence: format!("\\u{}", String::from_utf8_lossy(hex)),
                                })?;
                            let code = u32::from_str_radix(hex_str, 16).map_err(|_| {
                                JsonError::InvalidEscape {
                                    position: escape_pos,
                                    sequence: format!("\\u{}", hex_str),
                                }
                            })?;
                            let ch =
                                char::from_u32(code).ok_or_else(|| JsonError::InvalidEscape {
                                    position: escape_pos,
                                    sequence: format!("\\u{}", hex_str),
                                })?;
                            string.push(ch);
                            self.pos += 4;
                        }
                        Some(c) => {
                            return Err(JsonError::InvalidEscape {
                                position: escape_pos,
                                sequence: format!("\\{}", c as char),
                            });
                        }
                    }
                }
                Some(c) if c < 0x20 => {
                    // Control characters must be escaped
                    return Err(JsonError::ParseError {
                        position: self.pos - 1,
                        message: format!("Unescaped control character 0x{:02x}", c),
                    });
                }
                Some(c) => {
                    // Handle UTF-8 properly
                    if c < 0x80 {
                        string.push(c as char);
                    } else {
                        // Multi-byte UTF-8 character
                        self.pos -= 1; // Back up to start of character
                        let remaining = &self.bytes[self.pos..];
                        match std::str::from_utf8(remaining) {
                            Ok(s) => {
                                if let Some(ch) = s.chars().next() {
                                    string.push(ch);
                                    self.pos += ch.len_utf8();
                                }
                            }
                            Err(e) => {
                                // Try to recover by taking valid portion
                                if e.valid_up_to() > 0 {
                                    let valid = &remaining[..e.valid_up_to()];
                                    if let Ok(s) = std::str::from_utf8(valid) {
                                        for ch in s.chars() {
                                            string.push(ch);
                                        }
                                        self.pos += e.valid_up_to();
                                    }
                                } else {
                                    self.pos += 1; // Skip invalid byte
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(runtime
            .allocate_string(stack_pointer, string)
            .map_err(|e| JsonError::ParseError {
                position: start,
                message: e.to_string(),
            })?
            .into())
    }

    fn parse_number(
        &mut self,
        runtime: &mut Runtime,
        stack_pointer: usize,
    ) -> Result<usize, JsonError> {
        let start = self.pos;

        // Consume the number according to JSON spec
        // number = [ "-" ] int [ frac ] [ exp ]
        // int = "0" | ( digit1-9 *digit )
        // frac = "." 1*digit
        // exp = ("e" | "E") ["-" | "+"] 1*digit

        // Optional minus
        if self.peek() == Some(b'-') {
            self.advance();
        }

        // Integer part
        match self.peek() {
            Some(b'0') => {
                self.advance();
                // Leading zeros not allowed (except standalone 0)
            }
            Some(c) if c.is_ascii_digit() => {
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.advance();
                }
            }
            _ => {
                return Err(JsonError::InvalidNumber {
                    position: start,
                    value: String::from_utf8_lossy(
                        &self.bytes[start..self.pos.min(self.bytes.len())],
                    )
                    .to_string(),
                });
            }
        }

        let mut is_float = false;

        // Optional fraction
        if self.peek() == Some(b'.') {
            is_float = true;
            self.advance();
            // Must have at least one digit after decimal point
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(JsonError::InvalidNumber {
                    position: start,
                    value: String::from_utf8_lossy(&self.bytes[start..self.pos]).to_string(),
                });
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.advance();
            }
        }

        // Optional exponent
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.advance();
            // Optional sign
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.advance();
            }
            // Must have at least one digit
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(JsonError::InvalidNumber {
                    position: start,
                    value: String::from_utf8_lossy(&self.bytes[start..self.pos]).to_string(),
                });
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.advance();
            }
        }

        let num_str = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| {
            JsonError::InvalidNumber {
                position: start,
                value: String::from_utf8_lossy(&self.bytes[start..self.pos]).to_string(),
            }
        })?;

        if is_float {
            let f: f64 = num_str.parse().map_err(|_| JsonError::InvalidNumber {
                position: start,
                value: num_str.to_string(),
            })?;
            let float_ptr = runtime
                .allocate(1, stack_pointer, BuiltInTypes::Float)
                .map_err(|e| JsonError::ParseError {
                    position: start,
                    message: e.to_string(),
                })?;
            let untagged = BuiltInTypes::untag(float_ptr);
            let ptr = untagged as *mut f64;
            unsafe {
                *ptr.add(1) = f;
            }
            Ok(float_ptr)
        } else {
            let i: i64 = num_str.parse().map_err(|_| JsonError::InvalidNumber {
                position: start,
                value: num_str.to_string(),
            })?;
            Ok(BuiltInTypes::Int.tag(i as isize) as usize)
        }
    }
}

/// Parse a JSON string and return the corresponding Beagle value.
pub fn parse_json(
    runtime: &mut Runtime,
    stack_pointer: usize,
    json: &str,
) -> Result<usize, JsonError> {
    let mut parser = JsonParser::new(json);
    let result = parser.parse_value(runtime, stack_pointer)?;

    // Check for trailing content (not allowed in strict JSON)
    parser.skip_whitespace();
    if parser.peek().is_some() {
        return Err(JsonError::ParseError {
            position: parser.pos,
            message: "Unexpected content after JSON value".to_string(),
        });
    }

    Ok(result)
}

/// json-decode builtin: Parse a JSON string and return the corresponding Beagle value.
/// Throws JsonError if the JSON is invalid.
pub extern "C" fn json_decode(
    stack_pointer: usize,
    frame_pointer: usize,
    json_str: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let json = runtime.get_string(stack_pointer, json_str);

    match parse_json(runtime, stack_pointer, &json) {
        Ok(result) => result,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "JsonError", e.to_string());
        },
    }
}
