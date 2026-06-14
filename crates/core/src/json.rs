//! Strict JSON parser and writer (RFC 8259), hand-rolled — zero crates.
//!
//! Objects preserve insertion order because entry order is semantic in the IR
//! format ("Order is semantic everywhere", docs/ir-format-v1.md). Int/float
//! lexical identity is preserved: `1` parses to `Int(1)`, `1.0` to
//! `Float(1.0)`. `NaN`/`Infinity` tokens are rejected — they are not JSON,
//! and a non-finite constant in a log density is an upstream bug.

use crate::error::{Error, ErrorKind};

/// A parsed JSON value. Object entries keep document order.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    /// Look up an object key (first match in document order).
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Numeric value as f64, accepting both lexical forms.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }
}

fn err(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedJson, message)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), Error> {
        match self.bump() {
            Some(got) if got == b => Ok(()),
            Some(got) => Err(err(format!(
                "expected '{}' at byte {}, found '{}'",
                b as char,
                self.pos - 1,
                got as char
            ))),
            None => Err(err(format!(
                "expected '{}' at byte {}, found end of input",
                b as char, self.pos
            ))),
        }
    }

    fn parse_value(&mut self) -> Result<Value, Error> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Value::Str(self.parse_string()?)),
            Some(b't') => self.parse_keyword("true", Value::Bool(true)),
            Some(b'f') => self.parse_keyword("false", Value::Bool(false)),
            Some(b'n') => self.parse_keyword("null", Value::Null),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(other) => Err(err(format!(
                "unexpected character '{}' at byte {}; JSON values start with one of {{ [ \" t f n - 0-9",
                other as char, self.pos
            ))),
            None => Err(err("unexpected end of input; expected a JSON value")),
        }
    }

    fn parse_keyword(&mut self, keyword: &str, value: Value) -> Result<Value, Error> {
        let end = self.pos + keyword.len();
        if self.bytes.len() >= end && &self.bytes[self.pos..end] == keyword.as_bytes() {
            self.pos = end;
            Ok(value)
        } else {
            Err(err(format!(
                "invalid token at byte {}; strict JSON accepts only true/false/null keywords \
                 (NaN and Infinity are rejected)",
                self.pos
            )))
        }
    }

    fn parse_object(&mut self) -> Result<Value, Error> {
        self.expect(b'{')?;
        let mut entries: Vec<(String, Value)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.parse_value()?;
            entries.push((key, value));
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(Value::Object(entries)),
                Some(other) => {
                    return Err(err(format!(
                        "expected ',' or '}}' in object at byte {}, found '{}'",
                        self.pos - 1,
                        other as char
                    )))
                }
                None => return Err(err("unterminated object; add the closing '}'")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, Error> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => return Ok(Value::Array(items)),
                Some(other) => {
                    return Err(err(format!(
                        "expected ',' or ']' in array at byte {}, found '{}'",
                        self.pos - 1,
                        other as char
                    )))
                }
                None => return Err(err("unterminated array; add the closing ']'")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, Error> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err(err("unterminated string; add the closing '\"'")),
                Some(b'"') => return Ok(out),
                Some(b'\\') => match self.bump() {
                    Some(b'"') => out.push('"'),
                    Some(b'\\') => out.push('\\'),
                    Some(b'/') => out.push('/'),
                    Some(b'b') => out.push('\u{0008}'),
                    Some(b'f') => out.push('\u{000C}'),
                    Some(b'n') => out.push('\n'),
                    Some(b'r') => out.push('\r'),
                    Some(b't') => out.push('\t'),
                    Some(b'u') => out.push(self.parse_unicode_escape()?),
                    Some(other) => {
                        return Err(err(format!(
                            "invalid escape '\\{}' at byte {}",
                            other as char,
                            self.pos - 1
                        )))
                    }
                    None => return Err(err("unterminated escape at end of input")),
                },
                Some(b) if b < 0x20 => {
                    return Err(err(format!(
                        "unescaped control character 0x{b:02x} in string at byte {}; \
                         escape it as \\u00{b:02X}",
                        self.pos - 1
                    )))
                }
                Some(b) if b < 0x80 => out.push(b as char),
                Some(first) => {
                    // Multi-byte UTF-8 sequence: validate via str conversion.
                    let len = utf8_len(first).ok_or_else(|| {
                        err(format!("invalid UTF-8 lead byte 0x{first:02x} in string"))
                    })?;
                    let start = self.pos - 1;
                    let end = start + len;
                    if end > self.bytes.len() {
                        return Err(err("truncated UTF-8 sequence in string"));
                    }
                    let s = std::str::from_utf8(&self.bytes[start..end])
                        .map_err(|_| err("invalid UTF-8 sequence in string"))?;
                    out.push_str(s);
                    self.pos = end;
                }
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, Error> {
        let first = self.parse_hex4()?;
        if (0xD800..0xDC00).contains(&first) {
            // High surrogate: require a low surrogate escape.
            if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                return Err(err(
                    "lone high surrogate in \\u escape; pair it with a low surrogate",
                ));
            }
            let second = self.parse_hex4()?;
            if !(0xDC00..0xE000).contains(&second) {
                return Err(err(
                    "high surrogate followed by non-low-surrogate in \\u escape",
                ));
            }
            let code = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
            char::from_u32(code).ok_or_else(|| err("invalid surrogate pair in \\u escape"))
        } else if (0xDC00..0xE000).contains(&first) {
            Err(err(
                "lone low surrogate in \\u escape; pair it with a high surrogate",
            ))
        } else {
            char::from_u32(first).ok_or_else(|| err("invalid \\u escape"))
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, Error> {
        let mut value = 0u32;
        for _ in 0..4 {
            let b = self
                .bump()
                .ok_or_else(|| err("truncated \\u escape; four hex digits are required"))?;
            let digit = (b as char)
                .to_digit(16)
                .ok_or_else(|| err(format!("non-hex digit '{}' in \\u escape", b as char)))?;
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Value, Error> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        // Integer part: a single 0, or 1-9 followed by digits. Leading zeros rejected.
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(err(format!(
                        "number with leading zero at byte {start}; strict JSON forbids it"
                    )));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => {
                return Err(err(format!(
                    "invalid number at byte {start}; a digit must follow '-'"
                )))
            }
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(err(format!(
                    "invalid number at byte {start}; a digit must follow '.'"
                )));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(err(format!(
                    "invalid number at byte {start}; a digit must follow the exponent marker"
                )));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("number lexeme is ASCII by construction");
        if is_float {
            let value: f64 = text
                .parse()
                .map_err(|_| err(format!("unparseable float literal '{text}'")))?;
            if !value.is_finite() {
                return Err(err(format!(
                    "float literal '{text}' overflows f64; non-finite values are rejected"
                )));
            }
            Ok(Value::Float(value))
        } else {
            match text.parse::<i64>() {
                Ok(value) => Ok(Value::Int(value)),
                // Integer lexeme too large for i64: keep it as a float.
                Err(_) => {
                    let value: f64 = text
                        .parse()
                        .map_err(|_| err(format!("unparseable integer literal '{text}'")))?;
                    if !value.is_finite() {
                        return Err(err(format!(
                            "integer literal '{text}' overflows f64; non-finite values are rejected"
                        )));
                    }
                    Ok(Value::Float(value))
                }
            }
        }
    }
}

/// Parse a complete JSON document. Trailing non-whitespace is an error.
pub fn parse(text: &str) -> Result<Value, Error> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(err(format!(
            "trailing content at byte {}; a JSON document holds exactly one value",
            parser.pos
        )));
    }
    Ok(value)
}

/// Serialize compactly (no whitespace), rejecting non-finite floats.
pub fn write(value: &Value) -> Result<String, Error> {
    let mut out = String::new();
    write_into(value, &mut out)?;
    Ok(out)
}

fn write_into(value: &Value, out: &mut String) -> Result<(), Error> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::Float(f) => out.push_str(&format_f64(*f)?),
        Value::Str(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_into(item, out)?;
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (i, (key, val)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_into(val, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Shortest round-trip f64 formatting with a guaranteed float lexeme
/// (a `.` or exponent), so int/float identity survives a round trip.
pub fn format_f64(f: f64) -> Result<String, Error> {
    if !f.is_finite() {
        return Err(Error::new(
            ErrorKind::NonFiniteDensity,
            "cannot serialize a non-finite float to JSON",
        ));
    }
    let mut s = format!("{f}");
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    Ok(s)
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn utf8_len(lead: u8) -> Option<usize> {
    match lead {
        0xC0..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF7 => Some(4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars() {
        assert_eq!(parse("null").unwrap(), Value::Null);
        assert_eq!(parse("true").unwrap(), Value::Bool(true));
        assert_eq!(parse("false").unwrap(), Value::Bool(false));
        assert_eq!(parse("42").unwrap(), Value::Int(42));
        assert_eq!(parse("-7").unwrap(), Value::Int(-7));
        assert_eq!(parse("0").unwrap(), Value::Int(0));
    }

    #[test]
    fn preserves_int_float_lexical_identity() {
        assert_eq!(parse("1").unwrap(), Value::Int(1));
        assert_eq!(parse("1.0").unwrap(), Value::Float(1.0));
        assert_eq!(parse("1e3").unwrap(), Value::Float(1000.0));
        assert_eq!(parse("-2.5E-2").unwrap(), Value::Float(-0.025));
    }

    #[test]
    fn parses_strings_with_escapes() {
        assert_eq!(parse(r#""hello""#).unwrap(), Value::Str("hello".into()));
        assert_eq!(
            parse(r#""a\"b\\c\/d\n\t""#).unwrap(),
            Value::Str("a\"b\\c/d\n\t".into())
        );
        assert_eq!(parse(r#""é""#).unwrap(), Value::Str("é".into()));
        // Surrogate pair: U+1D11E musical G clef.
        assert_eq!(parse(r#""𝄞""#).unwrap(), Value::Str("𝄞".into()));
        // Raw multi-byte UTF-8 passes through.
        assert_eq!(parse("\"é\"").unwrap(), Value::Str("é".into()));
    }

    #[test]
    fn parses_containers_preserving_order() {
        let doc = parse(r#"{"b": 1, "a": [2, 3.5, "x", null]}"#).unwrap();
        let Value::Object(entries) = &doc else {
            panic!("expected object")
        };
        assert_eq!(entries[0].0, "b");
        assert_eq!(entries[1].0, "a");
        assert_eq!(
            entries[1].1,
            Value::Array(vec![
                Value::Int(2),
                Value::Float(3.5),
                Value::Str("x".into()),
                Value::Null
            ])
        );
    }

    #[test]
    fn rejects_non_finite_tokens() {
        assert!(parse("NaN").is_err());
        assert!(parse("Infinity").is_err());
        assert!(parse("-Infinity").is_err());
        assert!(parse("1e999").is_err());
    }

    #[test]
    fn rejects_malformed_documents() {
        for text in [
            "",
            "{",
            "[1,",
            "{\"a\":}",
            "01",
            "1.",
            "1.e3",
            "+1",
            "'x'",
            "{\"a\" 1}",
            "[1] []",
            "\"\\q\"",
            "\"\u{0001}\"",
            "tru",
            "\"\\ud834\"",
        ] {
            assert!(parse(text).is_err(), "expected parse error for {text:?}");
        }
    }

    #[test]
    fn lone_low_surrogate_is_rejected() {
        assert!(parse(r#""\udd1e""#).is_err());
    }

    #[test]
    fn writes_round_trip() {
        let doc = parse(r#"{"a":1,"b":[2.5,true,null,"s\n"],"c":{"d":-0.025}}"#).unwrap();
        let text = write(&doc).unwrap();
        assert_eq!(parse(&text).unwrap(), doc);
        assert_eq!(
            text,
            r#"{"a":1,"b":[2.5,true,null,"s\n"],"c":{"d":-0.025}}"#
        );
    }

    #[test]
    fn float_lexeme_survives_round_trip() {
        let text = write(&Value::Float(1.0)).unwrap();
        assert_eq!(text, "1.0");
        assert_eq!(parse(&text).unwrap(), Value::Float(1.0));
    }

    #[test]
    fn huge_integer_lexemes_degrade_to_float() {
        let doc = parse("123456789012345678901234567890").unwrap();
        assert_eq!(doc, Value::Float(1.2345678901234568e29));
    }
}
