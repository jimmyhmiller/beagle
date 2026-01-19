// Stolen from my editor, so probably not great
// Need to deal with failure?
// Maybe not at the token level?

use crate::{
    Data,
    ast::{
        ArityCase, Ast, FieldPattern, MapFieldPattern, MapKey, MatchArm, Pattern,
        StringInterpolationPart, TokenRange,
    },
    builtins::debugger,
};
use std::{error::Error, fmt};

#[derive(Debug, Clone)]
pub enum ParseError {
    InvalidUtf8 {
        position: usize,
    },
    UnexpectedToken {
        expected: String,
        found: String,
        position: usize,
    },
    UnexpectedEof {
        expected: String,
    },
    InvalidNumberLiteral {
        literal: String,
        position: usize,
    },
    InvalidStringEscape {
        position: usize,
    },
    MissingToken {
        expected: String,
        position: usize,
    },
    InvalidPattern {
        message: String,
        position: usize,
    },
    InvalidExpression {
        message: String,
        position: usize,
    },
    InvalidDeclaration {
        message: String,
        position: usize,
    },
    IoError {
        message: String,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidUtf8 { position } => {
                write!(f, "Invalid UTF-8 at position {}", position)
            }
            ParseError::UnexpectedToken {
                expected,
                found,
                position,
            } => {
                write!(
                    f,
                    "Expected {} but found {} at position {}",
                    expected, found, position
                )
            }
            ParseError::UnexpectedEof { expected } => {
                write!(f, "Unexpected end of file, expected {}", expected)
            }
            ParseError::InvalidNumberLiteral { literal, position } => {
                write!(
                    f,
                    "Invalid number literal '{}' at position {}",
                    literal, position
                )
            }
            ParseError::InvalidStringEscape { position } => {
                write!(f, "Invalid string escape sequence at position {}", position)
            }
            ParseError::MissingToken { expected, position } => {
                write!(f, "Missing {} at position {}", expected, position)
            }
            ParseError::InvalidPattern { message, position } => {
                write!(f, "Invalid pattern: {} at position {}", message, position)
            }
            ParseError::InvalidExpression { message, position } => {
                write!(
                    f,
                    "Invalid expression: {} at position {}",
                    message, position
                )
            }
            ParseError::InvalidDeclaration { message, position } => {
                write!(
                    f,
                    "Invalid declaration: {} at position {}",
                    message, position
                )
            }
            ParseError::IoError { message } => {
                write!(f, "IO error: {}", message)
            }
        }
    }
}

impl Error for ParseError {}

impl From<std::io::Error> for ParseError {
    fn from(err: std::io::Error) -> Self {
        ParseError::IoError {
            message: err.to_string(),
        }
    }
}

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    OpenParen,
    CloseParen,
    OpenCurly,
    CloseCurly,
    HashOpenCurly,
    OpenBracket,
    CloseBracket,
    SemiColon,
    Colon,
    Comma,
    Dot,
    DotDotDot,
    NewLine,
    Fn,
    Loop,
    While,
    Break,
    Continue,
    If,
    Else,
    LessThanOrEqual,
    LessThan,
    Equal,
    EqualEqual,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Plus,
    Minus,
    Mul,
    Div,
    Concat,
    True,
    False,
    Null,
    Infinity,
    NegativeInfinity,
    ShiftRight,
    ShiftRightZero,
    ShiftLeft,
    BitWiseAnd,
    BitWiseOr,
    BitWiseXor,
    Or,
    Let,
    Struct,
    Enum,
    Comment((usize, usize)),
    Spaces((usize, usize)),
    String((usize, usize)),
    /// Interpolated string: contains segments that are either string literals or expressions
    /// Each segment is (start, end, is_expression) - if is_expression is true, the bytes represent
    /// an expression to be parsed; otherwise they represent a string literal
    InterpolatedString(Vec<(usize, usize, bool)>),
    Integer((usize, usize)),
    Float((usize, usize)),
    // I should replace this with builtins
    // like fn and stuff
    Atom((usize, usize)),
    Keyword((usize, usize)),
    Never,
    Namespace,
    Import,
    As,
    And,
    Protocol,
    Extend,
    With,
    Mut,
    Try,
    Catch,
    Throw,
    Match,
    Arrow,
    Underscore,
    For,
    In,
    Pipe,
    PipeLast,
    Modulo,
    Reset,
    Shift,
}
impl Token {
    fn is_binary_operator(&self) -> bool {
        match self {
            Token::LessThanOrEqual
            | Token::LessThan
            | Token::EqualEqual
            | Token::NotEqual
            | Token::GreaterThan
            | Token::GreaterThanOrEqual
            | Token::Plus
            | Token::Minus
            | Token::Mul
            | Token::Div
            | Token::Concat
            | Token::ShiftRight
            | Token::ShiftRightZero
            | Token::ShiftLeft
            | Token::BitWiseAnd
            | Token::BitWiseOr
            | Token::BitWiseXor
            | Token::Or
            | Token::And
            | Token::Equal
            | Token::Pipe
            | Token::PipeLast
            | Token::Modulo => true,
            _ => false,
        }
    }

    fn literal(&self, input_bytes: &[u8]) -> ParseResult<String> {
        match self {
            Token::OpenParen => Ok("(".to_string()),
            Token::CloseParen => Ok(")".to_string()),
            Token::OpenCurly => Ok("{".to_string()),
            Token::CloseCurly => Ok("}".to_string()),
            Token::HashOpenCurly => Ok("#{".to_string()),
            Token::OpenBracket => Ok("[".to_string()),
            Token::CloseBracket => Ok("]".to_string()),
            Token::SemiColon => Ok(";".to_string()),
            Token::Colon => Ok(":".to_string()),
            Token::Comma => Ok(",".to_string()),
            Token::Dot => Ok(".".to_string()),
            Token::DotDotDot => Ok("...".to_string()),
            Token::NewLine => Ok("\n".to_string()),
            Token::Fn => Ok("fn".to_string()),
            Token::And => Ok("&&".to_string()),
            Token::Or => Ok("||".to_string()),
            Token::LessThanOrEqual => Ok("<=".to_string()),
            Token::LessThan => Ok("<".to_string()),
            Token::Equal => Ok("=".to_string()),
            Token::EqualEqual => Ok("==".to_string()),
            Token::NotEqual => Ok("!=".to_string()),
            Token::GreaterThan => Ok(">".to_string()),
            Token::GreaterThanOrEqual => Ok(">=".to_string()),
            Token::Plus => Ok("+".to_string()),
            Token::Minus => Ok("-".to_string()),
            Token::Mul => Ok("*".to_string()),
            Token::Div => Ok("/".to_string()),
            Token::Concat => Ok("++".to_string()),
            Token::True => Ok("true".to_string()),
            Token::False => Ok("false".to_string()),
            Token::Null => Ok("null".to_string()),
            Token::Infinity => Ok("infinity".to_string()),
            Token::NegativeInfinity => Ok("-infinity".to_string()),
            Token::ShiftRight => Ok(">>".to_string()),
            Token::ShiftRightZero => Ok(">>>".to_string()),
            Token::ShiftLeft => Ok("<<".to_string()),
            Token::BitWiseAnd => Ok("&".to_string()),
            Token::BitWiseOr => Ok("|".to_string()),
            Token::BitWiseXor => Ok("^".to_string()),
            Token::Loop => Ok("loop".to_string()),
            Token::While => Ok("while".to_string()),
            Token::Break => Ok("break".to_string()),
            Token::Continue => Ok("continue".to_string()),
            Token::If => Ok("if".to_string()),
            Token::Else => Ok("else".to_string()),
            Token::Let => Ok("let".to_string()),
            Token::Mut => Ok("mut".to_string()),
            Token::Struct => Ok("struct".to_string()),
            Token::Enum => Ok("enum".to_string()),
            Token::Namespace => Ok("namespace".to_string()),
            Token::Import => Ok("import".to_string()),
            Token::Protocol => Ok("protocol".to_string()),
            Token::Extend => Ok("extend".to_string()),
            Token::As => Ok("as".to_string()),
            Token::With => Ok("with".to_string()),
            Token::Try => Ok("try".to_string()),
            Token::Catch => Ok("catch".to_string()),
            Token::Throw => Ok("throw".to_string()),
            Token::Match => Ok("match".to_string()),
            Token::Arrow => Ok("=>".to_string()),
            Token::Underscore => Ok("_".to_string()),
            Token::For => Ok("for".to_string()),
            Token::In => Ok("in".to_string()),
            Token::Pipe => Ok("|>".to_string()),
            Token::PipeLast => Ok("|>>".to_string()),
            Token::Modulo => Ok("%".to_string()),
            Token::Reset => Ok("reset".to_string()),
            Token::Shift => Ok("shift".to_string()),
            Token::Comment((start, end))
            | Token::Atom((start, end))
            | Token::Keyword((start, end))
            | Token::Spaces((start, end))
            | Token::String((start, end))
            | Token::Integer((start, end))
            | Token::Float((start, end)) => String::from_utf8(input_bytes[*start..*end].to_vec())
                .map_err(|_| ParseError::InvalidUtf8 { position: *start }),
            Token::InterpolatedString(_) => Ok("<interpolated string>".to_string()),
            Token::Never => Err(ParseError::InvalidExpression {
                message: "Internal error: Token::Never should not be used".to_string(),
                position: 0,
            }),
        }
    }
}

enum Associativity {
    Left,
}

static ZERO: u8 = b'0';
static NINE: u8 = b'9';
static SPACE: u8 = b' ';
static NEW_LINE: u8 = b'\n';
static DOUBLE_QUOTE: u8 = b'"';
static OPEN_PAREN: u8 = b'(';
static CLOSE_PAREN: u8 = b')';
static PERIOD: u8 = b'.';
static NEGATIVE: u8 = b'-';

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub position: usize,
    pub line: usize,
    pub column: usize,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

fn stripslashes(s: &str) -> String {
    let mut n = String::new();

    let mut chars = s.chars();

    while let Some(c) = chars.next() {
        n.push(match c {
            '\\' => {
                let next = chars.next();
                if let Some(c) = next {
                    match c {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '0' => '\0',
                        _ => c,
                    }
                } else {
                    c
                }
            }
            c => c,
        });
    }

    n
}

impl Tokenizer {
    pub fn new() -> Tokenizer {
        Tokenizer {
            position: 0,
            line: 1,
            column: 1,
        }
    }

    fn peek(&self, input_bytes: &[u8]) -> Option<u8> {
        if self.position + 1 < input_bytes.len() {
            Some(input_bytes[self.position + 1])
        } else {
            None
        }
    }

    fn is_comment_start(&self, input_bytes: &[u8]) -> bool {
        input_bytes[self.position] == b'/' && self.peek(input_bytes) == Some(b'/')
    }

    fn parse_comment(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
            self.consume(input_bytes);
        }
        Token::Comment((start, self.position))
    }

    pub fn consume(&mut self, input_bytes: &[u8]) {
        if self.current_byte(input_bytes) == NEW_LINE {
            self.increment_line();
            self.reset_column();
        } else {
            self.increment_column();
        }
        self.position += 1;
    }

    fn increment_line(&mut self) {
        self.line += 1;
    }

    fn reset_column(&mut self) {
        self.column = 1;
    }

    fn increment_column(&mut self) {
        self.column += 1;
    }

    pub fn current_byte(&self, input_bytes: &[u8]) -> u8 {
        input_bytes[self.position]
    }

    pub fn next_n_bytes(self, n: usize, input_bytes: &[u8]) -> &[u8] {
        // truncate if n is too large
        &input_bytes[self.position..std::cmp::min(self.position + n, input_bytes.len())]
    }

    pub fn is_space(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == SPACE
    }

    pub fn at_end(&self, input_bytes: &[u8]) -> bool {
        self.position >= input_bytes.len()
    }

    pub fn is_quote(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == DOUBLE_QUOTE
    }

    pub fn parse_string(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        self.consume(input_bytes); // consume opening quote

        let mut segments: Vec<(usize, usize, bool)> = Vec::new();
        let mut has_interpolation = false;
        let mut current_string_start = self.position;

        while !self.at_end(input_bytes) && !self.is_quote(input_bytes) {
            // Check for escape sequences
            if self.current_byte(input_bytes) == b'\\' {
                if let Some(next) = self.peek(input_bytes) {
                    // Handle escaped $ to allow literal ${
                    if next == b'$'
                        || next == b'"'
                        || next == b'\\'
                        || next == b'n'
                        || next == b'r'
                        || next == b't'
                        || next == b'0'
                    {
                        self.consume(input_bytes); // consume backslash
                        self.consume(input_bytes); // consume escaped char
                        continue;
                    }
                }
            }

            // Check for interpolation start: ${
            if self.current_byte(input_bytes) == b'$' && self.peek(input_bytes) == Some(b'{') {
                has_interpolation = true;

                // Save the string segment before the interpolation (if any)
                if self.position > current_string_start {
                    segments.push((current_string_start, self.position, false));
                }

                self.consume(input_bytes); // consume $
                self.consume(input_bytes); // consume {

                let expr_start = self.position;
                let mut brace_depth = 1;

                // Find the matching closing brace, handling nested braces
                while !self.at_end(input_bytes) && brace_depth > 0 {
                    let byte = self.current_byte(input_bytes);
                    if byte == b'{' {
                        brace_depth += 1;
                        self.consume(input_bytes);
                    } else if byte == b'}' {
                        brace_depth -= 1;
                        if brace_depth > 0 {
                            self.consume(input_bytes);
                        }
                    } else if byte == b'"' {
                        // Skip nested strings in expressions
                        self.consume(input_bytes); // opening quote
                        while !self.at_end(input_bytes) && self.current_byte(input_bytes) != b'"' {
                            if self.current_byte(input_bytes) == b'\\' {
                                self.consume(input_bytes); // escape char
                            }
                            self.consume(input_bytes);
                        }
                        if !self.at_end(input_bytes) {
                            self.consume(input_bytes); // consume closing quote
                        }
                    } else {
                        self.consume(input_bytes);
                    }
                }

                let expr_end = self.position;
                segments.push((expr_start, expr_end, true));

                if !self.at_end(input_bytes) {
                    self.consume(input_bytes); // consume closing }
                }

                current_string_start = self.position;
                continue;
            }

            self.consume(input_bytes);
        }

        // If we have interpolation, add the final string segment and return InterpolatedString
        if has_interpolation {
            if self.position > current_string_start {
                segments.push((current_string_start, self.position, false));
            }
            if !self.at_end(input_bytes) {
                self.consume(input_bytes); // consume closing quote
            }
            Token::InterpolatedString(segments)
        } else {
            // Regular string
            if !self.at_end(input_bytes) {
                self.consume(input_bytes); // consume closing quote
            }
            Token::String((start, self.position))
        }
    }

    pub fn is_open_paren(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == OPEN_PAREN
    }

    pub fn is_close_paren(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == CLOSE_PAREN
    }

    pub fn is_open_curly(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'{'
    }

    pub fn is_close_curly(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'}'
    }

    pub fn is_open_bracket(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'['
    }

    pub fn is_close_bracket(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b']'
    }

    pub fn parse_spaces(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes) && self.is_space(input_bytes) {
            self.consume(input_bytes);
        }
        Token::Spaces((start, self.position))
    }

    pub fn is_valid_number_char(&mut self, input_bytes: &[u8]) -> bool {
        (self.current_byte(input_bytes) >= ZERO && self.current_byte(input_bytes) <= NINE)
            || (self.current_byte(input_bytes) == PERIOD
                && self.peek(input_bytes).unwrap() >= ZERO
                && self.peek(input_bytes).unwrap() <= NINE)
            || (self.current_byte(input_bytes) == NEGATIVE
                && self.peek(input_bytes).unwrap() >= ZERO
                && self.peek(input_bytes).unwrap() <= NINE)
    }

    pub fn parse_number(&mut self, input_bytes: &[u8]) -> Result<Token, ParseError> {
        let mut is_float = false;
        let start = self.position;
        while !self.at_end(input_bytes)
            && (self.is_valid_number_char(input_bytes) || self.current_byte(input_bytes) == PERIOD)
        {
            if self.current_byte(input_bytes) == PERIOD {
                if is_float {
                    // Multiple dots in number literal - this is an error
                    let literal = String::from_utf8(input_bytes[start..self.position].to_vec())
                        .unwrap_or_else(|_| "<invalid utf8>".to_string());
                    return Err(ParseError::InvalidNumberLiteral {
                        literal: format!("{}. (multiple decimal points)", literal),
                        position: start,
                    });
                }
                is_float = true;
            }
            self.consume(input_bytes);
        }
        if is_float {
            Ok(Token::Float((start, self.position)))
        } else {
            Ok(Token::Integer((start, self.position)))
        }
    }

    fn is_pipe(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'|'
    }

    pub fn parse_identifier(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes)
            && !self.is_space(input_bytes)
            && !self.is_open_paren(input_bytes)
            && !self.is_close_paren(input_bytes)
            && !self.is_open_curly(input_bytes)
            && !self.is_close_curly(input_bytes)
            && !self.is_open_bracket(input_bytes)
            && !self.is_close_bracket(input_bytes)
            && !self.is_semi_colon(input_bytes)
            && !self.is_colon(input_bytes)
            && !self.is_comma(input_bytes)
            && !self.is_newline(input_bytes)
            && !self.is_quote(input_bytes)
            && !self.is_dot(input_bytes)
            && !self.is_pipe(input_bytes)
        {
            self.consume(input_bytes);
        }
        match &input_bytes[start..self.position] {
            b"fn" => Token::Fn,
            b"loop" => Token::Loop,
            b"while" => Token::While,
            b"break" => Token::Break,
            b"continue" => Token::Continue,
            b"if" => Token::If,
            b"else" => Token::Else,
            b"<=" => Token::LessThanOrEqual,
            b"<" => Token::LessThan,
            b"=" => Token::Equal,
            b"==" => Token::EqualEqual,
            b"=>" => Token::Arrow,
            b"!=" => Token::NotEqual,
            b">" => Token::GreaterThan,
            b">=" => Token::GreaterThanOrEqual,
            b"+" => Token::Plus,
            b"++" => Token::Concat,
            b"-" => Token::Minus,
            b"*" => Token::Mul,
            b"/" => Token::Div,
            b"%" => Token::Modulo,
            b">>" => Token::ShiftRight,
            b">>>" => Token::ShiftRightZero,
            b"<<" => Token::ShiftLeft,
            b"&" => Token::BitWiseAnd,
            b"|" => Token::BitWiseOr,
            b"|>" => Token::Pipe,
            b"|>>" => Token::PipeLast,
            b"^" => Token::BitWiseXor,
            b"||" => Token::Or,
            b"&&" => Token::And,
            b"true" => Token::True,
            b"false" => Token::False,
            b"null" => Token::Null,
            b"infinity" => Token::Infinity,
            b"-infinity" => Token::NegativeInfinity,
            b"let" => Token::Let,
            b"mut" => Token::Mut,
            b"struct" => Token::Struct,
            b"enum" => Token::Enum,
            b"." => Token::Dot,
            b"namespace" => Token::Namespace,
            b"import" => Token::Import,
            b"protocol" => Token::Protocol,
            b"extend" => Token::Extend,
            b"with" => Token::With,
            b"as" => Token::As,
            b"try" => Token::Try,
            b"catch" => Token::Catch,
            b"throw" => Token::Throw,
            b"match" => Token::Match,
            b"_" => Token::Underscore,
            b"for" => Token::For,
            b"reset" => Token::Reset,
            b"shift" => Token::Shift,
            _ => Token::Atom((start, self.position)),
        }
    }

    pub fn parse_keyword(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes)
            && !self.is_space(input_bytes)
            && !self.is_open_paren(input_bytes)
            && !self.is_close_paren(input_bytes)
            && !self.is_open_curly(input_bytes)
            && !self.is_close_curly(input_bytes)
            && !self.is_open_bracket(input_bytes)
            && !self.is_close_bracket(input_bytes)
            && !self.is_semi_colon(input_bytes)
            && !self.is_comma(input_bytes)
            && !self.is_newline(input_bytes)
        {
            self.consume(input_bytes);
        }
        Token::Keyword((start, self.position))
    }

    pub fn parse_single(&mut self, input_bytes: &[u8]) -> Result<Option<Token>, ParseError> {
        if self.at_end(input_bytes) {
            return Ok(None);
        }
        let result = if self.is_space(input_bytes) {
            self.parse_spaces(input_bytes)
        } else if self.is_newline(input_bytes) {
            self.consume(input_bytes);
            Token::NewLine
        } else if self.is_comment_start(input_bytes) {
            self.parse_comment(input_bytes)
        } else if self.is_open_paren(input_bytes) {
            self.consume(input_bytes);
            Token::OpenParen
        } else if self.is_close_paren(input_bytes) {
            self.consume(input_bytes);
            Token::CloseParen
        } else if self.is_valid_number_char(input_bytes) {
            self.parse_number(input_bytes)?
        } else if self.is_quote(input_bytes) {
            self.parse_string(input_bytes)
        } else if self.is_semi_colon(input_bytes) {
            self.consume(input_bytes);
            Token::SemiColon
        } else if self.is_comma(input_bytes) {
            self.consume(input_bytes);
            Token::Comma
        } else if self.is_colon(input_bytes) {
            self.consume(input_bytes);
            // Look ahead to see if this is a keyword (:identifier)
            if !self.at_end(input_bytes)
                && !self.is_space(input_bytes)
                && !self.is_open_paren(input_bytes)
                && !self.is_close_paren(input_bytes)
                && !self.is_open_curly(input_bytes)
                && !self.is_close_curly(input_bytes)
                && !self.is_colon(input_bytes)
                && !self.is_newline(input_bytes)
            {
                // This is a keyword like :foo
                self.parse_keyword(input_bytes)
            } else {
                // This is a standalone colon (struct field separator)
                Token::Colon
            }
        } else if self.current_byte(input_bytes) == b'#'
            && self.position + 1 < input_bytes.len()
            && input_bytes[self.position + 1] == b'{'
        {
            self.consume(input_bytes); // #
            self.consume(input_bytes); // {
            Token::HashOpenCurly
        } else if self.is_open_curly(input_bytes) {
            self.consume(input_bytes);
            Token::OpenCurly
        } else if self.is_close_curly(input_bytes) {
            self.consume(input_bytes);
            Token::CloseCurly
        } else if self.is_open_bracket(input_bytes) {
            self.consume(input_bytes);
            Token::OpenBracket
        } else if self.is_close_bracket(input_bytes) {
            self.consume(input_bytes);
            Token::CloseBracket
        } else if self.is_dot_dot_dot(input_bytes) {
            self.consume(input_bytes);
            self.consume(input_bytes);
            self.consume(input_bytes);
            Token::DotDotDot
        } else if self.is_dot(input_bytes) {
            self.consume(input_bytes);
            Token::Dot
        } else if self.is_pipe(input_bytes) {
            self.consume(input_bytes);
            // Check for multi-character pipe operators
            if !self.at_end(input_bytes) {
                if self.current_byte(input_bytes) == b'|' {
                    self.consume(input_bytes);
                    Token::Or // ||
                } else if self.current_byte(input_bytes) == b'>' {
                    self.consume(input_bytes);
                    if !self.at_end(input_bytes) && self.current_byte(input_bytes) == b'>' {
                        self.consume(input_bytes);
                        Token::PipeLast // |>>
                    } else {
                        Token::Pipe // |>
                    }
                } else {
                    Token::BitWiseOr // |
                }
            } else {
                Token::BitWiseOr // | at end
            }
        } else {
            // println!("identifier");
            self.parse_identifier(input_bytes)
        };
        Ok(Some(result))
    }

    pub fn is_semi_colon(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b';'
    }

    pub fn is_colon(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b':'
    }

    pub fn is_newline(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == NEW_LINE
    }

    pub fn is_comma(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b','
    }

    pub fn is_dot(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'.'
    }

    pub fn is_dot_dot_dot(&self, input_bytes: &[u8]) -> bool {
        self.position + 2 < input_bytes.len()
            && input_bytes[self.position] == b'.'
            && input_bytes[self.position + 1] == b'.'
            && input_bytes[self.position + 2] == b'.'
    }

    // TODO: Make a lazy method of tokenizing
    #[allow(clippy::type_complexity)]
    pub fn parse_all(
        &mut self,
        input_bytes: &[u8],
    ) -> Result<(Vec<Token>, Vec<(usize, usize)>), ParseError> {
        let mut result = Vec::new();
        let mut token_line_column_map = Vec::new();
        while !self.at_end(input_bytes) {
            if let Some(token) = self.parse_single(input_bytes)? {
                result.push(token);
                token_line_column_map.push((self.line, self.column));
            }
        }
        self.position = 0;
        Ok((result, token_line_column_map))
    }
}

#[test]
fn test_tokenizer1() {
    let mut tokenizer = Tokenizer::new();
    let input = "hello world";
    let input_bytes = input.as_bytes();
    let result = tokenizer.parse_all(input_bytes).unwrap();
    assert_eq!(result.0.len(), 3);
    assert_eq!(result.0[0], Token::Atom((0, 5)));
    assert_eq!(result.0[1], Token::Spaces((5, 6)));
    assert_eq!(result.0[2], Token::Atom((6, 11)));
}

pub struct Parser {
    file_name: String,
    source: String,
    position: usize,
    tokens: Vec<Token>,
    token_line_column_map: Vec<(usize, usize)>,
}

impl Parser {
    pub fn new(file_name: String, source: String) -> ParseResult<Parser> {
        let mut tokenizer = Tokenizer::new();
        let input_bytes = source.as_bytes();
        // TODO: It is probably better not to parse all at once
        let (tokens, token_line_column_map) = tokenizer.parse_all(input_bytes)?;

        debugger(crate::Message {
            kind: "tokens".to_string(),
            data: Data::Tokens {
                file_name: file_name.clone(),
                tokens: tokens
                    .clone()
                    .iter()
                    .filter_map(|x| x.literal(input_bytes).ok())
                    .collect(),
                token_line_column_map: token_line_column_map.clone(),
            },
        });

        Ok(Parser {
            file_name,
            source,
            position: 0,
            tokens,
            token_line_column_map,
        })
    }

    pub fn current_location(&self) -> String {
        let (line, column) = self.token_line_column_map[self.position];
        format!("{}:{}:{}", self.file_name, line, column)
    }

    pub fn get_token_line_column_map(&self) -> Vec<(usize, usize)> {
        self.token_line_column_map.clone()
    }

    pub fn print_tokens(&self) {
        for token in &self.tokens {
            println!("{:?}", token);
        }
    }

    // Helper methods for error handling
    fn expect_atom(&self) -> ParseResult<String> {
        match self.current_token() {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "identifier".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            }),
        }
    }

    fn expect_string(&self) -> ParseResult<String> {
        match self.current_token() {
            Token::String((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start + 1..end - 1].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "string".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            }),
        }
    }

    pub fn parse(&mut self) -> ParseResult<Ast> {
        Ok(Ast::Program {
            elements: self.parse_elements()?,
            token_range: TokenRange::new(0, self.tokens.len()),
        })
    }

    fn parse_elements(&mut self) -> ParseResult<Vec<Ast>> {
        let mut result = Vec::new();
        while !self.at_end() {
            match self.parse_expression(0, true, true) {
                Ok(Some(elem)) => result.push(elem),
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(result)
    }

    fn at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    fn get_precedence(&self) -> (usize, Associativity) {
        match self.current_token() {
            // Pipe operators have the lowest precedence.
            Token::Pipe | Token::PipeLast => (5, Associativity::Left),
            // Logical OR (||) has the lowest precedence among common operators.
            Token::Or => (10, Associativity::Left),
            // Logical AND (&&) comes after OR.
            Token::And => (20, Associativity::Left),
            // Comparison operators.
            Token::LessThanOrEqual
            | Token::LessThan
            | Token::EqualEqual
            | Token::NotEqual
            | Token::GreaterThan
            | Token::GreaterThanOrEqual => (30, Associativity::Left),
            // Addition and subtraction.
            Token::Plus | Token::Minus => (40, Associativity::Left),
            // Multiplication, division, modulo.
            Token::Mul | Token::Div | Token::Modulo => (50, Associativity::Left),
            // Bitwise operations (lower precedence than arithmetic).
            Token::BitWiseOr => (60, Associativity::Left),
            Token::BitWiseXor => (70, Associativity::Left),
            Token::BitWiseAnd => (80, Associativity::Left),
            // Shift operations.
            Token::ShiftLeft | Token::ShiftRight | Token::ShiftRightZero => {
                (90, Associativity::Left)
            }

            // Dot (e.g., for member access) should have very high precedence.
            Token::Dot | Token::OpenBracket | Token::OpenCurly => (100, Associativity::Left),
            // Default for unrecognized tokens.
            _ => (0, Associativity::Left),
        }
    }

    // Based on
    // https://eli.thegreenplace.net/2012/08/02/parsing-expressions-by-precedence-climbing
    fn parse_expression(
        &mut self,
        min_precedence: usize,
        should_skip: bool,
        struct_creation_allowed: bool,
    ) -> ParseResult<Option<Ast>> {
        let mut min_precedence = min_precedence;
        if should_skip {
            self.skip_whitespace();
        }
        if self.at_end() {
            return Ok(None);
        }

        let mut lhs = match self.parse_atom(min_precedence)? {
            Some(ast) => ast,
            None => return Ok(None),
        };
        // TODO: this is ugly
        self.skip_spaces();

        let old_min_precedence = min_precedence;
        while self.is_postfix(&lhs, struct_creation_allowed)
            && self.get_precedence().0 > min_precedence
        {
            let (precedence, associativity) = self.get_precedence();
            let next_min_precedence = if matches!(associativity, Associativity::Left) {
                precedence + 1
            } else {
                precedence
            };
            lhs = match self.parse_postfix(lhs, next_min_precedence, struct_creation_allowed)? {
                Some(ast) => ast,
                None => return Ok(None),
            };
            self.skip_spaces();
        }
        min_precedence = old_min_precedence;

        // TODO: This is ugly
        self.skip_spaces();
        loop {
            // Check if we should look ahead across newlines for a binary operator.
            // This allows any binary operator to appear at the start of a new line,
            // enabling multi-line expressions like: 1 + 2 \n + 3
            if self.is_newline() {
                let saved_pos = self.position;
                self.skip_whitespace();
                if !self.at_end() && self.current_token().is_binary_operator() {
                    // There's a binary operator - continue with it
                } else {
                    // No binary operator - restore position and break
                    self.position = saved_pos;
                    break;
                }
            }

            if self.at_end()
                || !self.current_token().is_binary_operator()
                || self.get_precedence().0 < min_precedence
            {
                break;
            }

            let current_token = self.current_token();

            let (precedence, associativity) = self.get_precedence();
            let next_min_precedence = if matches!(associativity, Associativity::Left) {
                precedence + 1
            } else {
                precedence
            };

            self.move_to_next_non_whitespace();
            let rhs =
                match self.parse_expression(next_min_precedence, true, struct_creation_allowed)? {
                    Some(ast) => ast,
                    None => {
                        return Err(ParseError::UnexpectedEof {
                            expected: "expression after binary operator".to_string(),
                        });
                    }
                };

            lhs = self.compose_binary_op(lhs.clone(), current_token, rhs)?;
        }

        Ok(Some(lhs))
    }

    fn parse_atom(&mut self, min_precedence: usize) -> ParseResult<Option<Ast>> {
        match self.current_token() {
            Token::Fn => Ok(Some(self.parse_function()?)),
            Token::Loop => Ok(Some(self.parse_loop()?)),
            Token::While => Ok(Some(self.parse_while()?)),
            Token::Break => Ok(Some(self.parse_break()?)),
            Token::Continue => Ok(Some(self.parse_continue()?)),
            Token::For => Ok(Some(self.parse_for()?)),
            Token::Struct => Ok(Some(self.parse_struct()?)),
            Token::Enum => Ok(Some(self.parse_enum()?)),
            Token::If => Ok(Some(self.parse_if()?)),
            Token::Try => Ok(Some(self.parse_try()?)),
            Token::Throw => Ok(Some(self.parse_throw()?)),
            Token::Reset => Ok(Some(self.parse_reset()?)),
            Token::Shift => Ok(Some(self.parse_shift()?)),
            Token::Match => Ok(Some(self.parse_match()?)),
            Token::Namespace => Ok(Some(self.parse_namespace()?)),
            Token::Import => Ok(Some(self.parse_import()?)),
            Token::Protocol => Ok(Some(self.parse_protocol()?)),
            Token::Extend => Ok(Some(self.parse_extend()?)),
            Token::Atom((start, end)) => {
                let start_position = self.position;
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                // TODO: Make better
                self.consume();
                self.skip_spaces();
                // self.move_to_next_non_whitespace();
                // I probably don't want to do this forever
                // But for right now I'm assuming that all double colon
                // Identifiers are creating enums
                // I really need to start thinking about namespacing
                // and if I want double colon for that.
                if self.is_open_paren() {
                    Ok(Some(self.parse_call(name, start_position)?))
                }
                // TODO: Hack to try and let struct creation work in ambiguous contexts
                // like if. Need a better way.
                else if self.is_open_curly() && min_precedence == 0 {
                    Ok(Some(self.parse_struct_creation(name, start_position)?))
                } else {
                    Ok(Some(Ast::Identifier(name, self.position)))
                }
            }
            Token::String((start, end)) => {
                let mut value =
                    String::from_utf8(self.source.as_bytes()[start + 1..end - 1].to_vec())
                        .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                // TODO: Test escapes properly
                // Maybe token shouldn't have a range but an actual string value
                // Or I do both
                value = stripslashes(&value);
                let position = self.consume();
                Ok(Some(Ast::String(value, position)))
            }
            Token::InterpolatedString(segments) => {
                let start_position = self.position;
                self.consume(); // consume the interpolated string token

                let mut parts = Vec::new();
                for (seg_start, seg_end, is_expr) in segments {
                    if is_expr {
                        // Parse the expression from the byte range
                        let expr_bytes = &self.source.as_bytes()[seg_start..seg_end];
                        let expr_source = String::from_utf8(expr_bytes.to_vec()).map_err(|_| {
                            ParseError::InvalidUtf8 {
                                position: seg_start,
                            }
                        })?;

                        // Create a new parser for the expression
                        let mut expr_parser =
                            Parser::new("<interpolation>".to_string(), expr_source)?;
                        let expr_ast = expr_parser.parse_expression(0, true, true)?;
                        if let Some(ast) = expr_ast {
                            parts.push(StringInterpolationPart::Expression(Box::new(ast)));
                        }
                    } else {
                        // String literal segment
                        let seg_bytes = &self.source.as_bytes()[seg_start..seg_end];
                        let seg_str = String::from_utf8(seg_bytes.to_vec()).map_err(|_| {
                            ParseError::InvalidUtf8 {
                                position: seg_start,
                            }
                        })?;
                        let processed = stripslashes(&seg_str);
                        if !processed.is_empty() {
                            parts.push(StringInterpolationPart::Literal(processed));
                        }
                    }
                }

                Ok(Some(Ast::StringInterpolation {
                    parts,
                    token_range: TokenRange::new(start_position, self.position),
                }))
            }
            Token::Keyword((start, end)) => {
                let keyword_text = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                let position = self.consume();
                Ok(Some(Ast::Keyword(keyword_text, position)))
            }
            Token::Integer((start, end)) => {
                let value = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                let position = self.consume();
                let parsed_value =
                    value
                        .parse::<i64>()
                        .map_err(|_| ParseError::InvalidNumberLiteral {
                            literal: format!(
                                "{} (must fit in i64: {} to {})",
                                value,
                                i64::MIN,
                                i64::MAX
                            ),
                            position: start,
                        })?;
                // Values use 3-bit tagging (shifted left by 3), so only 61 bits available
                const MAX_61_BIT: i64 = (1i64 << 60) - 1;
                const MIN_61_BIT: i64 = -(1i64 << 60);
                if !(MIN_61_BIT..=MAX_61_BIT).contains(&parsed_value) {
                    return Err(ParseError::InvalidNumberLiteral {
                        literal: format!(
                            "{} (tagged integers use 61 bits: {} to {})",
                            value, MIN_61_BIT, MAX_61_BIT
                        ),
                        position: start,
                    });
                }
                Ok(Some(Ast::IntegerLiteral(parsed_value, position)))
            }
            Token::Float((start, end)) => {
                let value = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                let position = self.consume();
                Ok(Some(Ast::FloatLiteral(value, position)))
            }
            Token::True => {
                let position = self.consume();
                Ok(Some(Ast::True(position)))
            }
            Token::False => {
                let position = self.consume();
                Ok(Some(Ast::False(position)))
            }
            Token::Null => {
                let position = self.consume();
                Ok(Some(Ast::Null(position)))
            }
            Token::Infinity => {
                let position = self.consume();
                Ok(Some(Ast::FloatLiteral("infinity".to_string(), position)))
            }
            Token::NegativeInfinity => {
                let position = self.consume();
                Ok(Some(Ast::FloatLiteral("-infinity".to_string(), position)))
            }
            Token::Let => {
                let start_position = self.position;
                self.consume();
                self.move_to_next_non_whitespace();

                if self.peek_next_non_whitespace() == Token::Mut {
                    self.consume();
                    self.move_to_next_non_whitespace();
                    let pattern = self.parse_binding_pattern()?;
                    self.skip_whitespace();
                    self.expect_equal()?;
                    self.move_to_next_non_whitespace();
                    let value = self.parse_expression(0, true, true)?.ok_or_else(|| {
                        ParseError::UnexpectedEof {
                            expected: "value after '='".to_string(),
                        }
                    })?;
                    let end_position = self.position;
                    return Ok(Some(Ast::LetMut {
                        pattern,
                        value: Box::new(value),
                        token_range: TokenRange::new(start_position, end_position),
                    }));
                }
                let pattern = self.parse_binding_pattern()?;
                self.skip_whitespace();
                self.expect_equal()?;
                self.move_to_next_non_whitespace();
                let value = self.parse_expression(0, true, true)?.ok_or_else(|| {
                    ParseError::UnexpectedEof {
                        expected: "value after '='".to_string(),
                    }
                })?;
                let end_position = self.position;
                Ok(Some(Ast::Let {
                    pattern,
                    value: Box::new(value),
                    token_range: TokenRange::new(start_position, end_position),
                }))
            }
            Token::NewLine | Token::Spaces(_) | Token::Comment(_) => {
                self.consume();
                self.parse_atom(min_precedence)
            }
            Token::OpenParen => {
                self.consume();
                let result = self.parse_expression(0, true, true)?;
                self.expect_close_paren()?;
                Ok(result)
            }
            Token::OpenBracket => {
                let result = self.parse_array()?;
                Ok(Some(result))
            }
            Token::OpenCurly => {
                let result = self.parse_map_literal()?;
                Ok(Some(result))
            }
            Token::HashOpenCurly => {
                let result = self.parse_set_literal()?;
                Ok(Some(result))
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            }),
        }
    }

    fn parse_function(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                self.move_to_next_non_whitespace();
                Some(
                    String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                        .map_err(|_| ParseError::InvalidUtf8 { position: start })?,
                )
            }
            _ => None,
        };

        // Check if this is a multi-arity function (fn name { ... })
        if self.current_token() == Token::OpenCurly {
            return self.parse_multi_arity_function(name, start_position);
        }

        self.expect_open_paren()?;
        let (args, rest_param) = self.parse_args()?;
        self.expect_close_paren()?;
        let body = self.parse_block()?;
        let end_position = self.position;
        Ok(Ast::Function {
            name,
            args,
            rest_param,
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    /// Parse a multi-arity function: fn name { () => expr (x) => expr (x, y) => expr }
    fn parse_multi_arity_function(
        &mut self,
        name: Option<String>,
        start_position: usize,
    ) -> ParseResult<Ast> {
        // We're at the opening {
        self.move_to_next_non_whitespace();

        let mut cases = Vec::new();

        while self.current_token() != Token::CloseCurly {
            let case = self.parse_arity_case()?;
            cases.push(case);

            // Skip whitespace/newlines between cases
            while matches!(
                self.current_token(),
                Token::NewLine | Token::Spaces(_) | Token::Comment(_)
            ) {
                self.consume();
            }
        }

        // Consume the closing }
        self.move_to_next_non_whitespace();

        let end_position = self.position;
        Ok(Ast::MultiArityFunction {
            name,
            cases,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    /// Parse a single arity case: (args...) => body or (args...) => { body }
    fn parse_arity_case(&mut self) -> ParseResult<ArityCase> {
        let start_position = self.position;

        // Parse the argument list
        self.expect_open_paren()?;
        let (args, rest_param) = self.parse_args()?;
        self.expect_close_paren()?;

        // Skip whitespace before '=>'
        self.skip_whitespace();

        // Expect '=>'
        if !matches!(self.current_token(), Token::Arrow) {
            return Err(ParseError::MissingToken {
                expected: "'=>' after arity case arguments".to_string(),
                position: self.position,
            });
        }
        self.move_to_next_non_whitespace();

        // Parse the body - either a single expression or a block
        let body = if self.current_token() == Token::OpenCurly {
            self.parse_block()?
        } else {
            // Single expression body
            let expr = self.parse_expression(0, true, false)?.ok_or_else(|| {
                ParseError::UnexpectedEof {
                    expected: "expression after '=>'".to_string(),
                }
            })?;
            vec![expr]
        };

        let end_position = self.position;
        Ok(ArityCase {
            args,
            rest_param,
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_loop(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let body = self.parse_block()?;
        let end_position = self.position;
        Ok(Ast::Loop {
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_while(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        // Parse condition (no parens required)
        let condition = Box::new(self.parse_expression(1, true, false)?.ok_or_else(|| {
            ParseError::UnexpectedEof {
                expected: "condition after while".to_string(),
            }
        })?);
        let body = self.parse_block()?;
        let end_position = self.position;
        Ok(Ast::While {
            condition,
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_break(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        // Expect function call syntax: break(value)
        if self.current_token() != Token::OpenParen {
            return Err(ParseError::MissingToken {
                expected: "'(' after break".to_string(),
                position: self.position,
            });
        }
        self.move_to_next_non_whitespace();
        let value =
            self.parse_expression(0, true, true)?
                .ok_or_else(|| ParseError::UnexpectedEof {
                    expected: "value in break()".to_string(),
                })?;
        // parse_expression should leave us at the closing paren
        if self.current_token() != Token::CloseParen {
            return Err(ParseError::MissingToken {
                expected: "')' after break value".to_string(),
                position: self.position,
            });
        }
        self.move_to_next_non_whitespace();
        let end_position = self.position;
        Ok(Ast::Break {
            value: Box::new(value),
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_continue(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        // Expect function call syntax: continue()
        if self.current_token() != Token::OpenParen {
            return Err(ParseError::MissingToken {
                expected: "'(' after continue".to_string(),
                position: self.position,
            });
        }
        self.move_to_next_non_whitespace();
        // Should be at closing paren immediately
        if self.current_token() != Token::CloseParen {
            return Err(ParseError::UnexpectedToken {
                expected: "')' - continue takes no arguments".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            });
        }
        self.move_to_next_non_whitespace();
        let end_position = self.position;
        Ok(Ast::Continue {
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_for(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();

        // Parse binding name
        let binding = match self.current_token() {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "identifier after 'for'".to_string(),
                    found: self.get_token_repr(),
                    position: self.position,
                });
            }
        };
        self.move_to_next_non_whitespace();

        // Expect 'in' keyword (context-sensitive - only in for-loops)
        match self.current_token() {
            Token::Atom((start, end)) => {
                let atom = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                if atom != "in" {
                    return Err(ParseError::UnexpectedToken {
                        expected: "'in' after for binding".to_string(),
                        found: atom,
                        position: self.position,
                    });
                }
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "'in' after for binding".to_string(),
                    found: self.get_token_repr(),
                    position: self.position,
                });
            }
        }
        self.move_to_next_non_whitespace();

        // Parse collection expression
        let collection = Box::new(self.parse_expression(1, true, false)?.ok_or_else(|| {
            ParseError::UnexpectedEof {
                expected: "collection expression after 'in'".to_string(),
            }
        })?);

        // Parse body block
        let body = self.parse_block()?;
        let end_position = self.position;

        Ok(Ast::For {
            binding,
            collection,
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_struct(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_atom();
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "struct name".to_string(),
                    found: self.get_token_repr(),
                    position: self.position,
                });
            }
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly()?;
        let fields = self.parse_struct_fields()?;
        self.expect_close_curly()?;
        let end_position = self.position;
        Ok(Ast::Struct {
            name,
            fields,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_protocol(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_atom();
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "protocol name".to_string(),
                    found: self.get_token_repr(),
                    position: self.position,
                });
            }
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly()?;
        let body = self.parse_protocol_body()?;
        self.expect_close_curly()?;
        let end_position = self.position;
        Ok(Ast::Protocol {
            name,
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_protocol_body(&mut self) -> ParseResult<Vec<Ast>> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            self.skip_spaces();
            result.push(self.parse_protocol_member()?);
            self.skip_spaces();
            if !self.is_close_curly() && self.peek_next_non_whitespace() != Token::CloseCurly {
                self.data_delimiter()?;
            }
            self.skip_spaces();
        }
        Ok(result)
    }

    fn parse_protocol_member(&mut self) -> ParseResult<Ast> {
        match self.current_token() {
            Token::Fn => {
                self.consume();
                self.move_to_next_non_whitespace();
                let name = match self.current_token() {
                    Token::Atom((start, end)) => {
                        String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                            .map_err(|_| ParseError::InvalidUtf8 { position: start })?
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "protocol member name".to_string(),
                            found: self.get_token_repr(),
                            position: self.position,
                        });
                    }
                };
                self.move_to_next_non_whitespace();
                self.expect_open_paren()?;
                let (args, rest_param) = self.parse_args()?;
                self.expect_close_paren()?;
                self.skip_spaces();
                if self.is_open_curly() {
                    let body = self.parse_block()?;
                    let end_position = self.position;
                    Ok(Ast::Function {
                        name: Some(name),
                        args,
                        rest_param,
                        body,
                        token_range: TokenRange::new(self.position, end_position),
                    })
                } else {
                    let end_position = self.position;
                    Ok(Ast::FunctionStub {
                        name,
                        args,
                        rest_param,
                        token_range: TokenRange::new(self.position, end_position),
                    })
                }
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "protocol member".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            }),
        }
    }

    fn parse_extend(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_atom();
        let target_type = match self.current_token() {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "type name after 'extend'".to_string(),
                    found: self.get_token_repr(),
                    position: self.position,
                });
            }
        };
        self.move_to_next_non_whitespace();
        self.expect_with()?;
        self.move_to_next_non_whitespace();
        let protocol = match self.current_token() {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "protocol name after 'with'".to_string(),
                    found: self.get_token_repr(),
                    position: self.position,
                });
            }
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly()?;
        let body = self.parse_extend_body()?;
        self.expect_close_curly()?;
        let end_position = self.position;
        Ok(Ast::Extend {
            target_type,
            protocol,
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_extend_body(&mut self) -> ParseResult<Vec<Ast>> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            self.skip_spaces();
            result.push(self.parse_extend_member()?);
            self.skip_spaces();
            if !self.is_close_curly() {
                self.data_delimiter()?;
            }
            self.skip_spaces();
        }
        Ok(result)
    }

    fn parse_extend_member(&mut self) -> ParseResult<Ast> {
        self.skip_whitespace();
        self.parse_function()
    }

    fn parse_enum(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_atom();
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "enum name".to_string(),
                    found: self.get_token_repr(),
                    position: self.position,
                });
            }
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly()?;
        let variants = self.parse_enum_variants()?;
        self.expect_close_curly()?;
        let end_position = self.position;
        Ok(Ast::Enum {
            name,
            variants,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn consume(&mut self) -> usize {
        self.position += 1;
        self.position - 1
    }

    fn move_to_next_atom(&mut self) {
        self.consume();
        while !self.at_end() && !self.is_atom() {
            self.consume();
        }
    }

    // TODO: These two are similar and one of them should be removed
    // but also, why does this one only use is_space and the other
    // doesn't care about comments?
    fn move_to_next_non_whitespace(&mut self) {
        self.consume();
        while !self.at_end() && (self.is_space() || self.is_comment()) {
            self.consume();
        }
    }

    fn peek_next_non_whitespace(&mut self) -> Token {
        let starting_position = self.position;
        while !self.at_end() && (self.is_space() || self.is_comment()) {
            self.consume();
        }
        let result = self.current_token();
        self.position = starting_position;
        result
    }

    fn skip_whitespace(&mut self) {
        while !self.at_end() && self.is_whitespace() {
            self.consume();
        }
    }

    fn expect_open_paren(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_open_paren() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "open paren '('".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn expect_with(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_with() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "'with' keyword".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn expect_close_bracket(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_close_bracket() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "close bracket ']'".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn expect_comma(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_comma() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "comma ','".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn expect_as(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_as() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "'as' keyword".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn is_open_paren(&self) -> bool {
        self.current_token() == Token::OpenParen
    }

    fn is_comma(&self) -> bool {
        self.current_token() == Token::Comma
    }

    fn parse_args(&mut self) -> ParseResult<(Vec<Pattern>, Option<String>)> {
        let mut result = Vec::new();
        let mut rest_param = None;
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_paren() {
            let (pattern, is_rest) = self.parse_arg()?;
            if is_rest {
                if rest_param.is_some() {
                    return Err(ParseError::InvalidDeclaration {
                        message: "Only one rest parameter allowed".to_string(),
                        position: self.position,
                    });
                }
                // Rest params must be simple identifiers
                if let Some(name) = pattern.as_identifier() {
                    rest_param = Some(name.to_string());
                } else {
                    return Err(ParseError::InvalidDeclaration {
                        message: "Rest parameter must be a simple identifier".to_string(),
                        position: self.position,
                    });
                }
            } else {
                if rest_param.is_some() {
                    return Err(ParseError::InvalidDeclaration {
                        message: "Rest parameter must be last".to_string(),
                        position: self.position,
                    });
                }
                result.push(pattern);
            }
            self.skip_whitespace();
        }
        Ok((result, rest_param))
    }

    fn parse_struct_fields(&mut self) -> ParseResult<Vec<Ast>> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            self.skip_spaces();
            result.push(self.parse_struct_field()?);
            self.skip_spaces();
            if !self.is_close_curly() {
                self.data_delimiter()?;
            }
            self.skip_spaces();
        }
        Ok(result)
    }

    fn parse_struct_field(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;

        // Check for 'mut' keyword
        let mutable = if self.current_token() == Token::Mut {
            self.consume();
            self.skip_spaces();
            true
        } else {
            false
        };

        match self.current_token() {
            Token::Atom((start, end)) => {
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                let end_position = self.consume();
                Ok(Ast::StructField {
                    name,
                    mutable,
                    token_range: TokenRange::new(start_position, end_position),
                })
            }
            // Allow 'shift' and 'reset' as field names (they're only keywords in expression context)
            Token::Shift => {
                let end_position = self.consume();
                Ok(Ast::StructField {
                    name: "shift".to_string(),
                    mutable,
                    token_range: TokenRange::new(start_position, end_position),
                })
            }
            Token::Reset => {
                let end_position = self.consume();
                Ok(Ast::StructField {
                    name: "reset".to_string(),
                    mutable,
                    token_range: TokenRange::new(start_position, end_position),
                })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "field name".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            }),
        }
    }

    fn parse_enum_variants(&mut self) -> ParseResult<Vec<Ast>> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            result.push(self.parse_enum_variant()?);
            self.skip_whitespace();
        }
        Ok(result)
    }

    fn parse_enum_variant(&mut self) -> ParseResult<Ast> {
        // We need to parse enum variants that are just a name
        // and enum variants that are struct like

        match self.current_token() {
            Token::Atom((start, end)) => {
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                let position = self.consume();
                self.skip_spaces();
                let result = if self.is_open_curly() {
                    let start_position = self.consume();
                    let fields = self.parse_struct_fields()?;
                    self.expect_close_curly()?;
                    let end_position = self.position;
                    Ast::EnumVariant {
                        name,
                        fields,
                        token_range: TokenRange::new(start_position, end_position),
                    }
                } else {
                    Ast::EnumStaticVariant {
                        name,
                        token_range: TokenRange::new(position, position),
                    }
                };
                self.data_delimiter()?;
                Ok(result)
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "variant name".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            }),
        }
    }

    fn is_space(&self) -> bool {
        match self.current_token() {
            Token::Spaces(_) => true,
            _ => false,
        }
    }

    fn is_comment(&self) -> bool {
        match self.current_token() {
            Token::Comment(_) => true,
            _ => false,
        }
    }

    // TODO: Deal with tabs (People shouldn't use them though lol)
    fn skip_spaces(&mut self) {
        while !self.at_end() && (self.is_space() || self.is_comment()) {
            self.consume();
        }
    }

    fn data_delimiter(&mut self) -> ParseResult<()> {
        self.skip_spaces();
        if self.is_comma() || self.is_newline() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "comma or newline".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn parse_arg(&mut self) -> ParseResult<(Pattern, bool)> {
        // Check for rest parameter syntax: ...name
        let is_rest = if self.current_token() == Token::DotDotDot {
            self.consume();
            self.skip_spaces();
            true
        } else {
            false
        };

        let pattern = self.parse_binding_pattern()?;
        self.skip_whitespace();
        if !self.is_close_paren() {
            self.expect_comma()?;
        }
        Ok((pattern, is_rest))
    }

    /// Parses a binding pattern for let bindings and function arguments.
    /// Supports:
    /// - Simple identifiers: `x`
    /// - Wildcard: `_` (ignores the value)
    /// - Struct destructuring: `Point { x, y }` or `Point { x: a, y: b }`
    /// - Array destructuring: `[first, second, ...rest]`
    /// - Map destructuring: `{ name, age }` or `{ "key": value }`
    fn parse_binding_pattern(&mut self) -> ParseResult<Pattern> {
        self.skip_whitespace();
        let start = self.position;

        match self.current_token() {
            // Wildcard pattern - ignores the value
            Token::Underscore => {
                self.consume();
                Ok(Pattern::Wildcard {
                    token_range: TokenRange::new(start, self.position),
                })
            }
            // Simple identifier
            Token::Atom((atom_start, atom_end)) => {
                let name = String::from_utf8(self.source.as_bytes()[atom_start..atom_end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 {
                        position: atom_start,
                    })?;
                self.consume();
                self.skip_whitespace();

                // Check if it's struct destructuring: `Name { ... }`
                if matches!(self.current_token(), Token::OpenCurly) {
                    let fields = self.parse_field_patterns()?;
                    Ok(Pattern::Struct {
                        name,
                        fields,
                        token_range: TokenRange::new(start, self.position),
                    })
                } else {
                    // Simple identifier pattern
                    Ok(Pattern::Identifier {
                        name,
                        token_range: TokenRange::new(start, self.position),
                    })
                }
            }
            // Array destructuring: [a, b, ...rest]
            Token::OpenBracket => {
                self.consume();
                self.skip_whitespace();

                let mut elements = vec![];
                let mut rest = None;

                while !matches!(self.current_token(), Token::CloseBracket) {
                    // Check for rest pattern
                    if matches!(self.current_token(), Token::DotDotDot) {
                        self.consume();
                        self.skip_whitespace();
                        let rest_pattern = self.parse_binding_pattern()?;
                        rest = Some(Box::new(rest_pattern));
                        self.skip_whitespace();
                        // Rest must be last
                        if !matches!(self.current_token(), Token::CloseBracket) {
                            return Err(ParseError::InvalidPattern {
                                message: "Rest pattern must be last in array destructuring"
                                    .to_string(),
                                position: self.position,
                            });
                        }
                        break;
                    }

                    let element_pattern = self.parse_binding_pattern()?;
                    elements.push(element_pattern);
                    self.skip_whitespace();

                    if matches!(self.current_token(), Token::Comma) {
                        self.consume();
                        self.skip_whitespace();
                    } else if !matches!(self.current_token(), Token::CloseBracket) {
                        return Err(ParseError::UnexpectedToken {
                            expected: "',' or ']'".to_string(),
                            found: self.get_token_repr(),
                            position: self.position,
                        });
                    }
                }

                // Consume ']'
                if !matches!(self.current_token(), Token::CloseBracket) {
                    return Err(ParseError::MissingToken {
                        expected: "']' to close array pattern".to_string(),
                        position: self.position,
                    });
                }
                self.consume();

                Ok(Pattern::Array {
                    elements,
                    rest,
                    token_range: TokenRange::new(start, self.position),
                })
            }
            // Map destructuring: { name, age } or { "key": value }
            Token::OpenCurly => {
                self.consume();
                self.skip_whitespace();

                let mut fields = vec![];

                while !matches!(self.current_token(), Token::CloseCurly) {
                    let field_start = self.position;

                    match self.current_token() {
                        // Keyword key syntax: { name } or { name: binding }
                        Token::Atom((atom_start, atom_end)) => {
                            let key_name = String::from_utf8(
                                self.source.as_bytes()[atom_start..atom_end].to_vec(),
                            )
                            .map_err(|_| ParseError::InvalidUtf8 {
                                position: atom_start,
                            })?;
                            self.consume();
                            self.skip_whitespace();

                            // Check for rename syntax: name: binding or name: _
                            let binding_name = if matches!(self.current_token(), Token::Colon) {
                                self.consume();
                                self.skip_whitespace();

                                match self.current_token() {
                                    Token::Atom((bind_start, bind_end)) => {
                                        let binding = String::from_utf8(
                                            self.source.as_bytes()[bind_start..bind_end].to_vec(),
                                        )
                                        .map_err(|_| ParseError::InvalidUtf8 {
                                            position: bind_start,
                                        })?;
                                        self.consume();
                                        self.skip_whitespace();
                                        binding
                                    }
                                    Token::Underscore => {
                                        self.consume();
                                        self.skip_whitespace();
                                        "_".to_string()
                                    }
                                    _ => {
                                        return Err(ParseError::InvalidPattern {
                                            message: "Expected binding name or '_' after ':'"
                                                .to_string(),
                                            position: self.position,
                                        });
                                    }
                                }
                            } else {
                                key_name.clone()
                            };

                            fields.push(MapFieldPattern {
                                key: MapKey::Keyword(key_name),
                                binding_name,
                                token_range: TokenRange::new(field_start, self.position),
                            });
                        }
                        // String key syntax: { "some-key": binding }
                        Token::String((str_start, str_end)) => {
                            // Strip quotes from string (start+1 to end-1)
                            let key_string = String::from_utf8(
                                self.source.as_bytes()[str_start + 1..str_end - 1].to_vec(),
                            )
                            .map_err(|_| ParseError::InvalidUtf8 {
                                position: str_start,
                            })?;
                            self.consume();
                            self.skip_whitespace();

                            // String keys require the : binding syntax
                            if !matches!(self.current_token(), Token::Colon) {
                                return Err(ParseError::InvalidPattern {
                                    message: "String keys in map destructuring require ': binding' syntax".to_string(),
                                    position: self.position,
                                });
                            }
                            self.consume();
                            self.skip_whitespace();

                            let binding_name = match self.current_token() {
                                Token::Atom((bind_start, bind_end)) => {
                                    let binding = String::from_utf8(
                                        self.source.as_bytes()[bind_start..bind_end].to_vec(),
                                    )
                                    .map_err(|_| ParseError::InvalidUtf8 {
                                        position: bind_start,
                                    })?;
                                    self.consume();
                                    self.skip_whitespace();
                                    binding
                                }
                                Token::Underscore => {
                                    self.consume();
                                    self.skip_whitespace();
                                    "_".to_string()
                                }
                                _ => {
                                    return Err(ParseError::InvalidPattern {
                                        message: "Expected binding name or '_' after ':'"
                                            .to_string(),
                                        position: self.position,
                                    });
                                }
                            };

                            fields.push(MapFieldPattern {
                                key: MapKey::String(key_string),
                                binding_name,
                                token_range: TokenRange::new(field_start, self.position),
                            });
                        }
                        _ => {
                            return Err(ParseError::InvalidPattern {
                                message: "Expected field name or string key in map pattern"
                                    .to_string(),
                                position: self.position,
                            });
                        }
                    }

                    // Allow comma or newline between fields
                    if matches!(self.current_token(), Token::Comma) {
                        self.consume();
                        self.skip_whitespace();
                    } else {
                        self.skip_whitespace();
                    }
                }

                // Consume '}'
                if !matches!(self.current_token(), Token::CloseCurly) {
                    return Err(ParseError::MissingToken {
                        expected: "'}' to close map pattern".to_string(),
                        position: self.position,
                    });
                }
                self.consume();

                Ok(Pattern::Map {
                    fields,
                    token_range: TokenRange::new(start, self.position),
                })
            }
            _ => Err(ParseError::InvalidPattern {
                message: format!(
                    "Expected identifier or destructuring pattern, found {:?}",
                    self.current_token()
                ),
                position: self.position,
            }),
        }
    }

    fn expect_close_paren(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_close_paren() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "close paren ')'".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn is_close_paren(&self) -> bool {
        self.current_token() == Token::CloseParen
    }

    fn parse_block(&mut self) -> ParseResult<Vec<Ast>> {
        self.expect_open_curly()?;
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            match self.parse_expression(0, true, true)? {
                Some(elem) => result.push(elem),
                None => break,
            }
            self.skip_whitespace();
        }
        self.expect_close_curly()?;
        Ok(result)
    }

    fn expect_open_curly(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_open_curly() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "open curly '{'".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn is_open_curly(&self) -> bool {
        self.current_token() == Token::OpenCurly
    }

    fn is_close_curly(&self) -> bool {
        self.current_token() == Token::CloseCurly
    }

    fn expect_close_curly(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_close_curly() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "close curly '}'".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn is_atom(&self) -> bool {
        match self.current_token() {
            Token::Atom(_) => true,
            _ => false,
        }
    }

    fn is_as(&self) -> bool {
        match self.current_token() {
            Token::As => true,
            _ => false,
        }
    }

    fn is_with(&self) -> bool {
        match self.current_token() {
            Token::With => true,
            _ => false,
        }
    }

    fn current_token(&self) -> Token {
        if self.position >= self.tokens.len() {
            Token::Never
        } else {
            self.tokens[self.position].clone()
        }
    }

    fn parse_namespace(&mut self) -> ParseResult<Ast> {
        // TODO: Reconsider this design
        // namespaces can be names with dots
        // so beagle.core is a valid namespace

        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let mut name = String::new();
        while !self.at_end() && (self.is_atom() || self.is_dot()) {
            match self.current_token() {
                Token::Dot => {
                    name.push('.');
                }
                Token::Atom((start, end)) => {
                    name.push_str(&self.source[start..end]);
                }
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "namespace identifier".to_string(),
                        found: self.get_token_repr(),
                        position: self.position,
                    });
                }
            }
            self.consume();
        }
        if name.is_empty() {
            return Err(ParseError::UnexpectedToken {
                expected: "namespace name".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            });
        }
        self.consume();
        let end_position = self.position;
        Ok(Ast::Namespace {
            name,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_import(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let library_name = self.expect_string()?;
        self.consume();
        self.expect_as()?;
        self.move_to_next_non_whitespace();
        let name_position = self.position;
        let alias = Box::new(Ast::Identifier(self.expect_atom()?, name_position));
        self.consume();
        let end_position = self.position;
        Ok(Ast::Import {
            library_name,
            alias,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_call(&mut self, name: String, start_position: usize) -> ParseResult<Ast> {
        self.expect_open_paren()?;
        let mut args = Vec::new();
        while !self.at_end() && !self.is_close_paren() {
            match self.parse_expression(0, true, true)? {
                Some(arg) => {
                    args.push(arg);
                    self.skip_whitespace();
                    if !self.is_close_paren() {
                        self.expect_comma()?;
                    }
                }
                None => break,
            }
        }
        self.expect_close_paren()?;
        let end_position = self.position;
        Ok(Ast::Call {
            name,
            args,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_struct_creation(&mut self, name: String, start_position: usize) -> ParseResult<Ast> {
        self.expect_open_curly()?;
        let fields = self.parse_struct_fields_creations()?;

        self.expect_close_curly()?;
        let end_position = self.position;
        Ok(Ast::StructCreation {
            name,
            fields,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_struct_fields_creations(&mut self) -> ParseResult<Vec<(String, Ast)>> {
        let mut fields = Vec::new();
        while !self.at_end() && !self.is_close_curly() {
            if let Some(field) = self.parse_struct_field_creation()? {
                fields.push(field);
            } else {
                break;
            }
        }
        Ok(fields)
    }

    fn parse_struct_field_creation(&mut self) -> ParseResult<Option<(String, Ast)>> {
        self.skip_whitespace();
        match self.current_token() {
            Token::Atom((start, end)) => {
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                self.consume();
                self.skip_spaces();
                self.expect_colon()?;
                self.skip_spaces();
                let value = self.parse_expression(0, false, true)?.ok_or_else(|| {
                    ParseError::InvalidExpression {
                        message: "Expected value for struct field".to_string(),
                        position: self.position,
                    }
                })?;
                if !self.is_close_curly() {
                    self.data_delimiter()?;
                }
                Ok(Some((name, value)))
            }
            // Allow 'shift' and 'reset' as struct field names
            Token::Shift => {
                self.consume();
                self.skip_spaces();
                self.expect_colon()?;
                self.skip_spaces();
                let value = self.parse_expression(0, false, true)?.ok_or_else(|| {
                    ParseError::InvalidExpression {
                        message: "Expected value for struct field".to_string(),
                        position: self.position,
                    }
                })?;
                if !self.is_close_curly() {
                    self.data_delimiter()?;
                }
                Ok(Some(("shift".to_string(), value)))
            }
            Token::Reset => {
                self.consume();
                self.skip_spaces();
                self.expect_colon()?;
                self.skip_spaces();
                let value = self.parse_expression(0, false, true)?.ok_or_else(|| {
                    ParseError::InvalidExpression {
                        message: "Expected value for struct field".to_string(),
                        position: self.position,
                    }
                })?;
                if !self.is_close_curly() {
                    self.data_delimiter()?;
                }
                Ok(Some(("reset".to_string(), value)))
            }
            _ => Ok(None),
        }
    }

    fn get_token_repr(&self) -> String {
        match self.current_token() {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            Token::String((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            Token::Integer((start, end)) => {
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            _ => format!("{:?}", self.current_token()),
        }
    }

    fn is_whitespace(&self) -> bool {
        match self.current_token() {
            Token::Spaces(_) | Token::NewLine | Token::Comment(_) | Token::SemiColon => true,
            _ => false,
        }
    }

    fn parse_if(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let condition = Box::new(self.parse_expression(1, true, false)?.ok_or_else(|| {
            ParseError::UnexpectedEof {
                expected: "condition after 'if'".to_string(),
            }
        })?);
        let then = self.parse_block()?;
        self.move_to_next_non_whitespace();
        if self.is_else() {
            self.consume();
            self.skip_whitespace();
            if self.is_if() {
                self.consume();
                let else_ = vec![self.parse_if()?];
                let end_position = self.position;
                return Ok(Ast::If {
                    condition,
                    then,
                    else_,
                    token_range: TokenRange::new(start_position, end_position),
                });
            }
            self.skip_whitespace();
            let else_ = self.parse_block()?;
            let end_position = self.position;
            Ok(Ast::If {
                condition,
                then,
                else_,
                token_range: TokenRange::new(start_position, end_position),
            })
        } else {
            let end_position = self.position;
            Ok(Ast::If {
                condition,
                then,
                else_: Vec::new(),
                token_range: TokenRange::new(start_position, end_position),
            })
        }
    }

    fn is_else(&self) -> bool {
        match self.current_token() {
            Token::Else => true,
            _ => false,
        }
    }

    fn is_if(&self) -> bool {
        match self.current_token() {
            Token::If => true,
            _ => false,
        }
    }

    fn parse_try(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();

        // Parse try block
        let body = self.parse_block()?;

        // Expect 'catch'
        self.move_to_next_non_whitespace();
        if !matches!(self.current_token(), Token::Catch) {
            return Err(ParseError::MissingToken {
                expected: "'catch' after try block".to_string(),
                position: self.position,
            });
        }
        self.consume();
        self.move_to_next_non_whitespace();

        // Parse catch parameter: catch(e)
        if !matches!(self.current_token(), Token::OpenParen) {
            return Err(ParseError::MissingToken {
                expected: "'(' after 'catch'".to_string(),
                position: self.position,
            });
        }
        self.consume();
        self.skip_whitespace();

        // Get the exception binding identifier
        let exception_binding = if let Token::Atom((start, end)) = self.current_token() {
            let binding = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
            self.consume();
            binding
        } else {
            return Err(ParseError::UnexpectedToken {
                expected: "identifier for exception binding".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            });
        };

        self.skip_whitespace();
        if !matches!(self.current_token(), Token::CloseParen) {
            return Err(ParseError::MissingToken {
                expected: "')' after exception binding".to_string(),
                position: self.position,
            });
        }
        self.consume();

        // Parse catch block
        let catch_body = self.parse_block()?;

        let end_position = self.position;
        Ok(Ast::Try {
            body,
            exception_binding,
            catch_body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_throw(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();

        // throw is a function call: throw(value)
        if !matches!(self.current_token(), Token::OpenParen) {
            return Err(ParseError::MissingToken {
                expected: "'(' after 'throw'".to_string(),
                position: self.position,
            });
        }
        self.consume();
        self.skip_whitespace();

        let value = Box::new(self.parse_expression(1, true, false)?.ok_or_else(|| {
            ParseError::UnexpectedEof {
                expected: "value in throw()".to_string(),
            }
        })?);

        self.skip_whitespace();
        if !matches!(self.current_token(), Token::CloseParen) {
            return Err(ParseError::MissingToken {
                expected: "')' after throw value".to_string(),
                position: self.position,
            });
        }
        self.consume();

        let end_position = self.position;
        Ok(Ast::Throw {
            value,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_reset(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();

        // Parse reset block: reset { body }
        let body = self.parse_block()?;

        let end_position = self.position;
        Ok(Ast::Reset {
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_shift(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();

        // Parse shift(|k| { body })
        // Expect '('
        if !matches!(self.current_token(), Token::OpenParen) {
            return Err(ParseError::MissingToken {
                expected: "'(' after 'shift'".to_string(),
                position: self.position,
            });
        }
        self.consume();
        self.skip_whitespace();

        // Expect '|'
        if !matches!(self.current_token(), Token::BitWiseOr) {
            return Err(ParseError::MissingToken {
                expected: "'|' for continuation parameter".to_string(),
                position: self.position,
            });
        }
        self.consume();
        self.skip_whitespace();

        // Get the continuation parameter identifier
        let continuation_param = if let Token::Atom((start, end)) = self.current_token() {
            let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
            self.consume();
            name
        } else {
            return Err(ParseError::UnexpectedToken {
                expected: "identifier for continuation parameter".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            });
        };

        self.skip_whitespace();

        // Expect closing '|'
        if !matches!(self.current_token(), Token::BitWiseOr) {
            return Err(ParseError::MissingToken {
                expected: "'|' after continuation parameter".to_string(),
                position: self.position,
            });
        }
        self.consume();
        self.skip_whitespace();

        // Parse the body block
        let body = self.parse_block()?;

        self.skip_whitespace();

        // Expect closing ')'
        if !matches!(self.current_token(), Token::CloseParen) {
            return Err(ParseError::MissingToken {
                expected: "')' after shift body".to_string(),
                position: self.position,
            });
        }
        self.consume();

        let end_position = self.position;
        Ok(Ast::Shift {
            continuation_param,
            body,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_match(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.move_to_next_non_whitespace();

        // Parse value to match on - same as if condition
        let value = Box::new(self.parse_expression(1, true, false)?.ok_or_else(|| {
            ParseError::UnexpectedEof {
                expected: "value after 'match'".to_string(),
            }
        })?);

        // Expect '{'
        self.skip_whitespace();
        if !matches!(self.current_token(), Token::OpenCurly) {
            return Err(ParseError::MissingToken {
                expected: "'{{' after match value".to_string(),
                position: self.position,
            });
        }
        self.consume();
        self.move_to_next_non_whitespace();

        // Parse match arms
        let mut arms = vec![];
        while !matches!(self.current_token(), Token::CloseCurly) {
            let arm = self.parse_match_arm()?;
            arms.push(arm);
            self.skip_whitespace();

            // Allow optional comma after each arm
            if matches!(self.current_token(), Token::Comma) {
                self.consume();
                self.skip_whitespace();
            }
        }

        // Consume '}'
        self.consume();
        let end_position = self.position;

        Ok(Ast::Match {
            value,
            arms,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_match_arm(&mut self) -> ParseResult<MatchArm> {
        let arm_start = self.position;

        // Parse pattern
        let pattern = self.parse_pattern()?;

        // Expect '=>'
        self.skip_whitespace();
        if !matches!(self.current_token(), Token::Arrow) {
            return Err(ParseError::UnexpectedToken {
                expected: "'=>' after match pattern".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            });
        }
        self.consume();
        self.skip_whitespace();

        // Parse body - either a block or single expression
        let body = if matches!(self.current_token(), Token::OpenCurly) {
            self.parse_block()?
        } else {
            // Single expression
            let expr = self.parse_expression(1, true, false)?.ok_or_else(|| {
                ParseError::InvalidExpression {
                    message: "Expected expression in match arm body".to_string(),
                    position: self.position,
                }
            })?;
            vec![expr]
        };

        let arm_end = self.position;
        Ok(MatchArm {
            pattern,
            guard: None, // TODO: implement guards later
            body,
            token_range: TokenRange::new(arm_start, arm_end),
        })
    }

    /// Parses a pattern for match expressions.
    /// Supports all binding patterns plus:
    /// - Enum variants: `Enum.Variant` or `Enum.Variant { field }`
    /// - Literals: integers, strings, booleans, null
    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        self.skip_whitespace();
        let start = self.position;

        Ok(match self.current_token() {
            // Wildcard pattern
            Token::Underscore => {
                self.consume();
                Pattern::Wildcard {
                    token_range: TokenRange::new(start, self.position),
                }
            }
            // Identifier, Enum variant, or Struct destructuring
            Token::Atom((atom_start, atom_end)) => {
                let first_name =
                    String::from_utf8(self.source.as_bytes()[atom_start..atom_end].to_vec())
                        .map_err(|_| ParseError::InvalidUtf8 {
                            position: atom_start,
                        })?;
                self.consume();
                self.skip_whitespace();

                // Check if it's an enum variant pattern (Enum.Variant)
                if matches!(self.current_token(), Token::Dot) {
                    self.consume();
                    self.skip_whitespace();

                    // Get variant name
                    if let Token::Atom((var_start, var_end)) = self.current_token() {
                        let variant_name =
                            String::from_utf8(self.source.as_bytes()[var_start..var_end].to_vec())
                                .map_err(|_| ParseError::InvalidUtf8 {
                                    position: var_start,
                                })?;
                        self.consume();
                        self.skip_whitespace();

                        // Check for field patterns { }
                        let fields = if matches!(self.current_token(), Token::OpenCurly) {
                            self.parse_field_patterns()?
                        } else {
                            vec![]
                        };

                        Pattern::EnumVariant {
                            enum_name: first_name,
                            variant_name,
                            fields,
                            token_range: TokenRange::new(start, self.position),
                        }
                    } else {
                        return Err(ParseError::InvalidPattern {
                            message: "Expected variant name after '.'".to_string(),
                            position: self.position,
                        });
                    }
                } else if matches!(self.current_token(), Token::OpenCurly) {
                    // Struct destructuring: `Name { ... }`
                    let fields = self.parse_field_patterns()?;
                    Pattern::Struct {
                        name: first_name,
                        fields,
                        token_range: TokenRange::new(start, self.position),
                    }
                } else {
                    // Simple identifier pattern - binds value to name
                    Pattern::Identifier {
                        name: first_name,
                        token_range: TokenRange::new(start, self.position),
                    }
                }
            }
            // Array destructuring: [a, b, ...rest]
            Token::OpenBracket => {
                self.consume();
                self.skip_whitespace();

                let mut elements = vec![];
                let mut rest = None;

                while !matches!(self.current_token(), Token::CloseBracket) {
                    // Check for rest pattern
                    if matches!(self.current_token(), Token::DotDotDot) {
                        self.consume();
                        self.skip_whitespace();
                        let rest_pattern = self.parse_pattern()?;
                        rest = Some(Box::new(rest_pattern));
                        self.skip_whitespace();
                        // Rest must be last
                        if !matches!(self.current_token(), Token::CloseBracket) {
                            return Err(ParseError::InvalidPattern {
                                message: "Rest pattern must be last in array destructuring"
                                    .to_string(),
                                position: self.position,
                            });
                        }
                        break;
                    }

                    let element_pattern = self.parse_pattern()?;
                    elements.push(element_pattern);
                    self.skip_whitespace();

                    if matches!(self.current_token(), Token::Comma) {
                        self.consume();
                        self.skip_whitespace();
                    } else if !matches!(self.current_token(), Token::CloseBracket) {
                        return Err(ParseError::UnexpectedToken {
                            expected: "',' or ']'".to_string(),
                            found: self.get_token_repr(),
                            position: self.position,
                        });
                    }
                }

                // Consume ']'
                if !matches!(self.current_token(), Token::CloseBracket) {
                    return Err(ParseError::MissingToken {
                        expected: "']' to close array pattern".to_string(),
                        position: self.position,
                    });
                }
                self.consume();

                Pattern::Array {
                    elements,
                    rest,
                    token_range: TokenRange::new(start, self.position),
                }
            }
            // Map destructuring: { name, age } or { "key": value }
            Token::OpenCurly => {
                self.consume();
                self.skip_whitespace();

                let mut fields = vec![];

                while !matches!(self.current_token(), Token::CloseCurly) {
                    let field_start = self.position;

                    match self.current_token() {
                        // Keyword key syntax: { name } or { name: binding }
                        Token::Atom((atom_start, atom_end)) => {
                            let key_name = String::from_utf8(
                                self.source.as_bytes()[atom_start..atom_end].to_vec(),
                            )
                            .map_err(|_| ParseError::InvalidUtf8 {
                                position: atom_start,
                            })?;
                            self.consume();
                            self.skip_whitespace();

                            // Check for rename syntax: name: binding or name: _
                            let binding_name = if matches!(self.current_token(), Token::Colon) {
                                self.consume();
                                self.skip_whitespace();

                                match self.current_token() {
                                    Token::Atom((bind_start, bind_end)) => {
                                        let binding = String::from_utf8(
                                            self.source.as_bytes()[bind_start..bind_end].to_vec(),
                                        )
                                        .map_err(|_| ParseError::InvalidUtf8 {
                                            position: bind_start,
                                        })?;
                                        self.consume();
                                        self.skip_whitespace();
                                        binding
                                    }
                                    Token::Underscore => {
                                        self.consume();
                                        self.skip_whitespace();
                                        "_".to_string()
                                    }
                                    _ => {
                                        return Err(ParseError::InvalidPattern {
                                            message: "Expected binding name or '_' after ':'"
                                                .to_string(),
                                            position: self.position,
                                        });
                                    }
                                }
                            } else {
                                key_name.clone()
                            };

                            fields.push(MapFieldPattern {
                                key: MapKey::Keyword(key_name),
                                binding_name,
                                token_range: TokenRange::new(field_start, self.position),
                            });
                        }
                        // String key syntax: { "some-key": binding }
                        Token::String((str_start, str_end)) => {
                            // Strip quotes from string (start+1 to end-1)
                            let key_string = String::from_utf8(
                                self.source.as_bytes()[str_start + 1..str_end - 1].to_vec(),
                            )
                            .map_err(|_| ParseError::InvalidUtf8 {
                                position: str_start,
                            })?;
                            self.consume();
                            self.skip_whitespace();

                            // String keys require the : binding syntax
                            if !matches!(self.current_token(), Token::Colon) {
                                return Err(ParseError::InvalidPattern {
                                    message:
                                        "String keys in map destructuring require ': binding' syntax"
                                            .to_string(),
                                    position: self.position,
                                });
                            }
                            self.consume();
                            self.skip_whitespace();

                            let binding_name = match self.current_token() {
                                Token::Atom((bind_start, bind_end)) => {
                                    let binding = String::from_utf8(
                                        self.source.as_bytes()[bind_start..bind_end].to_vec(),
                                    )
                                    .map_err(|_| ParseError::InvalidUtf8 {
                                        position: bind_start,
                                    })?;
                                    self.consume();
                                    self.skip_whitespace();
                                    binding
                                }
                                Token::Underscore => {
                                    self.consume();
                                    self.skip_whitespace();
                                    "_".to_string()
                                }
                                _ => {
                                    return Err(ParseError::InvalidPattern {
                                        message: "Expected binding name or '_' after ':'"
                                            .to_string(),
                                        position: self.position,
                                    });
                                }
                            };

                            fields.push(MapFieldPattern {
                                key: MapKey::String(key_string),
                                binding_name,
                                token_range: TokenRange::new(field_start, self.position),
                            });
                        }
                        _ => {
                            return Err(ParseError::InvalidPattern {
                                message: "Expected field name or string key in map pattern"
                                    .to_string(),
                                position: self.position,
                            });
                        }
                    }

                    // Allow comma or newline between fields
                    if matches!(self.current_token(), Token::Comma) {
                        self.consume();
                        self.skip_whitespace();
                    } else {
                        self.skip_whitespace();
                    }
                }

                // Consume '}'
                if !matches!(self.current_token(), Token::CloseCurly) {
                    return Err(ParseError::MissingToken {
                        expected: "'}' to close map pattern".to_string(),
                        position: self.position,
                    });
                }
                self.consume();

                Pattern::Map {
                    fields,
                    token_range: TokenRange::new(start, self.position),
                }
            }
            // Literal patterns (only valid in match, not in let bindings)
            Token::Integer((int_start, int_end)) => {
                let int_str =
                    String::from_utf8(self.source.as_bytes()[int_start..int_end].to_vec())
                        .map_err(|_| ParseError::InvalidUtf8 {
                            position: int_start,
                        })?;
                let value =
                    int_str
                        .parse::<i64>()
                        .map_err(|_| ParseError::InvalidNumberLiteral {
                            literal: int_str,
                            position: int_start,
                        })?;
                self.consume();
                Pattern::Literal {
                    value: Box::new(Ast::IntegerLiteral(value, start)),
                    token_range: TokenRange::new(start, self.position),
                }
            }
            Token::String((str_start, str_end)) => {
                // Strip quotes from string
                let string_value =
                    String::from_utf8(self.source.as_bytes()[str_start + 1..str_end - 1].to_vec())
                        .map_err(|_| ParseError::InvalidUtf8 {
                            position: str_start,
                        })?;
                self.consume();
                Pattern::Literal {
                    value: Box::new(Ast::String(string_value, start)),
                    token_range: TokenRange::new(start, self.position),
                }
            }
            Token::True => {
                self.consume();
                Pattern::Literal {
                    value: Box::new(Ast::True(start)),
                    token_range: TokenRange::new(start, self.position),
                }
            }
            Token::False => {
                self.consume();
                Pattern::Literal {
                    value: Box::new(Ast::False(start)),
                    token_range: TokenRange::new(start, self.position),
                }
            }
            Token::Null => {
                self.consume();
                Pattern::Literal {
                    value: Box::new(Ast::Null(start)),
                    token_range: TokenRange::new(start, self.position),
                }
            }
            _ => {
                return Err(ParseError::InvalidPattern {
                    message: format!("Unexpected token in pattern: {:?}", self.current_token()),
                    position: self.position,
                });
            }
        })
    }

    fn parse_field_patterns(&mut self) -> ParseResult<Vec<FieldPattern>> {
        // Consume '{'
        self.consume();
        self.skip_whitespace();

        let mut fields = vec![];
        while !matches!(self.current_token(), Token::CloseCurly) {
            let field_start = self.position;

            // Get field name
            if let Token::Atom((start, end)) = self.current_token() {
                let field_name = String::from_utf8(self.source.as_bytes()[start..end].to_vec())
                    .map_err(|_| ParseError::InvalidUtf8 { position: start })?;
                self.consume();
                self.skip_whitespace();

                // Check for rename syntax: field: binding
                let binding_name = if matches!(self.current_token(), Token::Colon) {
                    self.consume();
                    self.skip_whitespace();

                    if let Token::Atom((bind_start, bind_end)) = self.current_token() {
                        let binding = String::from_utf8(
                            self.source.as_bytes()[bind_start..bind_end].to_vec(),
                        )
                        .map_err(|_| ParseError::InvalidUtf8 {
                            position: bind_start,
                        })?;
                        self.consume();
                        self.skip_whitespace();
                        Some(binding)
                    } else {
                        return Err(ParseError::InvalidPattern {
                            message: "Expected binding name after ':'".to_string(),
                            position: self.position,
                        });
                    }
                } else {
                    None
                };

                fields.push(FieldPattern {
                    field_name,
                    binding_name,
                    token_range: TokenRange::new(field_start, self.position),
                });

                // Allow comma or newline between fields
                if matches!(self.current_token(), Token::Comma) {
                    self.consume();
                    self.skip_whitespace();
                } else {
                    self.skip_whitespace();
                }
            } else {
                return Err(ParseError::InvalidPattern {
                    message: "Expected field name in pattern".to_string(),
                    position: self.position,
                });
            }
        }

        // Consume '}'
        self.consume();
        Ok(fields)
    }

    fn compose_binary_op(&mut self, lhs: Ast, current_token: Token, rhs: Ast) -> ParseResult<Ast> {
        let start_position = lhs.token_range().start;
        let end_position = rhs.token_range().end + 1;
        let token_range = TokenRange::new(start_position, end_position);
        Ok(match current_token {
            Token::LessThanOrEqual => Ast::Condition {
                operator: crate::ir::Condition::LessThanOrEqual,
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::LessThan => Ast::Condition {
                operator: crate::ir::Condition::LessThan,
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::EqualEqual => Ast::Call {
                name: "beagle.core/equal".to_string(),
                args: vec![lhs, rhs],
                token_range,
            },
            Token::NotEqual => Ast::Condition {
                operator: crate::ir::Condition::NotEqual,
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::GreaterThan => Ast::Condition {
                operator: crate::ir::Condition::GreaterThan,
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::GreaterThanOrEqual => Ast::Condition {
                operator: crate::ir::Condition::GreaterThanOrEqual,
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::Plus => Ast::Add {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::Minus => Ast::Sub {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::Mul => Ast::Mul {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::Div => Ast::Div {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::Modulo => Ast::Modulo {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::ShiftLeft => Ast::ShiftLeft {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::ShiftRight => Ast::ShiftRight {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::ShiftRightZero => Ast::ShiftRightZero {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::BitWiseAnd => Ast::BitWiseAnd {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::BitWiseOr => Ast::BitWiseOr {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::BitWiseXor => Ast::BitWiseXor {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::Or => Ast::Or {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },
            Token::And => Ast::And {
                left: Box::new(lhs),
                right: Box::new(rhs),
                token_range,
            },

            Token::OpenBracket => {
                let index = Box::new(rhs);
                self.expect_close_bracket()?;
                Ast::IndexOperator {
                    array: Box::new(lhs),
                    index,
                    token_range,
                }
            }
            Token::Concat => Ast::Call {
                name: "beagle.core/string-concat".to_string(),
                args: vec![lhs, rhs],
                token_range,
            },
            Token::Equal => Ast::Assignment {
                name: Box::new(lhs),
                value: Box::new(rhs),
                token_range,
            },
            // Pipe-first: x |> f becomes f(x), x |> f(y) becomes f(x, y)
            Token::Pipe => match rhs {
                Ast::Call { name, args, .. } => {
                    let mut new_args = vec![lhs];
                    new_args.extend(args);
                    Ast::Call {
                        name,
                        args: new_args,
                        token_range,
                    }
                }
                Ast::Identifier(name, _) => Ast::Call {
                    name,
                    args: vec![lhs],
                    token_range,
                },
                _ => {
                    return Err(ParseError::InvalidExpression {
                        message: "Pipe operator requires a function or function call on the right"
                            .to_string(),
                        position: start_position,
                    });
                }
            },
            // Pipe-last: x |>> f(y) becomes f(y, x)
            Token::PipeLast => match rhs {
                Ast::Call { name, mut args, .. } => {
                    args.push(lhs);
                    Ast::Call {
                        name,
                        args,
                        token_range,
                    }
                }
                Ast::Identifier(name, _) => Ast::Call {
                    name,
                    args: vec![lhs],
                    token_range,
                },
                _ => {
                    return Err(ParseError::InvalidExpression {
                        message:
                            "Pipe-last operator requires a function or function call on the right"
                                .to_string(),
                        position: start_position,
                    });
                }
            },
            _ => {
                return Err(ParseError::InvalidExpression {
                    message: format!("Unexpected binary operator: {:?}", current_token),
                    position: start_position,
                });
            }
        })
    }

    fn expect_equal(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_equal() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "equal '='".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn is_equal(&self) -> bool {
        self.current_token() == Token::Equal
    }

    fn expect_colon(&mut self) -> ParseResult<()> {
        self.skip_whitespace();
        if self.is_colon() {
            self.consume();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "colon ':'".to_string(),
                found: self.get_token_repr(),
                position: self.position,
            })
        }
    }

    fn is_colon(&self) -> bool {
        self.current_token() == Token::Colon
    }

    fn is_newline(&self) -> bool {
        self.current_token() == Token::NewLine
    }

    pub fn from_file(arg: &str) -> ParseResult<Ast> {
        let source = std::fs::read_to_string(arg)?;
        let mut parser = Parser::new(arg.to_string(), source)?;
        parser.parse()
    }

    fn is_dot(&self) -> bool {
        self.current_token() == Token::Dot
    }

    fn parse_array(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.consume();
        let mut elements = Vec::new();
        while !self.at_end() && !self.is_close_bracket() {
            let elem =
                self.parse_expression(0, true, true)?
                    .ok_or_else(|| ParseError::UnexpectedEof {
                        expected: "array element".to_string(),
                    })?;
            elements.push(elem);
            self.skip_whitespace();
            if !self.is_close_bracket() {
                self.expect_comma()?;
            }
        }
        self.expect_close_bracket()?;
        let end_position = self.position;
        Ok(Ast::Array {
            array: elements,
            token_range: TokenRange::new(start_position, end_position),
        })
    }

    fn parse_map_literal(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.consume(); // consume '{'
        self.skip_whitespace();

        // Empty map case
        if self.is_close_curly() {
            self.consume();
            return Ok(Ast::MapLiteral {
                pairs: vec![],
                token_range: TokenRange::new(start_position, self.position),
            });
        }

        // Parse key-value pairs
        let mut pairs = Vec::new();
        loop {
            // Parse key (any expression)
            let key =
                self.parse_expression(0, true, true)?
                    .ok_or_else(|| ParseError::UnexpectedEof {
                        expected: "key expression in map literal".to_string(),
                    })?;

            self.skip_whitespace();

            // Parse value (any expression)
            let value =
                self.parse_expression(0, true, true)?
                    .ok_or_else(|| ParseError::UnexpectedEof {
                        expected: "value expression in map literal".to_string(),
                    })?;

            pairs.push((key, value));

            self.skip_whitespace();

            // Check if done
            if self.is_close_curly() {
                break;
            }

            // Optional comma between pairs
            if self.is_comma() {
                self.consume();
                self.skip_whitespace();
            }
        }

        self.expect_close_curly()?;

        Ok(Ast::MapLiteral {
            pairs,
            token_range: TokenRange::new(start_position, self.position),
        })
    }

    fn parse_set_literal(&mut self) -> ParseResult<Ast> {
        let start_position = self.position;
        self.consume(); // consume '#{' (already consumed in tokenizer, consume the token)
        self.skip_whitespace();

        // Empty set case
        if self.is_close_curly() {
            self.consume();
            return Ok(Ast::SetLiteral {
                elements: vec![],
                token_range: TokenRange::new(start_position, self.position),
            });
        }

        // Parse elements
        let mut elements = Vec::new();
        loop {
            // Parse element (any expression)
            let element =
                self.parse_expression(0, true, true)?
                    .ok_or_else(|| ParseError::UnexpectedEof {
                        expected: "element expression in set literal".to_string(),
                    })?;

            elements.push(element);

            self.skip_whitespace();

            // Check if done
            if self.is_close_curly() {
                break;
            }

            // Optional comma between elements
            if self.is_comma() {
                self.consume();
                self.skip_whitespace();
            }
        }

        self.expect_close_curly()?;

        Ok(Ast::SetLiteral {
            elements,
            token_range: TokenRange::new(start_position, self.position),
        })
    }

    fn is_close_bracket(&self) -> bool {
        self.current_token() == Token::CloseBracket
    }

    // TODO: I tried to fix this parse, I completely broke lots of things
    // need to get it back into working order around postfix operations and
    // my hacks for struct creation

    fn is_postfix(&self, lhs: &Ast, struct_creation_allowed: bool) -> bool {
        match self.current_token() {
            Token::Dot | Token::OpenParen | Token::OpenBracket => true,
            Token::OpenCurly => {
                if matches!(lhs, Ast::Identifier(_, _) | Ast::PropertyAccess { .. }) {
                    struct_creation_allowed
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn parse_postfix(
        &mut self,
        lhs: Ast,
        min_precedence: usize,
        struct_creation_allowed: bool,
    ) -> ParseResult<Option<Ast>> {
        match self.current_token() {
            Token::Dot => {
                self.consume();
                let rhs = self
                    .parse_expression(min_precedence, true, struct_creation_allowed)?
                    .ok_or_else(|| ParseError::UnexpectedEof {
                        expected: "expression after '.'".to_string(),
                    })?;
                let start_position = lhs.token_range().start;
                let end_position = rhs.token_range().end + 1;
                let token_range = TokenRange::new(start_position, end_position);
                if matches!(rhs, Ast::StructCreation { .. }) {
                    // turn this into enum creation
                    match rhs {
                        Ast::StructCreation {
                            name,
                            fields,
                            token_range,
                        } => Ok(Some(Ast::EnumCreation {
                            name: if let Ast::Identifier(name, _) = lhs {
                                name
                            } else {
                                return Err(ParseError::InvalidExpression {
                                    message: "Expected identifier before '.'".to_string(),
                                    position: start_position,
                                });
                            },
                            variant: name,
                            fields,
                            token_range,
                        })),
                        _ => Err(ParseError::InvalidExpression {
                            message: "Expected struct creation".to_string(),
                            position: start_position,
                        }),
                    }
                } else {
                    Ok(Some(Ast::PropertyAccess {
                        object: Box::new(lhs),
                        property: Box::new(rhs),
                        token_range,
                    }))
                }
            }
            Token::OpenParen => {
                let position = self.consume();
                let mut args = Vec::new();
                while !self.at_end() && !self.is_close_paren() {
                    let arg = self.parse_expression(0, true, true)?.ok_or_else(|| {
                        ParseError::UnexpectedEof {
                            expected: "argument in function call".to_string(),
                        }
                    })?;
                    args.push(arg);
                    self.skip_whitespace();
                    if !self.is_close_paren() {
                        self.expect_comma()?;
                    }
                }
                self.expect_close_paren()?;
                Ok(Some(Ast::Call {
                    name: match lhs {
                        Ast::Identifier(name, _) => name,
                        _ => {
                            return Err(ParseError::InvalidExpression {
                                message: "Expected identifier before '('".to_string(),
                                position,
                            });
                        }
                    },
                    args,
                    token_range: TokenRange::new(position, self.position),
                }))
            }
            Token::OpenBracket => {
                let position = self.consume();
                let index = self.parse_expression(0, true, true)?.ok_or_else(|| {
                    ParseError::UnexpectedEof {
                        expected: "index expression".to_string(),
                    }
                })?;
                self.expect_close_bracket()?;
                Ok(Some(Ast::IndexOperator {
                    array: Box::new(lhs),
                    index: Box::new(index),
                    token_range: TokenRange::new(position, self.position),
                }))
            }
            Token::OpenCurly => {
                let position = self.consume();
                let fields = self.parse_struct_fields_creations()?;
                self.expect_close_curly()?;
                match lhs {
                    Ast::Identifier(name, _) => Ok(Some(Ast::StructCreation {
                        name,
                        fields,
                        token_range: TokenRange::new(position, self.position),
                    })),
                    Ast::PropertyAccess {
                        object,
                        property,
                        token_range,
                    } => {
                        // TODO: Ugly
                        let enum_name = match *property {
                            Ast::Identifier(name, _) => name,
                            _ => {
                                return Err(ParseError::InvalidExpression {
                                    message: "Expected identifier for enum variant".to_string(),
                                    position,
                                });
                            }
                        };
                        let parent_name = match *object {
                            Ast::Identifier(name, _) => name,
                            _ => {
                                return Err(ParseError::InvalidExpression {
                                    message: "Expected identifier for enum name".to_string(),
                                    position,
                                });
                            }
                        };
                        Ok(Some(Ast::EnumCreation {
                            name: parent_name,
                            variant: enum_name,
                            fields,
                            token_range,
                        }))
                    }
                    _ => Err(ParseError::InvalidExpression {
                        message: "Expected identifier or property access before '{'".to_string(),
                        position,
                    }),
                }
            }
            _ => Ok(None),
        }
    }
}

#[test]
fn test_tokenizer2() {
    let mut tokenizer = Tokenizer::new();
    let input = "
        fn hello() {
            print(\"Hello World!\")
        }
    ";
    let input_bytes = input.as_bytes();
    let (tokens, _mappings) = tokenizer.parse_all(input_bytes).unwrap();
    let literals = tokens
        .iter()
        .map(|x| x.literal(input_bytes).unwrap())
        .collect::<Vec<String>>()
        .join("");
    assert_eq!(literals, input);
}

#[test]
fn test_parse() {
    let mut parser = Parser::new(
        "test".to_string(),
        String::from(
            "
    fn hello() {
        print(\"Hello World!\")
    }",
        ),
    )
    .unwrap();

    let ast = parser.parse().unwrap();
    println!("{:#?}", ast);
}
#[test]
fn parse_array() {
    let mut parser = Parser::new(
        "test".to_string(),
        String::from(
            "
    let x = [1, 2, 3, 4]
    ",
        ),
    )
    .unwrap();

    let ast = parser.parse().unwrap();
    println!("{:#?}", ast);
}

#[test]
fn test_parse2() {
    let mut parser = Parser::new(
        "test".to_string(),
        String::from(
            "
    fn hello(x) {
        if x + 1 > 2 {
            print(\"Hello World!\")
        } else {
            print(\"Hello World!!!!\")
        }
    }",
        ),
    )
    .unwrap();

    let ast = parser.parse().unwrap();
    println!("{:#?}", ast);
}

#[test]
fn test_parens() {
    let mut parser =
        Parser::new("test".to_string(), String::from("(2 + 2) * 3 - (2 * 4)")).unwrap();

    let ast = parser.parse().unwrap();
    println!("{:#?}", ast);
}

#[test]
fn test_empty_function() {
    let mut parser = Parser::new(
        "test".to_string(),
        String::from(
            "
    fn empty(n) {

    }",
        ),
    )
    .unwrap();

    let ast = parser.parse().unwrap();
    println!("{:#?}", ast);
}

// Kind of pointless sense I have to pass a string
// stringify wasn't preserving new lines
// and I've now made my language new line sensitive
// for things like enums and structs
// Not sure about that decision yet
#[macro_export]
macro_rules! parse {
    ($input:expr) => {{
        let mut parser = Parser::new("test".to_string(), $input.to_string()).unwrap();
        parser.print_tokens();
        parser.parse().unwrap()
    }};
}
#[test]
fn parse_simple_enum() {
    let ast = parse! {
        "enum Color {
            red
            green
            blue
        }"
    };
    println!("{:#?}", ast);
}

#[test]
fn parse_struct_style_enum() {
    let ast = parse! {
        "enum Action {
            pause,
            run {
                direction
                speed
            },
            stop { time, location }
        }"
    };
    println!("{:#?}", ast);
}

#[test]
fn parse_enum_creation_simple() {
    let ast = parse! {
        "let action = Action.run"
    };
    println!("{:#?}", ast);
}

#[test]
fn parse_enum_creation_complex() {
    let ast = parse! {
        "let action = Action.run {
            direction: 1
            speed: 2
        }"
    };
    println!("{:#?}", ast);
}

#[test]
fn parse_property_access_if() {
    let ast = parse! {
        "

        if action.speed >= 3 {
            println(\"Fast\")
        }"
    };
    println!("{:#?}", ast);
}

#[test]
fn test_parsing_ast() {
    let ast = parse! {
        "array/read_field(node, (index >>> level) & 31)"
    };
    println!("{:#?}", ast);
}

#[test]
fn parse_struct_creation() {
    let ast = parse! {
        "let z = TreeNode {
            left: y
            right: y
        }"
    };
    println!("{:#?}", ast);
}

#[test]
fn parse_expression() {
    let ast = parse! {
        "current_state.rect_y + current_state.dy <= 0 ||
                         current_state.rect_y + current_state.dy + 180 >= screen_height"
    };
    println!("{:#?}", ast);
}
