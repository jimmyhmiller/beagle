// Stolen from my editor, so probably not great
// Need to deal with failure?
// Maybe not at the token level?

use crate::{
    Data,
    ast::{Ast, TokenRange},
    builtins::debugger,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    OpenParen,
    CloseParen,
    OpenCurly,
    CloseCurly,
    OpenBracket,
    CloseBracket,
    SemiColon,
    Colon,
    Comma,
    Dot,
    NewLine,
    Fn,
    Loop,
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
    Integer((usize, usize)),
    Float((usize, usize)),
    // I should replace this with builtins
    // like fn and stuff
    Atom((usize, usize)),
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
            | Token::Equal => true,
            _ => false,
        }
    }

    fn literal(&self, input_bytes: &[u8]) -> String {
        match self {
            Token::OpenParen => "(".to_string(),
            Token::CloseParen => ")".to_string(),
            Token::OpenCurly => "{".to_string(),
            Token::CloseCurly => "}".to_string(),
            Token::OpenBracket => "[".to_string(),
            Token::CloseBracket => "]".to_string(),
            Token::SemiColon => ";".to_string(),
            Token::Colon => ":".to_string(),
            Token::Comma => ",".to_string(),
            Token::Dot => ".".to_string(),
            Token::NewLine => "\n".to_string(),
            Token::Fn => "fn".to_string(),
            Token::And => "&&".to_string(),
            Token::Or => "||".to_string(),
            Token::LessThanOrEqual => "<=".to_string(),
            Token::LessThan => "<".to_string(),
            Token::Equal => "=".to_string(),
            Token::EqualEqual => "==".to_string(),
            Token::NotEqual => "!=".to_string(),
            Token::GreaterThan => ">".to_string(),
            Token::GreaterThanOrEqual => ">=".to_string(),
            Token::Plus => "+".to_string(),
            Token::Minus => "-".to_string(),
            Token::Mul => "*".to_string(),
            Token::Div => "/".to_string(),
            Token::Concat => "++".to_string(),
            Token::True => "true".to_string(),
            Token::False => "false".to_string(),
            Token::Null => "null".to_string(),
            Token::ShiftRight => ">>".to_string(),
            Token::ShiftRightZero => ">>>".to_string(),
            Token::ShiftLeft => "<<".to_string(),
            Token::BitWiseAnd => "&".to_string(),
            Token::BitWiseOr => "|".to_string(),
            Token::BitWiseXor => "^".to_string(),
            Token::Loop => "loop".to_string(),
            Token::If => "if".to_string(),
            Token::Else => "else".to_string(),
            Token::Let => "let".to_string(),
            Token::Mut => "mut".to_string(),
            Token::Struct => "struct".to_string(),
            Token::Enum => "enum".to_string(),
            Token::Namespace => "namespace".to_string(),
            Token::Import => "import".to_string(),
            Token::Protocol => "protocol".to_string(),
            Token::Extend => "extend".to_string(),
            Token::As => "as".to_string(),
            Token::With => "with".to_string(),
            Token::Try => "try".to_string(),
            Token::Catch => "catch".to_string(),
            Token::Throw => "throw".to_string(),
            Token::Comment((start, end))
            | Token::Atom((start, end))
            | Token::Spaces((start, end))
            | Token::String((start, end))
            | Token::Integer((start, end))
            | Token::Float((start, end)) => {
                String::from_utf8(input_bytes[*start..*end].to_vec()).unwrap()
            }
            Token::Never => panic!("Should never be called"),
        }
    }
}

enum Associativity {
    Left,
    #[allow(dead_code)]
    Right,
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
        self.consume(input_bytes);
        while !self.at_end(input_bytes) && !self.is_quote(input_bytes) {
            // TOOD: Better escape handling
            if self.current_byte(input_bytes) == b'\\' && self.peek(input_bytes).unwrap() == b'"' {
                self.consume(input_bytes);
            }
            self.consume(input_bytes);
        }
        if !self.at_end(input_bytes) {
            self.consume(input_bytes);
        }
        Token::String((start, self.position))
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

    pub fn parse_number(&mut self, input_bytes: &[u8]) -> Token {
        let mut is_float = false;
        let start = self.position;
        while !self.at_end(input_bytes)
            && (self.is_valid_number_char(input_bytes) || self.current_byte(input_bytes) == PERIOD)
        {
            // Need to handle making sure there is only one "."
            if self.current_byte(input_bytes) == PERIOD {
                is_float = true;
            }
            self.consume(input_bytes);
        }
        if is_float {
            Token::Float((start, self.position))
        } else {
            Token::Integer((start, self.position))
        }
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
        {
            self.consume(input_bytes);
        }
        match &input_bytes[start..self.position] {
            b"fn" => Token::Fn,
            b"loop" => Token::Loop,
            b"if" => Token::If,
            b"else" => Token::Else,
            b"<=" => Token::LessThanOrEqual,
            b"<" => Token::LessThan,
            b"=" => Token::Equal,
            b"==" => Token::EqualEqual,
            b"!=" => Token::NotEqual,
            b">" => Token::GreaterThan,
            b">=" => Token::GreaterThanOrEqual,
            b"+" => Token::Plus,
            b"++" => Token::Concat,
            b"-" => Token::Minus,
            b"*" => Token::Mul,
            b"/" => Token::Div,
            b">>" => Token::ShiftRight,
            b">>>" => Token::ShiftRightZero,
            b"<<" => Token::ShiftLeft,
            b"&" => Token::BitWiseAnd,
            b"|" => Token::BitWiseOr,
            b"^" => Token::BitWiseXor,
            b"||" => Token::Or,
            b"&&" => Token::And,
            b"true" => Token::True,
            b"false" => Token::False,
            b"null" => Token::Null,
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
            _ => Token::Atom((start, self.position)),
        }
    }

    pub fn parse_single(&mut self, input_bytes: &[u8]) -> Option<Token> {
        if self.at_end(input_bytes) {
            return None;
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
            self.parse_number(input_bytes)
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
            Token::Colon
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
        } else if self.is_dot(input_bytes) {
            self.consume(input_bytes);
            Token::Dot
        } else {
            // println!("identifier");
            self.parse_identifier(input_bytes)
        };
        Some(result)
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

    // TODO: Make a lazy method of tokenizing
    pub fn parse_all(&mut self, input_bytes: &[u8]) -> (Vec<Token>, Vec<(usize, usize)>) {
        let mut result = Vec::new();
        let mut token_line_column_map = Vec::new();
        while !self.at_end(input_bytes) {
            if let Some(token) = self.parse_single(input_bytes) {
                result.push(token);
                token_line_column_map.push((self.line, self.column));
            }
        }
        self.position = 0;
        (result, token_line_column_map)
    }
}

#[test]
fn test_tokenizer1() {
    let mut tokenizer = Tokenizer::new();
    let input = "hello world";
    let input_bytes = input.as_bytes();
    let result = tokenizer.parse_all(input_bytes);
    assert_eq!(result.0.len(), 3);
    assert_eq!(result.0[0], Token::Atom((0, 5)));
    assert_eq!(result.0[1], Token::Spaces((5, 6)));
    assert_eq!(result.0[2], Token::Atom((6, 11)));
}

pub struct Parser {
    file_name: String,
    source: String,
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    position: usize,
    tokens: Vec<Token>,
    token_line_column_map: Vec<(usize, usize)>,
}

impl Parser {
    pub fn new(file_name: String, source: String) -> Parser {
        let mut tokenizer = Tokenizer::new();
        let input_bytes = source.as_bytes();
        // TODO: It is probably better not to parse all at once
        let (tokens, token_line_column_map) = tokenizer.parse_all(input_bytes);

        debugger(crate::Message {
            kind: "tokens".to_string(),
            data: Data::Tokens {
                file_name: file_name.clone(),
                tokens: tokens
                    .clone()
                    .iter()
                    .map(|x| x.literal(input_bytes))
                    .collect(),
                token_line_column_map: token_line_column_map.clone(),
            },
        });

        Parser {
            file_name,
            source,
            tokenizer,
            position: 0,
            tokens,
            token_line_column_map,
        }
    }

    pub fn current_location(&self) -> String {
        let (line, column) = self.token_line_column_map[self.position];
        format!("{}:{}:{}", self.file_name, line, column)
    }

    pub fn print_tokens(&self) {
        for token in &self.tokens {
            println!("{:?}", token);
        }
    }

    pub fn parse(&mut self) -> Ast {
        Ast::Program {
            elements: self.parse_elements(),
            token_range: TokenRange::new(0, self.tokens.len()),
        }
    }

    fn parse_elements(&mut self) -> Vec<Ast> {
        let mut result = Vec::new();
        while !self.at_end() {
            if let Some(elem) = self.parse_expression(0, true, true) {
                result.push(elem);
            } else {
                break;
            }
        }
        result
    }

    fn at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    fn get_precedence(&self) -> (usize, Associativity) {
        match self.current_token() {
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
            // Multiplication, division, etc.
            Token::Mul | Token::Div => (50, Associativity::Left),
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
    ) -> Option<Ast> {
        let mut min_precedence = min_precedence;
        if should_skip {
            self.skip_whitespace();
        }
        if self.at_end() {
            return None;
        }

        let mut lhs = self.parse_atom(min_precedence)?;
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
            lhs = self.parse_postfix(lhs, next_min_precedence, struct_creation_allowed)?;
            self.skip_spaces();
        }
        min_precedence = old_min_precedence;

        // TODO: This is ugly
        self.skip_spaces();
        loop {
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
            let rhs = self.parse_expression(next_min_precedence, true, struct_creation_allowed)?;

            lhs = self.compose_binary_op(lhs.clone(), current_token, rhs);
        }

        Some(lhs)
    }

    fn parse_atom(&mut self, min_precedence: usize) -> Option<Ast> {
        match self.current_token() {
            Token::Fn => Some(self.parse_function()),
            Token::Loop => Some(self.parse_loop()),
            Token::Struct => Some(self.parse_struct()),
            Token::Enum => Some(self.parse_enum()),
            Token::If => Some(self.parse_if()),
            Token::Try => Some(self.parse_try()),
            Token::Throw => Some(self.parse_throw()),
            Token::Namespace => Some(self.parse_namespace()),
            Token::Import => Some(self.parse_import()),
            Token::Protocol => Some(self.parse_protocol()),
            Token::Extend => Some(self.parse_extend()),
            Token::Atom((start, end)) => {
                let start_position = self.position;
                // Gross
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap();
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
                    Some(self.parse_call(name, start_position))
                }
                // TODO: Hack to try and let struct creation work in ambiguous contexts
                // like if. Need a better way.
                else if self.is_open_curly() && min_precedence == 0 {
                    Some(self.parse_struct_creation(name, start_position))
                } else {
                    Some(Ast::Identifier(name, self.position))
                }
            }
            Token::String((start, end)) => {
                // Gross
                let mut value =
                    String::from_utf8(self.source.as_bytes()[start + 1..end - 1].to_vec()).unwrap();
                // TODO: Test escapes properly
                // Maybe token shouldn't have a range but an actual string value
                // Or I do both
                value = stripslashes(&value);
                let position = self.consume();
                Some(Ast::String(value, position))
            }
            Token::Integer((start, end)) => {
                // Gross
                let value = String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap();
                let position = self.consume();
                Some(Ast::IntegerLiteral(value.parse::<i64>().unwrap(), position))
            }
            Token::Float((start, end)) => {
                // Gross
                let value = String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap();
                let position = self.consume();
                Some(Ast::FloatLiteral(value, position))
            }
            Token::True => {
                let position = self.consume();
                Some(Ast::True(position))
            }
            Token::False => {
                let position = self.consume();
                Some(Ast::False(position))
            }
            Token::Null => {
                let position = self.consume();
                Some(Ast::Null(position))
            }
            Token::Let => {
                let start_position = self.position;
                self.consume();
                self.move_to_next_non_whitespace();

                if self.peek_next_non_whitespace() == Token::Mut {
                    self.consume();
                    self.move_to_next_non_whitespace();
                    let name_position = self.position;
                    let name = match self.current_token() {
                        Token::Atom((start, end)) => {
                            // Gross
                            String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
                        }
                        _ => panic!("Expected variable name got {}", self.get_token_repr()),
                    };
                    self.consume();
                    self.move_to_next_non_whitespace();
                    self.expect_equal();
                    self.move_to_next_non_whitespace();
                    let value = self.parse_expression(0, true, true).unwrap();
                    let end_position = self.position;
                    return Some(Ast::LetMut {
                        name: Box::new(Ast::Identifier(name, name_position)),
                        value: Box::new(value),
                        token_range: TokenRange::new(start_position, end_position),
                    });
                }
                let name_position = self.position;
                let name = match self.current_token() {
                    Token::Atom((start, end)) => {
                        // Gross
                        String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
                    }
                    _ => panic!("Expected variable name"),
                };
                self.move_to_next_non_whitespace();
                self.expect_equal();
                self.move_to_next_non_whitespace();
                let value = self.parse_expression(0, true, true).unwrap();
                let end_position = self.position;
                Some(Ast::Let {
                    name: Box::new(Ast::Identifier(name, name_position)),
                    value: Box::new(value),
                    token_range: TokenRange::new(start_position, end_position),
                })
            }
            Token::NewLine | Token::Spaces(_) | Token::Comment(_) => {
                self.consume();
                self.parse_atom(min_precedence)
            }
            Token::OpenParen => {
                self.consume();
                let result = self.parse_expression(0, true, true);
                self.expect_close_paren();
                result
            }
            Token::OpenBracket => {
                let result = self.parse_array();
                Some(result)
            }
            _ => panic!(
                "Expected atom, got {} at line {}",
                self.get_token_repr(),
                self.current_location()
            ),
        }
    }

    fn parse_function(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                self.move_to_next_non_whitespace();
                // Gross
                Some(String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap())
            }
            _ => None,
        };
        self.expect_open_paren();
        let args = self.parse_args();
        self.expect_close_paren();
        let body = self.parse_block();
        let end_position = self.position;
        Ast::Function {
            name,
            args,
            body,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_loop(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let body = self.parse_block();
        let end_position = self.position;
        Ast::Loop {
            body,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_struct(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_atom();
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            _ => panic!("Expected struct name"),
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly();
        let fields = self.parse_struct_fields();
        self.expect_close_curly();
        let end_position = self.position;
        Ast::Struct {
            name,
            fields,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_protocol(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_atom();
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            _ => panic!("Expected protocol name"),
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly();
        let body = self.parse_protocol_body();
        self.expect_close_curly();
        let end_position = self.position;
        Ast::Protocol {
            name,
            body,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_protocol_body(&mut self) -> Vec<Ast> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            self.skip_spaces();
            result.push(self.parse_protocol_member());
            self.skip_spaces();
            if !self.is_close_curly() && self.peek_next_non_whitespace() != Token::CloseCurly {
                self.data_delimiter();
            }
            self.skip_spaces();
        }
        result
    }

    fn parse_protocol_member(&mut self) -> Ast {
        match self.current_token() {
            Token::Fn => {
                self.consume();
                self.move_to_next_non_whitespace();
                let name = match self.current_token() {
                    Token::Atom((start, end)) => {
                        // Gross
                        String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
                    }
                    _ => panic!("Expected protocol member name"),
                };
                self.move_to_next_non_whitespace();
                self.expect_open_paren();
                let args = self.parse_args();
                self.expect_close_paren();
                self.skip_spaces();
                if self.is_open_curly() {
                    let body = self.parse_block();
                    let end_position = self.position;
                    Ast::Function {
                        name: Some(name),
                        args,
                        body,
                        token_range: TokenRange::new(self.position, end_position),
                    }
                } else {
                    let end_position = self.position;
                    Ast::FunctionStub {
                        name,
                        args,
                        token_range: TokenRange::new(self.position, end_position),
                    }
                }
            }
            _ => panic!("Expected protocol member"),
        }
    }

    fn parse_extend(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_atom();
        let target_type = match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            _ => panic!("Expected extend name"),
        };
        self.move_to_next_non_whitespace();
        self.expect_with();
        self.move_to_next_non_whitespace();
        let protocol = match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            _ => panic!("Expected extend name"),
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly();
        let body = self.parse_extend_body();
        self.expect_close_curly();
        let end_position = self.position;
        Ast::Extend {
            target_type,
            protocol,
            body,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_extend_body(&mut self) -> Vec<Ast> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            self.skip_spaces();
            result.push(self.parse_extend_member());
            self.skip_spaces();
            if !self.is_close_curly() {
                self.data_delimiter();
            }
            self.skip_spaces();
        }
        result
    }

    fn parse_extend_member(&mut self) -> Ast {
        self.skip_whitespace();
        self.parse_function()
    }

    fn parse_enum(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_atom();
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            _ => panic!("Expected enum name"),
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly();
        let variants = self.parse_enum_variants();
        self.expect_close_curly();
        let end_position = self.position;
        Ast::Enum {
            name,
            variants,
            token_range: TokenRange::new(start_position, end_position),
        }
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

    fn expect_open_paren(&mut self) {
        self.skip_whitespace();
        if self.is_open_paren() {
            self.consume();
        } else {
            let (line, column) = self.token_line_column_map[self.position];
            panic!(
                "Expected open paren {:?} at {}:{}",
                self.get_token_repr(),
                line,
                column
            );
        }
    }

    fn expect_with(&mut self) {
        self.skip_whitespace();
        if self.is_with() {
            self.consume();
        } else {
            panic!(
                "Expected with {:?} at {}",
                self.get_token_repr(),
                self.current_location()
            );
        }
    }

    fn expect_close_bracket(&mut self) {
        self.skip_whitespace();
        if self.is_close_bracket() {
            self.consume();
        } else {
            panic!(
                "Expected close bracket {:?} at {}",
                self.get_token_repr(),
                self.current_location()
            );
        }
    }

    fn expect_comma(&mut self) {
        self.skip_whitespace();
        if self.is_comma() {
            self.consume();
        } else {
            panic!(
                "Expected comma {:?} at {}",
                self.get_token_repr(),
                self.current_location()
            );
        }
    }

    fn expect_string(&mut self) -> String {
        self.skip_whitespace();
        match self.current_token() {
            Token::String((start, end)) => {
                self.consume();
                // Gross
                String::from_utf8(self.source.as_bytes()[start + 1..end - 1].to_vec()).unwrap()
            }
            _ => panic!("Expected string got {:?}", self.get_token_repr()),
        }
    }

    fn expect_atom(&mut self) -> String {
        self.skip_whitespace();
        match self.current_token() {
            Token::Atom((start, end)) => {
                self.consume();
                // Gross
                String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap()
            }
            _ => panic!("Expected atom got {:?}", self.get_token_repr()),
        }
    }

    fn expect_as(&mut self) {
        self.skip_whitespace();
        if self.is_as() {
            self.consume();
        } else {
            panic!("Expected as got {:?}", self.get_token_repr());
        }
    }

    fn is_open_paren(&self) -> bool {
        self.current_token() == Token::OpenParen
    }

    fn is_comma(&self) -> bool {
        self.current_token() == Token::Comma
    }

    fn parse_args(&mut self) -> Vec<String> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_paren() {
            result.push(self.parse_arg());
            self.skip_whitespace();
        }
        result
    }

    fn parse_struct_fields(&mut self) -> Vec<Ast> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            self.skip_spaces();
            result.push(self.parse_struct_field());
            self.skip_spaces();
            if !self.is_close_curly() {
                self.data_delimiter();
            }
            self.skip_spaces();
        }
        result
    }

    fn parse_struct_field(&mut self) -> Ast {
        match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap();
                let position = self.consume();
                Ast::Identifier(name, position)
            }
            _ => panic!(
                "Expected field name got {:?} at {}",
                self.current_token(),
                self.current_location()
            ),
        }
    }

    fn parse_enum_variants(&mut self) -> Vec<Ast> {
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            result.push(self.parse_enum_variant());
            self.skip_whitespace();
        }
        result
    }

    fn parse_enum_variant(&mut self) -> Ast {
        // We need to parse enum variants that are just a name
        // and enum variants that are struct like

        match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap();
                let position = self.consume();
                self.skip_spaces();
                let result = if self.is_open_curly() {
                    let start_position = self.consume();
                    let fields = self.parse_struct_fields();
                    self.expect_close_curly();
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
                self.data_delimiter();
                result
            }
            _ => panic!(
                "Expected variant name got {:?} on line {}",
                self.current_token(),
                self.current_location()
            ),
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

    fn data_delimiter(&mut self) {
        self.skip_spaces();
        if self.is_comma() || self.is_newline() {
            self.consume();
        } else {
            panic!(
                "Expected comma or new_line got {:?} at {}",
                self.get_token_repr(),
                self.current_location()
            );
        }
    }

    fn parse_arg(&mut self) -> String {
        match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap();
                self.consume();
                self.skip_whitespace();
                if !self.is_close_paren() {
                    self.expect_comma();
                }
                name
            }
            _ => panic!(
                "Expected arg got {:?} on line {}",
                self.current_token(),
                self.current_location()
            ),
        }
    }

    fn expect_close_paren(&mut self) {
        self.skip_whitespace();
        if self.is_close_paren() {
            self.consume();
        } else {
            panic!(
                "Expected close paren got {:?} on line {}",
                self.current_token(),
                self.current_location()
            );
        }
    }

    fn is_close_paren(&self) -> bool {
        self.current_token() == Token::CloseParen
    }

    fn parse_block(&mut self) -> Vec<Ast> {
        self.expect_open_curly();
        let mut result = Vec::new();
        self.skip_whitespace();
        while !self.at_end() && !self.is_close_curly() {
            if let Some(elem) = self.parse_expression(0, true, true) {
                result.push(elem);
            } else {
                break;
            }
            self.skip_whitespace();
        }
        self.expect_close_curly();
        result
    }

    fn expect_open_curly(&mut self) {
        self.skip_whitespace();
        if self.is_open_curly() {
            self.consume();
        } else {
            panic!(
                "Expected open curly {} at line {}",
                self.get_token_repr(),
                self.current_location()
            );
        }
    }

    fn is_open_curly(&self) -> bool {
        self.current_token() == Token::OpenCurly
    }

    fn is_close_curly(&self) -> bool {
        self.current_token() == Token::CloseCurly
    }

    fn expect_close_curly(&mut self) {
        self.skip_whitespace();
        if self.is_close_curly() {
            self.consume();
        } else {
            panic!(
                "Expected close curly got {:?} at line {}",
                self.get_token_repr(),
                self.current_location()
            );
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
            self.tokens[self.position]
        }
    }

    fn parse_namespace(&mut self) -> Ast {
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
                _ => panic!("Expected atom"),
            }
            self.consume();
        }
        if name.is_empty() {
            panic!(
                "Unexpected token {:?} at {}",
                self.current_token(),
                self.current_location()
            );
        }
        self.consume();
        let end_position = self.position;
        Ast::Namespace {
            name,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_import(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let library_name = self.expect_string();
        self.expect_as();
        let name_position = self.position;
        let alias = Box::new(Ast::Identifier(self.expect_atom(), name_position));
        let end_position = self.position;
        Ast::Import {
            library_name,
            alias,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_call(&mut self, name: String, start_position: usize) -> Ast {
        self.expect_open_paren();
        let mut args = Vec::new();
        while !self.at_end() && !self.is_close_paren() {
            if let Some(arg) = self.parse_expression(0, true, true) {
                args.push(arg);
                self.skip_whitespace();
                if !self.is_close_paren() {
                    self.expect_comma();
                }
            } else {
                break;
            }
        }
        self.expect_close_paren();
        let end_position = self.position;
        Ast::Call {
            name,
            args,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_struct_creation(&mut self, name: String, start_position: usize) -> Ast {
        self.expect_open_curly();
        let fields = self.parse_struct_fields_creations();

        self.expect_close_curly();
        let end_position = self.position;
        Ast::StructCreation {
            name,
            fields,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_struct_fields_creations(&mut self) -> Vec<(String, Ast)> {
        let mut fields = Vec::new();
        while !self.at_end() && !self.is_close_curly() {
            if let Some(field) = self.parse_struct_field_creation() {
                fields.push(field);
            } else {
                break;
            }
        }
        fields
    }

    fn parse_struct_field_creation(&mut self) -> Option<(String, Ast)> {
        self.skip_whitespace();
        match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                let name = String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap();
                self.consume();
                self.skip_spaces();
                self.expect_colon();
                self.skip_spaces();
                let value = self.parse_expression(0, false, true).unwrap();
                if !self.is_close_curly() {
                    self.data_delimiter();
                }
                Some((name, value))
            }
            _ => None,
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

    fn parse_if(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_non_whitespace();
        let condition = Box::new(self.parse_expression(1, true, false).unwrap());
        let then = self.parse_block();
        self.move_to_next_non_whitespace();
        if self.is_else() {
            self.consume();
            self.skip_whitespace();
            if self.is_if() {
                self.consume();
                let else_ = vec![self.parse_if()];
                let end_position = self.position;
                return Ast::If {
                    condition,
                    then,
                    else_,
                    token_range: TokenRange::new(start_position, end_position),
                };
            }
            self.skip_whitespace();
            let else_ = self.parse_block();
            let end_position = self.position;
            Ast::If {
                condition,
                then,
                else_,
                token_range: TokenRange::new(start_position, end_position),
            }
        } else {
            let end_position = self.position;
            Ast::If {
                condition,
                then,
                else_: Vec::new(),
                token_range: TokenRange::new(start_position, end_position),
            }
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

    fn parse_try(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_non_whitespace();

        // Parse try block
        let body = self.parse_block();

        // Expect 'catch'
        self.move_to_next_non_whitespace();
        if !matches!(self.current_token(), Token::Catch) {
            panic!("Expected 'catch' after try block");
        }
        self.consume();
        self.move_to_next_non_whitespace();

        // Parse catch parameter: catch(e)
        if !matches!(self.current_token(), Token::OpenParen) {
            panic!("Expected '(' after 'catch'");
        }
        self.consume();
        self.skip_whitespace();

        // Get the exception binding identifier
        let exception_binding = if let Token::Atom((start, end)) = self.current_token() {
            let binding = String::from_utf8(self.source.as_bytes()[start..end].to_vec()).unwrap();
            self.consume();
            binding
        } else {
            panic!("Expected identifier for exception binding");
        };

        self.skip_whitespace();
        if !matches!(self.current_token(), Token::CloseParen) {
            panic!("Expected ')' after exception binding");
        }
        self.consume();

        // Parse catch block
        let catch_body = self.parse_block();

        let end_position = self.position;
        Ast::Try {
            body,
            exception_binding,
            catch_body,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn parse_throw(&mut self) -> Ast {
        let start_position = self.position;
        self.move_to_next_non_whitespace();

        // throw is a function call: throw(value)
        if !matches!(self.current_token(), Token::OpenParen) {
            panic!("Expected '(' after 'throw'");
        }
        self.consume();
        self.skip_whitespace();

        let value = Box::new(self.parse_expression(1, true, false).unwrap());

        self.skip_whitespace();
        if !matches!(self.current_token(), Token::CloseParen) {
            panic!("Expected ')' after throw value");
        }
        self.consume();

        let end_position = self.position;
        Ast::Throw {
            value,
            token_range: TokenRange::new(start_position, end_position),
        }
    }

    fn compose_binary_op(&mut self, lhs: Ast, current_token: Token, rhs: Ast) -> Ast {
        let start_position = lhs.token_range().start;
        let end_position = rhs.token_range().end + 1;
        let token_range = TokenRange::new(start_position, end_position);
        match current_token {
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
                self.expect_close_bracket();
                Ast::IndexOperator {
                    array: Box::new(lhs),
                    index,
                    token_range,
                }
            }
            Token::Concat => Ast::Call {
                name: "beagle.core/string_concat".to_string(),
                args: vec![lhs, rhs],
                token_range,
            },
            Token::Equal => Ast::Assignment {
                name: Box::new(lhs),
                value: Box::new(rhs),
                token_range,
            },
            _ => panic!("Exepcted binary op got {:?}", current_token),
        }
    }

    fn expect_equal(&mut self) {
        self.skip_whitespace();
        if self.is_equal() {
            self.consume();
        } else {
            panic!(
                "Expected equal got {:?} on line {}",
                self.get_token_repr(),
                self.current_location()
            );
        }
    }

    fn is_equal(&self) -> bool {
        self.current_token() == Token::Equal
    }

    fn expect_colon(&mut self) {
        self.skip_whitespace();
        if self.is_colon() {
            self.consume();
        } else {
            panic!(
                "Expected colon got {} at line {}",
                self.get_token_repr(),
                self.current_location()
            );
        }
    }

    fn is_colon(&self) -> bool {
        self.current_token() == Token::Colon
    }

    fn is_newline(&self) -> bool {
        self.current_token() == Token::NewLine
    }

    pub fn from_file(arg: &str) -> Result<Ast, std::io::Error> {
        let source = std::fs::read_to_string(arg)?;
        let mut parser = Parser::new(arg.to_string(), source);
        Ok(parser.parse())
    }

    fn is_dot(&self) -> bool {
        self.current_token() == Token::Dot
    }

    fn parse_array(&mut self) -> Ast {
        let start_position = self.position;
        self.consume();
        let mut elements = Vec::new();
        while !self.at_end() && !self.is_close_bracket() {
            elements.push(self.parse_expression(0, true, true).unwrap());
            self.skip_whitespace();
            if !self.is_close_bracket() {
                self.expect_comma();
            }
        }
        self.expect_close_bracket();
        let end_position = self.position;
        Ast::Array {
            array: elements,
            token_range: TokenRange::new(start_position, end_position),
        }
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
    ) -> Option<Ast> {
        match self.current_token() {
            Token::Dot => {
                self.consume();
                let rhs = self.parse_expression(min_precedence, true, struct_creation_allowed)?;
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
                        } => Some(Ast::EnumCreation {
                            name: if let Ast::Identifier(name, _) = lhs {
                                name
                            } else {
                                panic!("Expected identifier")
                            },
                            variant: name,
                            fields,
                            token_range,
                        }),
                        _ => panic!("Expected struct creation"),
                    }
                } else {
                    Some(Ast::PropertyAccess {
                        object: Box::new(lhs),
                        property: Box::new(rhs),
                        token_range,
                    })
                }
            }
            Token::OpenParen => {
                let position = self.consume();
                let mut args = Vec::new();
                while !self.at_end() && !self.is_close_paren() {
                    args.push(self.parse_expression(0, true, true).unwrap());
                    self.skip_whitespace();
                    if !self.is_close_paren() {
                        self.expect_comma();
                    }
                }
                self.expect_close_paren();
                Some(Ast::Call {
                    name: match lhs {
                        Ast::Identifier(name, _) => name,
                        _ => panic!("Expected identifier"),
                    },
                    args,
                    token_range: TokenRange::new(position, self.position),
                })
            }
            Token::OpenBracket => {
                let position = self.consume();
                let index = self.parse_expression(0, true, true).unwrap();
                self.expect_close_bracket();
                Some(Ast::IndexOperator {
                    array: Box::new(lhs),
                    index: Box::new(index),
                    token_range: TokenRange::new(position, self.position),
                })
            }
            Token::OpenCurly => {
                let position = self.consume();
                let fields = self.parse_struct_fields_creations();
                self.expect_close_curly();
                match lhs {
                    Ast::Identifier(name, _) => Some(Ast::StructCreation {
                        name,
                        fields,
                        token_range: TokenRange::new(position, self.position),
                    }),
                    Ast::PropertyAccess {
                        object,
                        property,
                        token_range,
                    } => {
                        // TODO: Ugly
                        let enum_name = match *property {
                            Ast::Identifier(name, _) => name,
                            _ => panic!("Expected identifier"),
                        };
                        let parent_name = match *object {
                            Ast::Identifier(name, _) => name,
                            _ => panic!("Expected identifier"),
                        };
                        Some(Ast::EnumCreation {
                            name: parent_name,
                            variant: enum_name,
                            fields,
                            token_range,
                        })
                    }
                    _ => panic!("Expected identifier"),
                }
            }
            _ => None,
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
    let (tokens, _mappings) = tokenizer.parse_all(input_bytes);
    let literals = tokens
        .iter()
        .map(|x| x.literal(input_bytes))
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
    );

    let ast = parser.parse();
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
    );

    let ast = parser.parse();
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
    );

    let ast = parser.parse();
    println!("{:#?}", ast);
}

#[test]
fn test_parens() {
    let mut parser = Parser::new("test".to_string(), String::from("(2 + 2) * 3 - (2 * 4)"));

    let ast = parser.parse();
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
    );

    let ast = parser.parse();
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
        let mut parser = Parser::new("test".to_string(), $input.to_string());
        parser.print_tokens();
        parser.parse()
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
