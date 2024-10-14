// Stolen from my edtior, so probably not great
// Need to deal with failure?
// Maybe not at the token level?\

// TODO: Fix parsing parentheses for precedence

use crate::ast::Ast;

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
    If,
    Fn,
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
            | Token::ShiftRight
            | Token::ShiftRightZero
            | Token::ShiftLeft
            | Token::BitWiseAnd
            | Token::BitWiseOr
            | Token::BitWiseXor
            | Token::Or
            | Token::And
            | Token::Dot => true,
            _ => false,
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
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    pub fn new() -> Tokenizer {
        Tokenizer { position: 0 }
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
            self.consume();
        }
        // self.consume();
        Token::Comment((start, self.position))
    }

    pub fn consume(&mut self) {
        self.position += 1;
    }

    pub fn current_byte(&self, input_bytes: &[u8]) -> u8 {
        input_bytes[self.position]
    }

    pub fn next_n_bytes<'a>(&'a self, n: usize, input_bytes: &'a [u8]) -> &[u8] {
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
        self.consume();
        while !self.at_end(input_bytes) && !self.is_quote(input_bytes) {
            self.consume();
        }
        // TODO: Deal with escapes
        if !self.at_end(input_bytes) {
            self.consume();
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
            self.consume();
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
            self.consume();
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
            self.consume();
        }
        match &input_bytes[start..self.position] {
            b"fn" => Token::Fn,
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
            b"struct" => Token::Struct,
            b"enum" => Token::Enum,
            b"." => Token::Dot,
            b"namespace" => Token::Namespace,
            b"import" => Token::Import,
            b"as" => Token::As,
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
            self.consume();
            Token::NewLine
        } else if self.is_comment_start(input_bytes) {
            self.parse_comment(input_bytes)
        } else if self.is_open_paren(input_bytes) {
            self.consume();
            Token::OpenParen
        } else if self.is_close_paren(input_bytes) {
            self.consume();
            Token::CloseParen
        } else if self.is_valid_number_char(input_bytes) {
            self.parse_number(input_bytes)
        } else if self.is_quote(input_bytes) {
            self.parse_string(input_bytes)
        } else if self.is_semi_colon(input_bytes) {
            self.consume();
            Token::SemiColon
        } else if self.is_comma(input_bytes) {
            self.consume();
            Token::Comma
        } else if self.is_colon(input_bytes) {
            self.consume();
            Token::Colon
        } else if self.is_open_curly(input_bytes) {
            self.consume();
            Token::OpenCurly
        } else if self.is_close_curly(input_bytes) {
            self.consume();
            Token::CloseCurly
        } else if self.is_open_bracket(input_bytes) {
            self.consume();
            Token::OpenBracket
        } else if self.is_close_bracket(input_bytes) {
            self.consume();
            Token::CloseBracket
        } else if self.is_dot(input_bytes) {
            self.consume();
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

    pub fn get_line(&mut self, input_bytes: &[u8]) -> Vec<Token> {
        let mut result = Vec::new();
        while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
            if let Some(token) = self.parse_single(input_bytes) {
                result.push(token);
            }
        }
        result
    }

    pub fn _skip_lines(&mut self, n: usize, input_bytes: &[u8]) -> &mut Self {
        for _ in 0..n {
            while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
                self.consume();
            }
            if !self.at_end(input_bytes) {
                self.consume();
            }
        }
        self
    }

    // The downside of this approach is that I will parse very large buffers
    // all the way at once.
    pub fn parse_all(&mut self, input_bytes: &[u8]) -> Vec<Token> {
        let mut result = Vec::new();
        while !self.at_end(input_bytes) {
            if let Some(token) = self.parse_single(input_bytes) {
                result.push(token);
            }
        }
        self.position = 0;
        result
    }
}

#[test]
fn test_tokenizer1() {
    let mut tokenizer = Tokenizer::new();
    let input = "hello world";
    let input_bytes = input.as_bytes();
    let result = tokenizer.parse_all(input_bytes);
    assert_eq!(result.len(), 3);
    assert_eq!(result[0], Token::Atom((0, 5)));
    assert_eq!(result[1], Token::Spaces((5, 6)));
    assert_eq!(result[2], Token::Atom((6, 11)));
}

pub struct Parser {
    file_name: String,
    source: String,
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    position: usize,
    tokens: Vec<Token>,
    current_line: usize,
}

impl Parser {
    pub fn new(file_name: String, source: String) -> Parser {
        let mut tokenizer = Tokenizer::new();
        let input_bytes = source.as_bytes();
        // TODO: I is probably better not to parse all at once
        let tokens = tokenizer.parse_all(input_bytes);
        Parser {
            file_name,
            source,
            tokenizer,
            position: 0,
            tokens,
            current_line: 1,
        }
    }

    pub fn current_location(&self) -> String {
        format!("{}:{}", self.file_name, self.current_line)
    }

    pub fn print_tokens(&self) {
        for token in &self.tokens {
            println!("{:?}", token);
        }
    }

    pub fn parse(&mut self) -> Ast {
        Ast::Program {
            elements: self.parse_elements(),
        }
    }

    fn parse_elements(&mut self) -> Vec<Ast> {
        let mut result = Vec::new();
        while !self.at_end() {
            if let Some(elem) = self.parse_expression(0, true) {
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
            Token::LessThanOrEqual
            | Token::LessThan
            | Token::EqualEqual
            | Token::NotEqual
            | Token::GreaterThan
            | Token::GreaterThanOrEqual => (10, Associativity::Left),
            Token::Plus | Token::Minus | Token::And | Token::Or => (20, Associativity::Left),
            Token::Mul
            | Token::Div
            | Token::BitWiseAnd
            | Token::BitWiseOr
            | Token::BitWiseXor
            | Token::ShiftLeft
            | Token::ShiftRight
            | Token::ShiftRightZero => (30, Associativity::Left),
            // TODO: No idea what this should be
            Token::Dot => (40, Associativity::Left),
            _ => (0, Associativity::Left),
        }
    }

    // Based on
    // https://eli.thegreenplace.net/2012/08/02/parsing-expressions-by-precedence-climbing
    fn parse_expression(&mut self, min_precedence: usize, should_skip: bool) -> Option<Ast> {
        if should_skip {
            self.skip_whitespace();
        }
        if self.at_end() {
            return None;
        }
        let mut lhs = self.parse_atom(min_precedence)?;

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
            let rhs = self.parse_expression(next_min_precedence, true)?;

            lhs = self.compose_binary_op(lhs.clone(), current_token, rhs, min_precedence);
        }

        Some(lhs)
    }

    fn parse_atom(&mut self, min_precedence: usize) -> Option<Ast> {
        match self.current_token() {
            Token::Fn => {
                self.move_to_next_non_whitespace();
                Some(self.parse_function())
            }
            Token::Struct => {
                self.move_to_next_atom();
                Some(self.parse_struct())
            }
            Token::Enum => {
                self.move_to_next_atom();
                Some(self.parse_enum())
            }
            Token::If => {
                self.move_to_next_non_whitespace();
                Some(self.parse_if())
            }
            Token::Namespace => {
                self.move_to_next_non_whitespace();
                Some(self.parse_namespace())
            }
            Token::Import => {
                self.move_to_next_non_whitespace();
                Some(self.parse_import())
            }
            Token::Atom((start, end)) => {
                // Gross
                let name = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
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
                    Some(self.parse_call(name))
                }
                // TODO: Hack to try and let struct creation work in ambiguous contexts
                // like if. Need a better way.
                else if self.is_open_curly() && min_precedence == 0 {
                    Some(self.parse_struct_creation(name))
                } else {
                    Some(Ast::Identifier(name))
                }
            }
            Token::String((start, end)) => {
                // Gross
                let value =
                    String::from_utf8(self.source[start + 1..end - 1].as_bytes().to_vec()).unwrap();
                self.consume();
                Some(Ast::String(value))
            }
            Token::Integer((start, end)) => {
                // Gross
                let value = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
                self.consume();
                Some(Ast::IntegerLiteral(value.parse::<i64>().unwrap()))
            }
            Token::Float((start, end)) => {
                // Gross
                let value = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
                self.consume();
                Some(Ast::FloatLiteral(value.parse::<f64>().unwrap()))
            }
            Token::True => {
                self.consume();
                Some(Ast::True)
            }
            Token::False => {
                self.consume();
                Some(Ast::False)
            }
            Token::Null => {
                self.consume();
                Some(Ast::Null)
            }
            Token::Let => {
                self.consume();
                self.move_to_next_non_whitespace();
                let name = match self.current_token() {
                    Token::Atom((start, end)) => {
                        // Gross
                        String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
                    }
                    _ => panic!("Expected variable name"),
                };
                self.move_to_next_non_whitespace();
                self.expect_equal();
                self.move_to_next_non_whitespace();
                let value = self.parse_expression(0, true).unwrap();
                Some(Ast::Let(Box::new(Ast::Identifier(name)), Box::new(value)))
            }
            Token::NewLine | Token::Spaces(_) | Token::Comment(_) => {
                self.consume();
                self.parse_atom(min_precedence)
            }
            Token::OpenParen => {
                self.consume();
                let result = self.parse_expression(0, true);
                self.expect_close_paren();
                result
            }
            _ => panic!(
                "Expected atom, got {} at line {}",
                self.get_token_repr(),
                self.current_location()
            ),
        }
    }

    fn parse_function(&mut self) -> Ast {
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                self.move_to_next_non_whitespace();
                // Gross
                Some(String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap())
            }
            _ => None,
        };
        self.expect_open_paren();
        let args = self.parse_args();
        self.expect_close_paren();
        let body = self.parse_block();
        Ast::Function { name, args, body }
    }

    fn parse_struct(&mut self) -> Ast {
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
            }
            _ => panic!("Expected struct name"),
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly();
        let fields = self.parse_struct_fields();
        self.expect_close_curly();
        Ast::Struct { name, fields }
    }

    fn parse_enum(&mut self) -> Ast {
        let name = match self.current_token() {
            Token::Atom((start, end)) => {
                // Gross
                String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
            }
            _ => panic!("Expected enum name"),
        };
        self.move_to_next_non_whitespace();
        self.expect_open_curly();
        let variants = self.parse_enum_variants();
        self.expect_close_curly();
        Ast::Enum { name, variants }
    }

    fn consume(&mut self) {
        if self.is_newline() {
            self.increment_line();
        }
        self.position += 1;
    }

    fn move_to_next_atom(&mut self) {
        self.consume();
        while !self.at_end() && !self.is_atom() {
            self.consume();
        }
    }

    // TODO: Why the difference between this and skip?
    fn move_to_next_non_whitespace(&mut self) {
        self.consume();
        while !self.at_end() && (self.is_space() || self.is_comment()) {
            self.consume();
        }
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
            panic!("Expected open paren {:?}", self.get_token_repr());
        }
    }

    fn expect_comma(&mut self) {
        self.skip_whitespace();
        if self.is_comma() {
            self.consume();
        } else {
            panic!("Expected comma {:?}", self.get_token_repr());
        }
    }

    fn expect_string(&mut self) -> String {
        self.skip_whitespace();
        match self.current_token() {
            Token::String((start, end)) => {
                self.consume();
                // Gross
                String::from_utf8(self.source[start + 1..end - 1].as_bytes().to_vec()).unwrap()
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
                String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
            }
            _ => panic!("Expected atom got {:?}", self.get_token_repr()),
        }
    }

    fn expect_as(&mut self) {
        self.skip_whitespace();
        if self.is_as() {
            self.consume();
        } else {
            panic!("Expected at got {:?}", self.get_token_repr());
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
                let name = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
                self.consume();
                Ast::Identifier(name)
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
                let name = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
                self.consume();
                self.skip_spaces();
                let result = if self.is_open_curly() {
                    self.consume();
                    let fields = self.parse_struct_fields();
                    self.expect_close_curly();
                    Ast::EnumVariant { name, fields }
                } else {
                    Ast::EnumStaticVariant { name }
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
                let name = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
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
            if let Some(elem) = self.parse_expression(0, true) {
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

    fn current_token(&self) -> Token {
        if self.position >= self.tokens.len() {
            // TODO: Maybe bad idea
            Token::Never
        } else {
            self.tokens[self.position]
        }
    }

    #[allow(dead_code)]
    fn peek(&self) -> Token {
        // TODO: Handle end
        self.tokens[self.position + 1]
    }

    fn parse_namespace(&mut self) -> Ast {
        // namespaces can be names with dots
        // so beagle.core is a valid namespace

        let mut name = String::new();
        while !self.at_end() && (self.is_atom() || self.is_dot()) {
            match self.current_token() {
                Token::Dot => {
                    name.push('.');
                }
                Token::Atom((start, end)) => {
                    name.push_str(
                        &String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap(),
                    );
                }
                _ => panic!("Expected atom"),
            }
            self.consume();
        }
        self.consume();
        Ast::Namespace { name }
    }

    fn parse_import(&mut self) -> Ast {
        let library_name = self.expect_string();
        self.expect_as();
        let alias = Box::new(Ast::Identifier(self.expect_atom()));
        Ast::Import {
            library_name,
            alias,
        }
    }

    fn parse_call(&mut self, name: String) -> Ast {
        self.expect_open_paren();
        let mut args = Vec::new();
        while !self.at_end() && !self.is_close_paren() {
            if let Some(arg) = self.parse_expression(0, true) {
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
        Ast::Call { name, args }
    }

    fn parse_struct_creation(&mut self, name: String) -> Ast {
        self.expect_open_curly();
        let fields = self.parse_struct_fields_creations();

        self.expect_close_curly();
        Ast::StructCreation { name, fields }
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
                let name = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
                self.consume();
                self.skip_spaces();
                self.expect_colon();
                self.skip_spaces();
                let value = self.parse_expression(0, false).unwrap();
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
                String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
            }
            Token::String((start, end)) => {
                String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
            }
            Token::Integer((start, end)) => {
                String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
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

    // TODO: If I'm not going to have null
    // how do I have an empty else?
    // or do I require an else block?
    // If the if were a statement, then it would be fine
    // but it's an expression

    fn parse_if(&mut self) -> Ast {
        let condition = Box::new(self.parse_expression(1, true).unwrap());
        let then = self.parse_block();
        self.move_to_next_non_whitespace();
        if self.is_else() {
            self.consume();
            self.skip_whitespace();
            let else_ = self.parse_block();
            Ast::If {
                condition,
                then,
                else_,
            }
        } else {
            Ast::If {
                condition,
                then,
                else_: Vec::new(),
            }
        }
    }

    fn is_else(&self) -> bool {
        match self.current_token() {
            Token::Else => true,
            _ => false,
        }
    }

    fn compose_binary_op(
        &mut self,
        lhs: Ast,
        current_token: Token,
        rhs: Ast,
        min_precedence: usize,
    ) -> Ast {
        match current_token {
            Token::LessThanOrEqual => Ast::Condition {
                operator: crate::ir::Condition::LessThanOrEqual,
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::LessThan => Ast::Condition {
                operator: crate::ir::Condition::LessThan,
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::EqualEqual => Ast::Condition {
                operator: crate::ir::Condition::Equal,
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::NotEqual => Ast::Condition {
                operator: crate::ir::Condition::NotEqual,
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::GreaterThan => Ast::Condition {
                operator: crate::ir::Condition::GreaterThan,
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::GreaterThanOrEqual => Ast::Condition {
                operator: crate::ir::Condition::GreaterThanOrEqual,
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::Plus => Ast::Add {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::Minus => Ast::Sub {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::Mul => Ast::Mul {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::Div => Ast::Div {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::ShiftLeft => Ast::ShiftLeft {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::ShiftRight => Ast::ShiftRight {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::ShiftRightZero => Ast::ShiftRightZero {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::BitWiseAnd => Ast::BitWiseAnd {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::BitWiseOr => Ast::BitWiseOr {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::BitWiseXor => Ast::BitWiseXor {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::Or => Ast::Or {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::And => Ast::And {
                left: Box::new(lhs),
                right: Box::new(rhs),
            },
            Token::Dot => {
                assert!(
                    matches!(rhs, Ast::Identifier(_)),
                    "Expected identifier got {:?}",
                    rhs
                );
                let rhs = match rhs {
                    Ast::Identifier(name) => Ast::Identifier(name),
                    _ => panic!("Not an identifier"),
                };

                if self.is_open_curly() && min_precedence == 0 {
                    self.expect_open_curly();
                    let fields = self.parse_struct_fields_creations();
                    self.expect_close_curly();
                    Ast::EnumCreation {
                        name: match lhs {
                            Ast::Identifier(name) => name,
                            _ => panic!("Not an identifier"),
                        },
                        variant: match rhs {
                            Ast::Identifier(name) => name,
                            _ => panic!("Not an identifier"),
                        },
                        fields,
                    }
                } else {
                    Ast::PropertyAccess {
                        object: Box::new(lhs),
                        property: Box::new(rhs),
                    }
                }
            }
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

    fn increment_line(&mut self) {
        self.current_line += 1;
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
    let result = tokenizer.parse_all(input_bytes);
    println!("{:#?}", result);
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
