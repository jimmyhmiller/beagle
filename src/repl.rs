use std::borrow::Cow;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustyline::completion::{Completer, Pair};
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{
    Cmd, ConditionalEventHandler, Config, Context, Editor, Event, EventContext, EventHandler,
    Helper, KeyCode, KeyEvent, Modifiers, RepeatCount,
};

use crate::parser::{Token, Tokenizer};
use crate::runtime::Runtime;

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------

#[allow(unused)]
const RESET: &str = "\x1b[0m";
#[allow(unused)]
const BOLD: &str = "\x1b[1m";
#[allow(unused)]
const DIM: &str = "\x1b[2m";
#[allow(unused)]
const GREEN: &str = "\x1b[32m";
#[allow(unused)]
const CYAN: &str = "\x1b[36m";
#[allow(unused)]
const MAGENTA: &str = "\x1b[35m";
#[allow(unused)]
const YELLOW: &str = "\x1b[33m";
#[allow(unused)]
const RED: &str = "\x1b[31m";
#[allow(unused)]
const GRAY: &str = "\x1b[90m";
#[allow(unused)]
const BOLD_BLUE: &str = "\x1b[1;34m";
#[allow(unused)]
const BOLD_MAGENTA: &str = "\x1b[1;35m";

// ---------------------------------------------------------------------------
// BeagleHelper — holds a snapshot of completions + implements all four traits
// ---------------------------------------------------------------------------

struct BeagleHelper {
    /// Snapshot of completable names, refreshed each eval cycle.
    completions: Vec<String>,
    /// Struct names for completion
    struct_names: Vec<String>,
    /// Enum name -> variant names
    enum_variants: Vec<(String, Vec<String>)>,
    /// All namespace names
    namespace_names: Vec<String>,
    /// Current prompt width (for continuation line display)
    prompt_width: usize,
}

impl BeagleHelper {
    fn new() -> Self {
        Self {
            completions: Vec::new(),
            struct_names: Vec::new(),
            enum_variants: Vec::new(),
            namespace_names: Vec::new(),
            prompt_width: 6, // "user> ".len()
        }
    }

    /// Refresh completion data from the runtime.
    fn refresh(&mut self, runtime: &Runtime) {
        // Visible function names (short names from current namespace + aliased namespaces)
        self.completions = runtime.visible_function_names();

        // Add language keywords
        self.completions
            .extend(KEYWORDS.iter().map(|s| s.to_string()));

        self.struct_names = runtime.all_struct_names();
        self.enum_variants = runtime.all_enum_names_and_variants();
        self.namespace_names = runtime.all_namespace_names();
    }
}

// Language keywords for completion + highlighting
const KEYWORDS: &[&str] = &[
    "fn",
    "let",
    "if",
    "else",
    "match",
    "struct",
    "enum",
    "namespace",
    "try",
    "catch",
    "throw",
    "for",
    "in",
    "loop",
    "while",
    "break",
    "continue",
    "return",
    "true",
    "false",
    "null",
    "infinity",
    "mut",
    "use",
    "as",
    "with",
    "extend",
    "protocol",
    "dynamic",
    "binding",
    "reset",
    "shift",
    "perform",
    "handle",
    "future",
    "test",
];

// REPL commands
const REPL_COMMANDS: &[(&str, &str)] = &[
    (":help", "List available commands"),
    (":doc", "Show docstring for a function"),
    (
        ":type",
        "Show type info (arity, args, variadic, source location)",
    ),
    (":source", "Show source file and line"),
    (":ns", "Show or switch current namespace"),
    (":ls", "List bindings in current namespace"),
    (":apropos", "Search functions by name"),
    (":clear", "Clear screen"),
    (":save", "Save session history to file"),
    (":load", "Load and execute a .bg file"),
    (":quit", "Exit the REPL"),
];

// ---------------------------------------------------------------------------
// Highlighter — syntax coloring using the tokenizer
// ---------------------------------------------------------------------------

/// Syntax-highlight a string of Beagle code using the tokenizer.
fn highlight_syntax(line: &str) -> String {
    let input_bytes = line.as_bytes();
    let mut tokenizer = Tokenizer::new();
    let spans = tokenizer.tokenize_with_spans(input_bytes);

    if spans.is_empty() {
        return line.to_string();
    }

    let mut out = String::with_capacity(line.len() + 128);
    let mut last_end = 0;

    for (token, start, end) in &spans {
        if *start > last_end {
            out.push_str(&line[last_end..*start]);
        }
        let text = &line[*start..*end];
        let colored = match token {
            // Keywords — bold blue
            Token::Fn
            | Token::Let
            | Token::If
            | Token::Else
            | Token::Match
            | Token::Struct
            | Token::Enum
            | Token::Namespace
            | Token::Try
            | Token::Catch
            | Token::Throw
            | Token::For
            | Token::In
            | Token::Loop
            | Token::While
            | Token::Break
            | Token::Continue
            | Token::Return
            | Token::Mut
            | Token::Use
            | Token::As
            | Token::With
            | Token::Extend
            | Token::Protocol
            | Token::Dynamic
            | Token::Binding
            | Token::Reset
            | Token::Shift
            | Token::Perform
            | Token::Handle
            | Token::Future
            | Token::Test
            | Token::Underscore => {
                format!("{BOLD_BLUE}{text}{RESET}")
            }

            // Literals — bold magenta
            Token::True
            | Token::False
            | Token::Null
            | Token::Infinity
            | Token::NegativeInfinity
            | Token::Never => {
                format!("{BOLD_MAGENTA}{text}{RESET}")
            }

            // Strings — green
            Token::String(_) | Token::InterpolatedString(_) => {
                format!("{GREEN}{text}{RESET}")
            }

            // Numbers — cyan
            Token::Integer(_) | Token::Float(_) => {
                format!("{CYAN}{text}{RESET}")
            }

            // Comments — dim gray
            Token::Comment(_) | Token::DocComment(_) => {
                format!("{DIM}{GRAY}{text}{RESET}")
            }

            // Operators — yellow
            Token::Plus
            | Token::Minus
            | Token::Mul
            | Token::Div
            | Token::Modulo
            | Token::Equal
            | Token::EqualEqual
            | Token::NotEqual
            | Token::LessThan
            | Token::LessThanOrEqual
            | Token::GreaterThan
            | Token::GreaterThanOrEqual
            | Token::And
            | Token::Or
            | Token::Not
            | Token::Concat
            | Token::Pipe
            | Token::PipeLast
            | Token::Arrow
            | Token::BitWiseAnd
            | Token::BitWiseOr
            | Token::BitWiseXor
            | Token::ShiftLeft
            | Token::ShiftRight
            | Token::ShiftRightZero => {
                format!("{YELLOW}{text}{RESET}")
            }

            // Keywords/atoms (:foo) — magenta
            Token::Keyword(_) => {
                format!("{MAGENTA}{text}{RESET}")
            }

            // Everything else (identifiers, brackets, whitespace) — default
            _ => text.to_string(),
        };
        out.push_str(&colored);
        last_end = *end;
    }

    if last_end < line.len() {
        out.push_str(&line[last_end..]);
    }

    out
}

impl Highlighter for BeagleHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if line.is_empty() {
            return Cow::Borrowed(line);
        }

        // If it starts with ':', dim it as a command
        if line.starts_with(':') {
            return Cow::Owned(format!("{BOLD}{CYAN}{line}{RESET}"));
        }

        // Syntax-highlight the full input
        let highlighted = highlight_syntax(line);

        // Post-process: on continuation lines, replace leading prompt-width
        // spaces with a visual "..." indicator (same char count so cursor stays correct)
        if !line.contains('\n') || self.prompt_width == 0 {
            return Cow::Owned(highlighted);
        }

        let pw = self.prompt_width;
        let dots = "...";
        let padding = " ".repeat(pw.saturating_sub(dots.len()));
        let visual_prefix = format!("{DIM}{dots}{padding}{RESET}");
        let space_prefix = " ".repeat(pw);

        let raw_lines: Vec<&str> = line.split('\n').collect();
        let result = highlighted
            .split('\n')
            .enumerate()
            .map(|(i, l): (usize, &str)| {
                if i == 0 || !l.starts_with(&space_prefix) {
                    return l.to_string();
                }

                // Check raw content: if it's just spaces + closing brace,
                // visually dedent by moving the brace left 4 positions
                // (with trailing spaces to preserve total width for cursor).
                if i < raw_lines.len() {
                    let raw_after_prefix = &raw_lines[i][pw..];
                    let trimmed = raw_after_prefix.trim_start();
                    if matches!(trimmed, "}" | ")" | "]") && raw_after_prefix.len() > trimmed.len()
                    {
                        let extra = raw_after_prefix.len() - trimmed.len();
                        let dedent = extra.min(4);
                        let leading = " ".repeat(extra - dedent);
                        let trailing = " ".repeat(dedent);
                        return format!("{}{}{}{}", visual_prefix, leading, trimmed, trailing);
                    }
                }

                format!("{}{}", visual_prefix, &l[pw..])
            })
            .collect::<Vec<String>>()
            .join("\n");

        Cow::Owned(result)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(format!("{BOLD}{GREEN}{prompt}{RESET}"))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        // Return true to trigger re-highlight on every keystroke
        true
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("{DIM}{hint}{RESET}"))
    }
}

// ---------------------------------------------------------------------------
// Completer — tab completion
// ---------------------------------------------------------------------------

impl Completer for BeagleHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (start, word) = find_word_before(line, pos);

        if word.is_empty() {
            return Ok((pos, Vec::new()));
        }

        // REPL commands
        if word.starts_with(':') {
            let matches: Vec<Pair> = REPL_COMMANDS
                .iter()
                .filter(|(cmd, _)| cmd.starts_with(word))
                .map(|(cmd, _desc)| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect();
            return Ok((start, matches));
        }

        // Enum.Variant completion (e.g., "Option.")
        if let Some(dot_pos) = word.find('.') {
            let enum_name = &word[..dot_pos];
            let variant_prefix = &word[dot_pos + 1..];
            for (ename, variants) in &self.enum_variants {
                let short_name = ename.rsplit('/').next().unwrap_or(ename);
                if short_name == enum_name {
                    let matches: Vec<Pair> = variants
                        .iter()
                        .filter(|v| v.starts_with(variant_prefix))
                        .map(|v| Pair {
                            display: format!("{}.{}", enum_name, v),
                            replacement: format!("{}.{}", enum_name, v),
                        })
                        .collect();
                    return Ok((start, matches));
                }
            }
        }

        // Namespace-qualified completion (e.g., "core/ma")
        if let Some(slash_pos) = word.find('/') {
            let ns_prefix = &word[..slash_pos];
            let fn_prefix = &word[slash_pos + 1..];
            let matches: Vec<Pair> = self
                .completions
                .iter()
                .filter(|name| {
                    if let Some(sp) = name.find('/') {
                        let ns_part = &name[..sp];
                        let fn_part = &name[sp + 1..];
                        ns_part == ns_prefix && fn_part.starts_with(fn_prefix)
                    } else {
                        false
                    }
                })
                .map(|name| Pair {
                    display: name.clone(),
                    replacement: name.clone(),
                })
                .collect();
            return Ok((start, matches));
        }

        // Unqualified name completion
        let mut matches: Vec<Pair> = Vec::new();

        // Functions and keywords
        for name in &self.completions {
            let short = name.rsplit('/').next().unwrap_or(name);
            if short.starts_with(word) {
                matches.push(Pair {
                    display: short.to_string(),
                    replacement: short.to_string(),
                });
            }
        }

        // Struct names
        for sname in &self.struct_names {
            let short = sname.rsplit('/').next().unwrap_or(sname);
            if short.starts_with(word) {
                matches.push(Pair {
                    display: short.to_string(),
                    replacement: short.to_string(),
                });
            }
        }

        // Enum names
        for (ename, _) in &self.enum_variants {
            let short = ename.rsplit('/').next().unwrap_or(ename);
            if short.starts_with(word) {
                matches.push(Pair {
                    display: short.to_string(),
                    replacement: short.to_string(),
                });
            }
        }

        // Namespace names (for typing "namespace/" patterns)
        for ns in &self.namespace_names {
            if ns.starts_with(word) {
                matches.push(Pair {
                    display: ns.clone(),
                    replacement: format!("{}/", ns),
                });
            }
        }

        // De-duplicate
        matches.sort_by(|a, b| a.display.cmp(&b.display));
        matches.dedup_by(|a, b| a.display == b.display);

        Ok((start, matches))
    }
}

/// Find the start of the word the cursor is in (for completion).
fn find_word_before(line: &str, pos: usize) -> (usize, &str) {
    let bytes = line.as_bytes();
    let mut start = pos;
    while start > 0 {
        let b = bytes[start - 1];
        if b == b' '
            || b == b'\t'
            || b == b'('
            || b == b'{'
            || b == b'['
            || b == b')'
            || b == b'}'
            || b == b']'
            || b == b','
            || b == b';'
            || b == b'\n'
        {
            break;
        }
        start -= 1;
    }
    (start, &line[start..pos])
}

// ---------------------------------------------------------------------------
// Validator — multi-line input detection
// ---------------------------------------------------------------------------

/// Check whether input is complete or needs more lines.
fn check_input_complete(input: &str) -> ValidationResult {
    // Don't try multi-line for REPL commands
    if input.starts_with(':') {
        return ValidationResult::Valid(None);
    }

    // Count unmatched brackets
    let mut parens = 0i32;
    let mut curlies = 0i32;
    let mut brackets = 0i32;
    let mut in_string = false;
    let mut in_comment = false;
    let mut prev = 0u8;

    for &b in input.as_bytes() {
        if in_comment {
            if b == b'\n' {
                in_comment = false;
            }
            prev = b;
            continue;
        }
        if in_string {
            if b == b'"' && prev != b'\\' {
                in_string = false;
            }
            prev = b;
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'/' if prev == b'/' => in_comment = true,
            b'(' => parens += 1,
            b')' => parens -= 1,
            b'{' => curlies += 1,
            b'}' => curlies -= 1,
            b'[' => brackets += 1,
            b']' => brackets -= 1,
            _ => {}
        }
        prev = b;
    }

    if parens > 0 || curlies > 0 || brackets > 0 {
        return ValidationResult::Incomplete;
    }

    // Check if line ends with a binary operator (likely continuation)
    let trimmed = input.trim_end();
    if !trimmed.is_empty() {
        let last_char = trimmed.as_bytes()[trimmed.len() - 1];
        if matches!(
            last_char,
            b'+' | b'-' | b'*' | b'/' | b'=' | b'|' | b'&' | b'^' | b'%'
        ) && !trimmed.ends_with("//")
        {
            // Check it's not a complete operator like `||` at end of valid expression
            // Only treat as incomplete if it's clearly a dangling binary op
            if !trimmed.ends_with("||") && !trimmed.ends_with("&&") {
                return ValidationResult::Incomplete;
            }
        }
    }

    // Unclosed string
    if in_string {
        return ValidationResult::Incomplete;
    }

    ValidationResult::Valid(None)
}

impl Validator for BeagleHelper {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        Ok(check_input_complete(ctx.input()))
    }
}

// ---------------------------------------------------------------------------
// Hinter — function signatures + history suggestions
// ---------------------------------------------------------------------------

impl Hinter for BeagleHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        None
    }
}

impl Helper for BeagleHelper {}

// ---------------------------------------------------------------------------
// REPL command handlers
// ---------------------------------------------------------------------------

fn handle_command(runtime: &mut Runtime, input: &str) -> bool {
    let parts: Vec<&str> = input.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        ":help" => {
            println!("{BOLD}Available commands:{RESET}");
            for (cmd, desc) in REPL_COMMANDS {
                println!("  {CYAN}{:<16}{RESET} {}", cmd, desc);
            }
            println!();
            println!("  Evaluate any Beagle expression by typing it directly.");
        }
        ":quit" | ":q" | ":exit" => {
            return true; // signal exit
        }
        ":clear" => {
            print!("\x1b[2J\x1b[H");
        }
        ":ns" => {
            if arg.is_empty() {
                println!("{}", runtime.current_namespace_name());
            } else {
                let ns_id = runtime
                    .get_namespace_id(arg)
                    .unwrap_or_else(|| runtime.reserve_namespace(arg.to_string()));
                runtime.set_current_namespace(ns_id);
                println!("Switched to namespace: {}", arg);
            }
        }
        ":ls" => {
            let names = runtime.visible_function_names();
            let mut short_names: Vec<String> =
                names.iter().filter(|n| !n.contains('/')).cloned().collect();
            short_names.sort();
            for name in &short_names {
                println!("  {}", name);
            }
            if short_names.is_empty() {
                println!("  (no bindings in current namespace)");
            }
        }
        ":doc" => {
            if arg.is_empty() {
                println!("Usage: :doc <function-name>");
                return false;
            }
            let full_name = resolve_name(runtime, arg);
            match runtime.get_function_by_name(&full_name) {
                Some(f) => {
                    if let Some(ref doc) = f.docstring {
                        println!("{BOLD}{}{RESET}", f.name);
                        println!("{}", doc);
                    } else {
                        println!("{BOLD}{}{RESET}", f.name);
                        println!("{DIM}(no docstring){RESET}");
                    }
                }
                None => {
                    println!("Not found: {}", arg);
                }
            }
        }
        ":type" => {
            if arg.is_empty() {
                println!("Usage: :type <function-name>");
                return false;
            }
            let full_name = resolve_name(runtime, arg);
            match runtime.get_function_by_name(&full_name) {
                Some(f) => {
                    let args_str = if f.arg_names.is_empty() {
                        format!("{} args", f.number_of_args)
                    } else {
                        f.arg_names.join(", ")
                    };
                    println!("{BOLD}{}{RESET}", f.name);
                    println!("  args:     ({})", args_str);
                    println!("  arity:    {}", f.number_of_args);
                    if f.is_variadic {
                        println!("  variadic: yes (min {})", f.min_args);
                    }
                    if f.is_builtin {
                        println!("  builtin:  yes");
                    }
                    if let Some(ref file) = f.source_file {
                        let loc = match f.source_line {
                            Some(line) => format!("{}:{}", file, line),
                            None => file.clone(),
                        };
                        println!("  source:   {}", loc);
                    }
                }
                None => {
                    println!("Not found: {}", arg);
                }
            }
        }
        ":source" => {
            if arg.is_empty() {
                println!("Usage: :source <function-name>");
                return false;
            }
            let full_name = resolve_name(runtime, arg);
            match runtime.get_function_by_name(&full_name) {
                Some(f) => match (&f.source_file, f.source_line) {
                    (Some(file), Some(line)) => println!("{}:{}", file, line),
                    (Some(file), None) => println!("{}", file),
                    _ => println!("{DIM}(no source location){RESET}"),
                },
                None => {
                    println!("Not found: {}", arg);
                }
            }
        }
        ":apropos" => {
            if arg.is_empty() {
                println!("Usage: :apropos <query>");
                return false;
            }
            let query = arg.to_lowercase();
            let mut found = 0;
            for f in &runtime.functions {
                if !f.is_defined {
                    continue;
                }
                if f.name.to_lowercase().contains(&query) {
                    let doc = f
                        .docstring
                        .as_deref()
                        .map(|d| {
                            let first_line = d.lines().next().unwrap_or("");
                            if first_line.len() > 60 {
                                format!("{}...", &first_line[..57])
                            } else {
                                first_line.to_string()
                            }
                        })
                        .unwrap_or_default();
                    println!("  {:<40} {DIM}{}{RESET}", f.name, doc);
                    found += 1;
                }
            }
            if found == 0 {
                println!("  No matches for: {}", arg);
            }
        }
        ":save" => {
            if arg.is_empty() {
                println!("Usage: :save <filename>");
            } else {
                println!("History save not yet implemented for file: {}", arg);
            }
        }
        ":load" => {
            if arg.is_empty() {
                println!("Usage: :load <file.bg>");
                return false;
            }
            match std::fs::read_to_string(arg) {
                Ok(source) => {
                    let escaped = source.replace("\\", "\\\\").replace("\"", "\\\"");
                    let wrapped = format!(
                        "try {{ repr(eval(\"{}\")) }} catch (__repl_error__) {{ \"__REPL_ERR__\" ++ to-string(__repl_error__) }}",
                        escaped
                    );
                    match runtime.compile_string(&wrapped) {
                        Ok(fn_ptr) => {
                            if fn_ptr != 0 {
                                let result = runtime.call_via_trampoline(fn_ptr);
                                let s = runtime.get_string(0, result);
                                if let Some(err_msg) = s.strip_prefix("__REPL_ERR__") {
                                    eprintln!("{RED}Error:{RESET} {}", err_msg);
                                } else if s != "null" {
                                    println!("{BOLD}=>{RESET} {}", s);
                                }
                            }
                        }
                        Err(e) => eprintln!("{RED}Error:{RESET} {}", e),
                    }
                }
                Err(e) => eprintln!("{RED}Error:{RESET} Could not read {}: {}", arg, e),
            }
        }
        _ => {
            println!(
                "Unknown command: {}. Type :help for available commands.",
                cmd
            );
        }
    }

    false // don't exit
}

/// Resolve a short function name to its fully-qualified form.
fn resolve_name(runtime: &Runtime, name: &str) -> String {
    if name.contains('/') {
        // Already qualified
        return name.to_string();
    }
    let ns = runtime.current_namespace_name();
    let full = format!("{}/{}", ns, name);
    if runtime.get_function_by_name(&full).is_some() {
        return full;
    }
    // Try core namespace
    let core = format!("beagle.core/{}", name);
    if runtime.get_function_by_name(&core).is_some() {
        return core;
    }
    // Try beagle.collections
    let coll = format!("beagle.collections/{}", name);
    if runtime.get_function_by_name(&coll).is_some() {
        return coll;
    }
    // Return the user/name form as fallback
    full
}

// ---------------------------------------------------------------------------
// History file path
// ---------------------------------------------------------------------------

fn history_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let dir = std::path::PathBuf::from(home).join(".beagle");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("history")
}

// ---------------------------------------------------------------------------
// Welcome banner
// ---------------------------------------------------------------------------

fn print_banner() {
    let version = env!("CARGO_PKG_VERSION");

    cfg_if::cfg_if! {
        if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
            let arch = "x86-64";
        } else {
            let arch = "ARM64";
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(feature = "compacting")] {
            let gc = "compacting GC";
        } else if #[cfg(feature = "mark-and-sweep")] {
            let gc = "mark-and-sweep GC";
        } else if #[cfg(feature = "generational")] {
            let gc = "generational GC";
        } else {
            let gc = "generational GC";
        }
    }

    println!("{BOLD}Beagle REPL{RESET} v{} ({}, {})", version, arch, gc);
    println!("Type {CYAN}:help{RESET} for commands, {CYAN}Ctrl-D{RESET} to exit.");
}

// ---------------------------------------------------------------------------
// Auto-indent handler — inserts newline + indentation on Enter when incomplete
// ---------------------------------------------------------------------------

/// Compute brace/paren/bracket depth, respecting strings and comments.
fn brace_depth(input: &str) -> i32 {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut in_comment = false;
    let mut prev = 0u8;

    for &b in input.as_bytes() {
        if in_comment {
            if b == b'\n' {
                in_comment = false;
            }
            prev = b;
            continue;
        }
        if in_string {
            if b == b'"' && prev != b'\\' {
                in_string = false;
            }
            prev = b;
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'/' if prev == b'/' => in_comment = true,
            b'(' | b'{' | b'[' => depth += 1,
            b')' | b'}' | b']' => depth -= 1,
            _ => {}
        }
        prev = b;
    }
    depth
}

struct AutoIndentHandler {
    prompt_width: Arc<AtomicUsize>,
}

impl ConditionalEventHandler for AutoIndentHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: RepeatCount,
        _positive: bool,
        ctx: &EventContext,
    ) -> Option<Cmd> {
        let input = ctx.line();
        let depth = brace_depth(input);

        if depth > 0 {
            // Incomplete — insert newline with:
            // - prompt_width spaces (aligns with first line's code start)
            // - depth * 4 spaces (nesting indentation)
            let pw = self.prompt_width.load(Ordering::Relaxed);
            let indent = " ".repeat(pw) + &"    ".repeat(depth as usize);
            Some(Cmd::Insert(1, format!("\n{}", indent)))
        } else {
            // Balanced — let default AcceptLine run (Validator still gets a say)
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Main interactive loop
// ---------------------------------------------------------------------------

pub fn run_interactive_loop(runtime: &mut Runtime) -> Result<(), Box<dyn Error>> {
    print_banner();

    let config = Config::builder()
        .auto_add_history(true)
        .max_history_size(10_000)
        .expect("valid history size")
        .build();

    let prompt_width = Arc::new(AtomicUsize::new(6)); // "user> ".len()

    let helper = BeagleHelper::new();
    let mut rl: Editor<BeagleHelper, rustyline::history::DefaultHistory> =
        Editor::with_config(config)?;
    rl.set_helper(Some(helper));

    // Bind Enter to auto-indent handler
    rl.bind_sequence(
        KeyEvent(KeyCode::Enter, Modifiers::NONE),
        EventHandler::Conditional(Box::new(AutoIndentHandler {
            prompt_width: prompt_width.clone(),
        })),
    );

    // Load history
    let hist = history_path();
    let _ = rl.load_history(&hist);

    // Initial refresh of completions
    rl.helper_mut().unwrap().refresh(runtime);

    loop {
        let prompt = format!("{}> ", runtime.current_namespace_name());
        prompt_width.store(prompt.len(), Ordering::Relaxed);
        rl.helper_mut().unwrap().prompt_width = prompt.len();

        match rl.readline(&prompt) {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                // REPL commands
                if input.starts_with(':') {
                    let should_exit = handle_command(runtime, input);
                    // Refresh completions after commands that may change state
                    rl.helper_mut().unwrap().refresh(runtime);
                    if should_exit {
                        break;
                    }
                    continue;
                }

                // Evaluate Beagle expression
                // Wrap in eval() + repr() so we always get a string back,
                // and try/catch so errors are displayed nicely.
                let escaped_input = input.replace("\\", "\\\\").replace("\"", "\\\"");
                let wrapped_input = format!(
                    "try {{ repr(eval(\"{}\")) }} catch (__repl_error__) {{ \"__REPL_ERR__\" ++ to-string(__repl_error__) }}",
                    escaped_input
                );

                match runtime.compile_string(&wrapped_input) {
                    Ok(function_pointer) => {
                        if function_pointer == 0 {
                            continue;
                        }
                        let result = runtime.call_via_trampoline(function_pointer);
                        let result_str = runtime.get_string(0, result);
                        if let Some(err_msg) = result_str.strip_prefix("__REPL_ERR__") {
                            eprintln!("{RED}Error:{RESET} {}", err_msg);
                        } else if result_str != "null" {
                            println!("{BOLD}=>{RESET} {}", result_str);
                        }
                    }
                    Err(e) => {
                        eprintln!("{RED}Error:{RESET} {}", e);
                    }
                }

                // Refresh completions after eval (new bindings may have been defined)
                rl.helper_mut().unwrap().refresh(runtime);
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                break;
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(e) => {
                eprintln!("{RED}Error:{RESET} {}", e);
                break;
            }
        }
    }

    // Save history
    let _ = rl.save_history(&hist);

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::highlight::Highlighter;

    // -- brace_depth --

    #[test]
    fn brace_depth_empty() {
        assert_eq!(brace_depth(""), 0);
    }

    #[test]
    fn brace_depth_simple_open() {
        assert_eq!(brace_depth("if true {"), 1);
    }

    #[test]
    fn brace_depth_nested() {
        assert_eq!(brace_depth("fn foo() { if true {"), 2);
    }

    #[test]
    fn brace_depth_balanced() {
        assert_eq!(brace_depth("fn foo() { 1 }"), 0);
    }

    #[test]
    fn brace_depth_mixed_brackets() {
        assert_eq!(brace_depth("foo([{"), 3);
        assert_eq!(brace_depth("foo([{}])"), 0);
    }

    #[test]
    fn brace_depth_ignores_string_contents() {
        assert_eq!(brace_depth("\"hello { world\""), 0);
    }

    #[test]
    fn brace_depth_ignores_comments() {
        assert_eq!(brace_depth("foo // { not counted"), 0);
    }

    #[test]
    fn brace_depth_comment_ends_at_newline() {
        assert_eq!(brace_depth("// comment\n{"), 1);
    }

    #[test]
    fn brace_depth_escaped_quote_in_string() {
        // The \" inside the string shouldn't end it
        assert_eq!(brace_depth(r#"foo("hello \" world")"#), 0);
    }

    // -- highlight_syntax --

    #[test]
    fn highlight_syntax_empty() {
        assert_eq!(highlight_syntax(""), "");
    }

    #[test]
    fn highlight_syntax_keyword_colored() {
        let result = highlight_syntax("fn");
        assert!(
            result.contains(BOLD_BLUE),
            "keyword 'fn' should be bold blue"
        );
        assert!(result.contains("fn"), "should contain the keyword text");
        assert!(result.contains(RESET), "should reset after keyword");
    }

    #[test]
    fn highlight_syntax_number_colored() {
        let result = highlight_syntax("42");
        assert!(result.contains(CYAN), "number should be cyan");
        assert!(result.contains("42"));
    }

    #[test]
    fn highlight_syntax_string_colored() {
        let result = highlight_syntax("\"hello\"");
        assert!(result.contains(GREEN), "string should be green");
    }

    #[test]
    fn highlight_syntax_preserves_plain_text() {
        // An identifier should come through without color codes
        let result = highlight_syntax("foo");
        // It should just be "foo" with no ANSI escapes (identifiers are default color)
        assert_eq!(result, "foo");
    }

    // -- Highlighter visual dedent --

    fn make_helper(prompt_width: usize) -> BeagleHelper {
        let mut h = BeagleHelper::new();
        h.prompt_width = prompt_width;
        h
    }

    #[test]
    fn highlighter_single_line_no_change() {
        let h = make_helper(6);
        let result = h.highlight("fn foo() {", 0);
        // Should just be syntax-highlighted, no continuation processing
        assert!(
            !result.contains("..."),
            "single line should not get continuation prefix"
        );
    }

    #[test]
    fn highlighter_continuation_line_gets_dots() {
        let h = make_helper(6);
        // Multi-line input: first line is code, second line has prompt-width leading spaces
        let input = "if true {\n      4";
        let result = h.highlight(input, 0);
        // The continuation line should have "..." prefix (dimmed)
        assert!(
            result.contains("..."),
            "continuation line should show dots prefix"
        );
    }

    #[test]
    fn highlighter_visual_dedent_closing_brace() {
        let h = make_helper(6);
        // Simulate: user typed "if true {" on line 1,
        // then on continuation line: 6 spaces (prompt) + 4 spaces (indent) + "}"
        // The visual dedent should move "}" 4 positions left
        let input = "if true {\n          }";
        //                       ^^^^^^ 6 prompt  ^^^^ 4 indent + "}"
        let result = h.highlight(input, 0);
        let result_str = result.to_string();

        // The continuation line should NOT have 4 leading spaces before the }
        // Instead it should have trailing spaces to preserve width
        let lines: Vec<&str> = result_str.split('\n').collect();
        assert!(lines.len() == 2, "should have 2 lines");

        // The second line should end with trailing spaces (dedent compensation)
        let second = lines[1];
        // Strip ANSI codes to check structure
        let stripped = strip_ansi(second);
        // After visual prefix ("..." + pad), should be "}" followed by spaces
        assert!(
            stripped.contains("}"),
            "should still contain the closing brace"
        );
        // The "}" should appear earlier (less leading space) than in raw input
        let brace_pos_raw = "          }".find('}').unwrap(); // position 10
        let brace_pos_display = stripped.find('}').unwrap();
        assert!(
            brace_pos_display < brace_pos_raw,
            "brace should be visually shifted left: display={} raw={}",
            brace_pos_display,
            brace_pos_raw
        );
    }

    #[test]
    fn highlighter_visual_dedent_preserves_width() {
        let h = make_helper(6);
        // 6 prompt spaces + 4 indent spaces + "}"
        let input = "if true {\n          }";
        let result = h.highlight(input, 0);
        let result_str = result.to_string();
        let lines: Vec<&str> = result_str.split('\n').collect();
        let second_stripped = strip_ansi(lines[1]);
        let raw_second = &"          }"[..]; // what raw second line looks like

        // Total visible width should be preserved (same number of visible chars)
        assert_eq!(
            second_stripped.len(),
            raw_second.len(),
            "visual dedent must preserve total character width: got '{}' (len={}) vs raw '{}' (len={})",
            second_stripped,
            second_stripped.len(),
            raw_second,
            raw_second.len()
        );
    }

    #[test]
    fn highlighter_no_dedent_for_content_after_brace() {
        let h = make_helper(6);
        // A line with content after "}" should NOT be dedented
        let input = "if true {\n          } else {";
        let result = h.highlight(input, 0);
        let result_str = result.to_string();
        let lines: Vec<&str> = result_str.split('\n').collect();
        let second_stripped = strip_ansi(lines[1]);
        // Should NOT have trailing space compensation — it's not "just a closing brace"
        // The continuation prefix replaces the prompt-width spaces but the rest is normal
        assert!(
            second_stripped.contains("} else"),
            "should preserve full line content"
        );
    }

    #[test]
    fn highlighter_dedent_paren() {
        let h = make_helper(6);
        let input = "foo(\n          )";
        let result = h.highlight(input, 0);
        let result_str = result.to_string();
        let lines: Vec<&str> = result_str.split('\n').collect();
        let second_stripped = strip_ansi(lines[1]);
        let paren_pos_raw = "          )".find(')').unwrap();
        let paren_pos_display = second_stripped.find(')').unwrap();
        assert!(
            paren_pos_display < paren_pos_raw,
            "closing paren should be visually dedented"
        );
    }

    #[test]
    fn highlighter_dedent_bracket() {
        let h = make_helper(6);
        let input = "foo[\n          ]";
        let result = h.highlight(input, 0);
        let result_str = result.to_string();
        let lines: Vec<&str> = result_str.split('\n').collect();
        let second_stripped = strip_ansi(lines[1]);
        let bracket_pos_raw = "          ]".find(']').unwrap();
        let bracket_pos_display = second_stripped.find(']').unwrap();
        assert!(
            bracket_pos_display < bracket_pos_raw,
            "closing bracket should be visually dedented"
        );
    }

    // -- Validator (check_input_complete) --

    #[test]
    fn validator_balanced_is_valid() {
        let result = check_input_complete("fn foo() { 1 }");
        assert!(matches!(result, ValidationResult::Valid(_)));
    }

    #[test]
    fn validator_unclosed_brace_is_incomplete() {
        let result = check_input_complete("if true {");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn validator_unclosed_paren_is_incomplete() {
        let result = check_input_complete("foo(1, 2");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn validator_dangling_operator_is_incomplete() {
        let result = check_input_complete("1 +");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn validator_unclosed_string_is_incomplete() {
        let result = check_input_complete("\"hello");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn validator_command_is_always_valid() {
        let result = check_input_complete(":help");
        assert!(matches!(result, ValidationResult::Valid(_)));
    }

    // -- find_word_before --

    #[test]
    fn find_word_at_end() {
        let (start, word) = find_word_before("map", 3);
        assert_eq!(start, 0);
        assert_eq!(word, "map");
    }

    #[test]
    fn find_word_after_space() {
        let (start, word) = find_word_before("foo bar", 7);
        assert_eq!(start, 4);
        assert_eq!(word, "bar");
    }

    #[test]
    fn find_word_after_paren() {
        let (start, word) = find_word_before("foo(bar", 7);
        assert_eq!(start, 4);
        assert_eq!(word, "bar");
    }

    #[test]
    fn find_word_empty_at_space() {
        let (start, word) = find_word_before("foo ", 4);
        assert_eq!(start, 4);
        assert_eq!(word, "");
    }

    // -- Helpers --

    /// Strip ANSI escape codes from a string for testing visible content.
    fn strip_ansi(s: &str) -> String {
        let mut result = String::new();
        let mut in_escape = false;
        for ch in s.chars() {
            if ch == '\x1b' {
                in_escape = true;
                continue;
            }
            if in_escape {
                if ch == 'm' {
                    in_escape = false;
                }
                continue;
            }
            result.push(ch);
        }
        result
    }
}
