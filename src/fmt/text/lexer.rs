//! Lexer to tokenize WAT.

use core::{fmt, str::FromStr};

const HORIZONTAL_TABULATION: char = '\u{09}';
const LINE_FEED: char = '\u{0A}';
const CARRIAGE_RETURN: char = '\u{0D}';

#[inline]
#[must_use]
const fn is_space(ch: char) -> bool {
    matches!(ch, ' ' | HORIZONTAL_TABULATION)
}

/// Returns true if the byte is a newline character.
#[inline]
#[must_use]
const fn is_newline(ch: char) -> bool {
    matches!(ch, LINE_FEED | CARRIAGE_RETURN)
}

#[must_use]
fn skip_space(s: &str, skip_new_lines: bool) -> usize {
    let mut char_indices = s.char_indices().peekable();

    'outer: while let Some((pos, ch)) = char_indices.next() {
        if is_space(ch) || is_newline(ch) {
            continue;
        }

        if is_newline(ch) {
            if skip_new_lines {
                continue;
            }

            return pos;
        }

        match ch {
            ';' => {
                let Some((_, ch)) = char_indices.peek() else {
                    return pos;
                };
                if *ch != ';' {
                    return pos;
                }
                let _ = char_indices.next();

                // Skip line comment
                for (_, ch) in char_indices.by_ref() {
                    if ch == LINE_FEED {
                        continue 'outer;
                    }
                }
                return s.len();
            }
            '(' => {
                let Some((_, ch)) = char_indices.peek() else {
                    return pos;
                };
                if *ch != ';' {
                    return pos;
                }
                let _ = char_indices.next();

                // Skip block comment
                let mut end_delim_count = 1usize;
                while let Some((_, ch)) = char_indices.next() {
                    match ch {
                        '(' => {
                            if let Some((_, ch)) = char_indices.peek() {
                                if *ch == ';' {
                                    let _ = char_indices.next();
                                    end_delim_count += 1;
                                }
                            }
                        }
                        ';' => {
                            if let Some((_, ch)) = char_indices.peek() {
                                if *ch == ')' {
                                    let _ = char_indices.next();
                                    end_delim_count -= 1;
                                    if end_delim_count == 0 {
                                        continue 'outer;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                return s.len();
            }
            _ => return pos,
        }
    }

    s.len()
}

#[inline]
#[must_use]
const fn is_id_char(ch: char) -> bool {
    matches!(ch,
      '0'..='9'
      | 'A'..='Z'
      | 'a'..='z'
      | '!' | '#' | '$' | '%' | '&' | '\'' | '*' | '+' | '-' | '.' | '/'
      | ':' | '<' | '=' | '>' | '?' | '@' | '\\' | '^' | '_' | '`' | '|' | '~'
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    /// Infinity
    Inf,
    /// Not a number
    Nan,
    I32,
    I64,
    F32,
    F64,
    V128,
    FuncRef,
    ExternRef,
    Func,
    Extern,
    Param,
    Result,
    Mut,
    Block,
    End,
    Loop,
    If,
    Else,
    Unreachable,
    Nop,
    Br,
    BrIf,
    BrTable,
    Return,
    Call,
    CallIndirect,
    RefNull,
    RefIsNull,
    RefFunc,
    Drop,
    Select,
    LocalGet,
    LocalSet,
    LocalTee,
    GlobalGet,
    GlobalSet,
    TableGet,
    TableSet,
    TableSize,
    TableGrow,
    TableFill,
    TableCopy,
    TableInit,
    ElemDrop,
    I32Const,
    Import,
    Export,
    Module,
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for Keyword {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Keyword {
    fn as_str(&self) -> &str {
        match self {
            Keyword::Inf => "inf",
            Keyword::Nan => "nan",
            Keyword::I32 => "i32",
            Keyword::I64 => "i64",
            Keyword::F32 => "f32",
            Keyword::F64 => "f64",
            Keyword::V128 => "v128",
            Keyword::FuncRef => "funcref",
            Keyword::ExternRef => "externref",
            Keyword::Func => "func",
            Keyword::Extern => "extern",
            Keyword::Param => "param",
            Keyword::Result => "result",
            Keyword::Mut => "mut",
            Keyword::Block => "block",
            Keyword::End => "end",
            Keyword::Loop => "loop",
            Keyword::If => "if",
            Keyword::Else => "else",
            Keyword::Unreachable => "unreachable",
            Keyword::Nop => "nop",
            Keyword::Br => "br",
            Keyword::BrIf => "br_if",
            Keyword::BrTable => "br_table",
            Keyword::Return => "return",
            Keyword::Call => "call",
            Keyword::CallIndirect => "call_indirect",
            Keyword::RefNull => "ref.null",
            Keyword::RefIsNull => "ref.is_null",
            Keyword::RefFunc => "ref.func",
            Keyword::Drop => "drop",
            Keyword::Select => "select",
            Keyword::LocalGet => "local.get",
            Keyword::LocalSet => "local.set",
            Keyword::LocalTee => "local.tee",
            Keyword::GlobalGet => "global.get",
            Keyword::GlobalSet => "global.set",
            Keyword::TableGet => "table.get",
            Keyword::TableSet => "table.set",
            Keyword::TableSize => "table.size",
            Keyword::TableGrow => "table.grow",
            Keyword::TableFill => "table.fill",
            Keyword::TableCopy => "table.copy",
            Keyword::TableInit => "table.init",
            Keyword::ElemDrop => "elem.drop",
            Keyword::I32Const => "i32.const",
            Keyword::Import => "import",
            Keyword::Export => "export",
            Keyword::Module => "module",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeywordParseErr;

impl FromStr for Keyword {
    type Err = KeywordParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "inf" => Self::Inf,
            "nan" => Self::Nan,
            "i32" => Keyword::I32,
            "i64" => Keyword::I64,
            "f32" => Keyword::F32,
            "f64" => Keyword::F64,
            "v128" => Keyword::V128,
            "funcref" => Keyword::FuncRef,
            "externref" => Keyword::ExternRef,
            "func" => Keyword::Func,
            "extern" => Keyword::Extern,
            "param" => Keyword::Param,
            "result" => Keyword::Result,
            "mut" => Keyword::Mut,
            "block" => Keyword::Block,
            "end" => Keyword::End,
            "loop" => Keyword::Loop,
            "if" => Keyword::If,
            "else" => Keyword::Else,
            "unreachable" => Keyword::Unreachable,
            "nop" => Keyword::Nop,
            "br" => Keyword::Br,
            "br_if" => Keyword::BrIf,
            "br_table" => Keyword::BrTable,
            "return" => Keyword::Return,
            "call" => Keyword::Call,
            "call_indirect" => Keyword::CallIndirect,
            "ref.null" => Keyword::RefNull,
            "ref.is_null" => Keyword::RefIsNull,
            "ref.func" => Keyword::RefFunc,
            "drop" => Keyword::Drop,
            "select" => Keyword::Select,
            "local.get" => Keyword::LocalGet,
            "local.set" => Keyword::LocalSet,
            "local.tee" => Keyword::LocalTee,
            "global.get" => Keyword::GlobalGet,
            "global.set" => Keyword::GlobalSet,
            "table.get" => Keyword::TableGet,
            "table.set" => Keyword::TableSet,
            "table.size" => Keyword::TableSize,
            "table.grow" => Keyword::TableGrow,
            "table.fill" => Keyword::TableFill,
            "table.copy" => Keyword::TableCopy,
            "table.init" => Keyword::TableInit,
            "elem.drop" => Keyword::ElemDrop,
            "i32.const" => Keyword::I32Const,
            "import" => Keyword::Import,
            "export" => Keyword::Export,
            "module" => Keyword::Module,
            _ => {
                return Err(KeywordParseErr);
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenTy {
    /// End of file
    Eof,
    /// Any literal terminal in the grammar
    Keyword(Keyword),
    /// Unsigned number
    UnsignedInt,
    /// Signed number
    SignedInt,
    /// Floating point number
    FloatingPoint,
    /// String in quotation marks
    String,
    /// Identifier is composed of [`is_id_char`].
    Identifier,
    /// `(`
    OpenParen,
    /// `)`
    CloseParen,
    /// Reserved word or unknown word
    Reserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lexeme<'a> {
    pub inner: &'a str,
    pub head_noise_len: usize,
    pub tail_noise_len: usize,
}

impl<'a> Lexeme<'a> {
    #[inline]
    #[must_use]
    pub fn noiseless_str(&self) -> &'a str {
        &self.inner[self.head_noise_len..self.inner.len() - self.tail_noise_len]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    ty: TokenTy,
    lexeme: Option<Lexeme<'a>>,
}

impl<'a> fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(lexeme) = self.lexeme {
            return fmt::Display::fmt(lexeme.inner, f);
        }
        Ok(())
    }
}

impl<'a> Token<'a> {
    /// Returns true if the token type is as expected and there are lexeme bytes representing the token.
    #[inline]
    #[must_use]
    pub fn is_type_present(&self, ty: TokenTy) -> bool {
        self.ty == ty && self.lexeme.is_some()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lexeme.map_or(true, |l| l.inner.is_empty())
    }

    /// Returns the length of the token's bytes. If there are no bytes representing the token, the length is 0.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        if let Some(lexeme) = &self.lexeme {
            lexeme.inner.len()
        } else {
            0
        }
    }

    #[must_use]
    pub fn as_keyword(&self) -> Option<Keyword> {
        match self.ty {
            TokenTy::Eof
            | TokenTy::UnsignedInt
            | TokenTy::SignedInt
            | TokenTy::FloatingPoint
            | TokenTy::String
            | TokenTy::Identifier
            | TokenTy::OpenParen
            | TokenTy::CloseParen
            | TokenTy::Reserved => None,
            TokenTy::Keyword(keyword) => Some(keyword),
        }
    }

    #[must_use]
    pub fn as_noiseless_str(&self) -> Option<&str> {
        match self.ty {
            TokenTy::Eof
            | TokenTy::Keyword(_)
            | TokenTy::UnsignedInt
            | TokenTy::SignedInt
            | TokenTy::FloatingPoint
            | TokenTy::Identifier
            | TokenTy::OpenParen
            | TokenTy::CloseParen
            | TokenTy::Reserved => None,
            TokenTy::String => self
                .lexeme
                .map(|l| &l.noiseless_str()[1..l.noiseless_str().len() - 1]),
        }
    }
}

#[derive(Debug)]
struct Tokenizer<'a> {
    input: &'a str,
    is_eof: bool,
}

impl<'a> Tokenizer<'a> {
    #[inline]
    #[must_use]
    const fn new(input: &'a str) -> Self {
        Self {
            input,
            is_eof: false,
        }
    }

    fn advance(&mut self, ty: TokenTy, head_noise_len: usize, lexeme_len: usize) -> Token<'a> {
        let len = head_noise_len + lexeme_len;
        let tail_noise_len = skip_space(&self.input[len..], false);

        let len = len + tail_noise_len;
        let token = Token {
            ty,
            lexeme: Some(Lexeme {
                inner: &self.input[..len],
                head_noise_len,
                tail_noise_len,
            }),
        };

        self.input = &self.input[len..];

        token
    }

    #[allow(clippy::too_many_lines)]
    fn next_token(&mut self) -> Option<Token<'a>> {
        if self.is_eof {
            return None;
        }

        let input = self.input;

        if input.is_empty() {
            self.is_eof = true;
            return Some(self.advance(TokenTy::Eof, 0, 0));
        }

        let head_noise_len = skip_space(input, true);
        let input = &input[head_noise_len..];

        let mut end_pos = input.len();

        let mut char_indicies = input.char_indices().peekable();
        // XXX: What about non-ASCII characters or the question mark, comma, semicolon, or bracket?
        // Based on the token definition, tokens are separated by
        let mut is_in_string = false;
        if let Some((_, first_ch)) = char_indicies.peek() {
            match first_ch {
                '(' | ')' => {
                    end_pos = 1;
                }
                _ => loop {
                    let Some((pos, ch)) = char_indicies.next() else {
                        break;
                    };

                    if is_in_string && ch == '\\' {
                        if let Some((_, ch)) = char_indicies.peek() {
                            if *ch == '"' {
                                let _ = char_indicies.next();
                            }
                        }
                    }

                    if ch == '"' {
                        is_in_string = !is_in_string;
                    }

                    if is_in_string {
                        continue;
                    }

                    if is_space(ch) || is_newline(ch) {
                        end_pos = pos;
                        break;
                    }

                    match ch {
                        ';' => {
                            if let Some((_, ch)) = char_indicies.peek() {
                                if *ch == ';' {
                                    end_pos = pos;
                                    break;
                                }
                            }
                        }
                        '(' | ')' => {
                            end_pos = pos;
                            break;
                        }
                        _ => {}
                    }
                },
            }
        }
        drop(char_indicies);

        let token = &input[..end_pos];
        macro_rules! ret_ty {
            ($ty:expr) => {
                return Some(self.advance($ty, head_noise_len, token.len()));
            };
        }
        let mut chars = token.chars().peekable();

        macro_rules! require_next_ch_else {
            ($err_ty:expr) => {{
                let Some(ch) = chars.next() else {
                    ret_ty!($err_ty);
                };
                ch
            }};
        }

        macro_rules! tokenize_hexnum_or_hexfloat {
            ($ty:expr) => {
                // hexnum or hexfloat
                let ch = require_next_ch_else!(TokenTy::Reserved);
                if !ch.is_ascii_hexdigit() {
                    ret_ty!(TokenTy::Reserved);
                }

                loop {
                    let ch = require_next_ch_else!($ty);

                    if ch.is_ascii_hexdigit() {
                        continue;
                    }

                    if ch == '_' {
                        let ch = require_next_ch_else!(TokenTy::Reserved);
                        if !ch.is_ascii_hexdigit() {
                            ret_ty!(TokenTy::Reserved);
                        }
                        continue;
                    }

                    if ch == '.' {
                        break;
                    }

                    ret_ty!(TokenTy::Reserved);
                }

                // hexfloat
                todo!()
            };
        }

        macro_rules! tokenize_num_or_float {
            ($ty:expr) => {
                // num or float
                loop {
                    let ch = require_next_ch_else!($ty);

                    if ch.is_ascii_digit() {
                        continue;
                    }

                    if ch == '_' {
                        let ch = require_next_ch_else!(TokenTy::Reserved);
                        if !ch.is_ascii_digit() {
                            ret_ty!(TokenTy::Reserved);
                        }
                        continue;
                    }

                    if ch == '.' {
                        break;
                    }

                    ret_ty!(TokenTy::Reserved);
                }

                // float
                todo!()
            };
        }

        macro_rules! tokenize_hex_or_non_hex_value {
            ($ty:expr) => {
                // could be a hexnum
                let ch = require_next_ch_else!($ty);
                match ch {
                    'x' => {
                        tokenize_hexnum_or_hexfloat!($ty);
                    }
                    _ if ch.is_ascii_digit() || ch == '_' => {
                        tokenize_num_or_float!($ty);
                    }
                    _ => {
                        ret_ty!(TokenTy::Reserved);
                    }
                }
            };
        }

        let Some(first_ch) = chars.next() else {
            self.is_eof = true;
            return Some(self.advance(TokenTy::Eof, head_noise_len, 0));
        };

        match first_ch {
            'a'..='z' => {
                // Keyword
                if chars.all(is_id_char) {
                    let Ok(keyword) = token.parse::<Keyword>() else {
                        ret_ty!(TokenTy::Reserved);
                    };

                    ret_ty!(TokenTy::Keyword(keyword));
                }

                ret_ty!(TokenTy::Reserved);
            }
            '$' => {
                // The first character must exist
                let first_ch = require_next_ch_else!(TokenTy::Reserved);
                if is_id_char(first_ch) && chars.all(is_id_char) {
                    ret_ty!(TokenTy::Identifier);
                }

                ret_ty!(TokenTy::Reserved);
            }
            '"' => {
                // String
                loop {
                    let ch = require_next_ch_else!(TokenTy::Reserved);

                    if ch.is_ascii_control() {
                        ret_ty!(TokenTy::Reserved);
                    }

                    match ch {
                        '"' => {
                            if chars.next().is_some() {
                                ret_ty!(TokenTy::Reserved);
                            }

                            ret_ty!(TokenTy::String);
                        }
                        '\\' => {
                            let ch = require_next_ch_else!(TokenTy::Reserved);
                            match ch {
                                't' | 'n' | 'r' | '"' | '\'' | '\\' => {}
                                'u' => {
                                    // TODO: Need to detect the next few characters are a valid Unicode hexadecimal number
                                    todo!()
                                }
                                _ => {
                                    if !ch.is_ascii_hexdigit() {
                                        ret_ty!(TokenTy::Reserved);
                                    }
                                    let Some(ch) = chars.next() else {
                                        ret_ty!(TokenTy::Reserved);
                                    };
                                    if !ch.is_ascii_hexdigit() {
                                        ret_ty!(TokenTy::Reserved);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            '(' => {
                debug_assert_eq!(token.len(), 1);
                ret_ty!(TokenTy::OpenParen);
            }
            ')' => {
                debug_assert_eq!(token.len(), 1);
                ret_ty!(TokenTy::CloseParen);
            }
            '+' | '-' => {
                // a signed number or floating point
                let ch = require_next_ch_else!(TokenTy::Reserved);
                match ch {
                    '0' => {
                        tokenize_hex_or_non_hex_value!(TokenTy::SignedInt);
                    }
                    _ if ch.is_ascii_digit() => {
                        // XXX: Order matters
                        debug_assert_ne!(ch, '0');

                        tokenize_num_or_float!(TokenTy::SignedInt);
                    }
                    _ => {
                        ret_ty!(TokenTy::Reserved);
                    }
                }
            }
            '0' => {
                tokenize_hex_or_non_hex_value!(TokenTy::UnsignedInt);
            }
            _ if first_ch.is_ascii_digit() => {
                // XXX: Order matters
                debug_assert_ne!(first_ch, '0');

                tokenize_num_or_float!(TokenTy::UnsignedInt);
            }
            _ => {
                // might be any type of number (signed or unsigned or floating point)

                // Technically, this should be checked for "unknown" characters, but let it go since Reserved is thrown away anyway
                ret_ty!(TokenTy::Reserved);
            }
        }
    }
}

impl<'a> IntoIterator for Tokenizer<'a> {
    type Item = Token<'a>;

    type IntoIter = IntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

/// An [`Iterator`] for [`Token`]s.
///
/// Use [`tokenize()`] to create the iterator.
#[derive(Debug)]
pub struct IntoIter<'a> {
    tokenizer: Tokenizer<'a>,
}

impl<'a> IntoIter<'a> {
    const fn new(tokenizer: Tokenizer<'a>) -> Self {
        Self { tokenizer }
    }
}

impl<'a> Iterator for IntoIter<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokenizer.next_token()
    }
}

/// Tokenizes a string.
#[must_use]
pub const fn tokenize(input: &str) -> IntoIter<'_> {
    IntoIter::new(Tokenizer::new(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::iter;

    use proptest::prelude::*;

    fn keyword_strategy() -> impl Strategy<Value = Keyword> {
        prop_oneof![
            Just(Keyword::Inf),
            Just(Keyword::Nan),
            Just(Keyword::I32),
            Just(Keyword::I64),
            Just(Keyword::F32),
            Just(Keyword::F64),
            Just(Keyword::V128),
            Just(Keyword::FuncRef),
            Just(Keyword::ExternRef),
            Just(Keyword::Func),
            Just(Keyword::Extern),
            Just(Keyword::Param),
            Just(Keyword::Result),
            Just(Keyword::Mut),
            Just(Keyword::Block),
            Just(Keyword::End),
            Just(Keyword::Loop),
            Just(Keyword::If),
            Just(Keyword::Else),
            Just(Keyword::Unreachable),
            Just(Keyword::Nop),
            Just(Keyword::Br),
            Just(Keyword::BrIf),
            Just(Keyword::BrTable),
            Just(Keyword::Return),
            Just(Keyword::Call),
            Just(Keyword::CallIndirect),
            Just(Keyword::RefNull),
            Just(Keyword::RefIsNull),
            Just(Keyword::RefFunc),
            Just(Keyword::Drop),
            Just(Keyword::Select),
            Just(Keyword::LocalGet),
            Just(Keyword::LocalSet),
            Just(Keyword::LocalTee),
            Just(Keyword::GlobalGet),
            Just(Keyword::GlobalSet),
            Just(Keyword::TableGet),
            Just(Keyword::TableSet),
            Just(Keyword::TableSize),
            Just(Keyword::TableGrow),
            Just(Keyword::TableFill),
            Just(Keyword::TableCopy),
            Just(Keyword::TableInit),
            Just(Keyword::ElemDrop),
            Just(Keyword::I32Const),
            Just(Keyword::Import),
            Just(Keyword::Export),
            Just(Keyword::Module),
        ]
    }

    #[cfg(feature = "std")]
    proptest! {
        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_keyword_string(keyword in keyword_strategy()) {
            let keyword_str = keyword.to_string();
            prop_assert_eq!(Ok(keyword), Keyword::from_str(&keyword_str));
        }
    }

    fn internal_underscored(s: &str, underscores: &[usize]) -> String {
        debug_assert_eq!(s.len() - 1, underscores.len());
        let mut new_s = String::new();
        let mut chars = s.chars();
        let mut i = 0;
        while i < s.len() - 1 {
            new_s.push(chars.next().unwrap());
            new_s.extend(iter::repeat('_').take(underscores[i]));
            i += 1;
        }
        new_s.push(chars.next().unwrap());

        new_s
    }

    prop_compose! {
        fn unsigned_underscored()
            (n in any::<u64>())
            (vec in prop::collection::vec(prop::bool::weighted(0.3).prop_map(|b| b.then_some(1).unwrap_or_default()), n.to_string().len() - 1), n in Just(n))
            -> String {
            internal_underscored(&n.to_string(), &vec)
        }
    }

    prop_compose! {
        fn unsigned_underscored_multiple(leading_count_values: prop::sample::Select<usize>, internal_count_values: prop::sample::Select<usize>, trailing_count_values: prop::sample::Select<usize>)
            (n in any::<u64>(), leading_count in leading_count_values, trailing_count in trailing_count_values)
            (vec in prop::collection::vec(internal_count_values.clone(), n.to_string().len() - 1), n in Just(n), leading_count in Just(leading_count), trailing_count in Just(trailing_count))
            -> (bool, String) {
            let mut is_valid = true;
            if leading_count > 0 || trailing_count > 0 || vec.iter().any(|v| *v > 1) {
                is_valid = false;
            }
            (is_valid, "_".repeat(leading_count) + &internal_underscored(&n.to_string(), &vec) + &"_".repeat(trailing_count))
        }
    }

    prop_compose! {
        fn unsigned_hex_underscored()
            (n in any::<u64>())
            (vec in prop::collection::vec(prop::bool::weighted(0.3).prop_map(|b| b.then_some(1).unwrap_or_default()), n.to_string().len() - 1), n in Just(n))
            -> String {
            internal_underscored(&n.to_string(), &vec)
        }
    }

    const ZERO_COUNT_VALUE: &[usize] = &[0];
    const INTERNAL_UNDERSCORE_COUNT_VALID_VALUES: &[usize] = &[0, 0, 0, 0, 0, 0, 0, 1, 1, 1];
    const INTERNAL_UNDERSCORE_COUNT_INVALID_VALUES: &[usize] = &[0, 0, 0, 0, 0, 0, 0, 1, 1, 2];
    const LEADING_UNDERSCORE_COUNT_INVALID_VALUES: &[usize] = &[0, 0, 0, 0, 0, 0, 0, 1, 2, 3];
    const TRAILING_UNDERSCORE_COUNT_INVALID_VALUES: &[usize] = &[0, 0, 0, 0, 0, 0, 0, 1, 2, 3];

    #[cfg(feature = "std")]
    proptest! {
        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_tokenize_unsigned(n_string in any::<u64>().prop_map(|n| n.to_string())) {
            let mut tokenizer = tokenize(&n_string);

            prop_assert_eq!(
                Some(Token {
                    ty: TokenTy::UnsignedInt,
                    lexeme: Some(Lexeme {
                        inner: &n_string,
                        head_noise_len: 0,
                        tail_noise_len: 0
                    })
                }),
                tokenizer.next()
            );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_tokenize_unsigned_with_underscores(n_string in unsigned_underscored()) {
            let mut tokenizer = tokenize(&n_string);

            prop_assert_eq!(
                Some(Token {
                    ty: TokenTy::UnsignedInt,
                    lexeme: Some(Lexeme {
                        inner: &n_string,
                        head_noise_len: 0,
                        tail_noise_len: 0
                    })
                }),
                tokenizer.next()
            );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_unsigned_num_with_invalid_underscores(
            (is_valid, n_string)
            in unsigned_underscored_multiple(
                prop::sample::select(ZERO_COUNT_VALUE),
                prop::sample::select(INTERNAL_UNDERSCORE_COUNT_INVALID_VALUES),
                prop::sample::select(ZERO_COUNT_VALUE),
                )
            ) {
            let mut tokenizer = tokenize(&n_string);

            assert_eq!(
                tokenizer.next(),
                Some(Token {
                    ty: if is_valid { TokenTy::UnsignedInt } else { TokenTy::Reserved },
                    lexeme: Some(Lexeme {
                        inner: &n_string,
                        head_noise_len: 0,
                        tail_noise_len: 0
                    })
                })
            );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_tokenize_unsigned_with_leading_underscore(
            (is_valid, n_string)
            in unsigned_underscored_multiple(
                prop::sample::select(LEADING_UNDERSCORE_COUNT_INVALID_VALUES),
                prop::sample::select(INTERNAL_UNDERSCORE_COUNT_INVALID_VALUES),
                prop::sample::select(ZERO_COUNT_VALUE),
                )
            ) {
            let mut tokenizer = tokenize(&n_string);

            prop_assert_eq!(
                Some(Token {
                    ty: if is_valid { TokenTy::UnsignedInt } else { TokenTy::Reserved },
                    lexeme: Some(Lexeme {
                        inner: &n_string,
                        head_noise_len: 0,
                        tail_noise_len: 0
                    })
                }),
                tokenizer.next()
            );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_unsigned_num_with_invalid_trailing_underscore(
            (is_valid, n_string)
            in unsigned_underscored_multiple(
                prop::sample::select(ZERO_COUNT_VALUE),
                prop::sample::select(INTERNAL_UNDERSCORE_COUNT_INVALID_VALUES),
                prop::sample::select(TRAILING_UNDERSCORE_COUNT_INVALID_VALUES),
                )
            ) {
            let mut tokenizer = tokenize(&n_string);

            assert_eq!(
                tokenizer.next(),
                Some(Token {
                    ty: if is_valid { TokenTy::UnsignedInt } else { TokenTy::Reserved },
                    lexeme: Some(Lexeme {
                        inner: &n_string,
                        head_noise_len: 0,
                        tail_noise_len: 0
                    })
                })
            );
        }
    }

    #[test]
    fn test_string() {
        let mut tokenizer = tokenize(r#""hello world""#);

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::String,
                lexeme: Some(Lexeme {
                    inner: r#""hello world""#,
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_string_no_quote_end() {
        let mut tokenizer = tokenize(r#""hello world"#);

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: r#""hello world"#,
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_string_escaped_quote() {
        let mut tokenizer = tokenize(r#""hello\" \" world \" ""#);

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::String,
                lexeme: Some(Lexeme {
                    inner: r#""hello\" \" world \" ""#,
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_identifier() {
        let mut tokenizer = tokenize("$a");

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Identifier,
                lexeme: Some(Lexeme {
                    inner: "$a",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_dollar_sign_eof() {
        let mut tokenizer = tokenize("$");

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: "$",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_dollar_sign_then_not_id_char() {
        let mut tokenizer = tokenize("$ ");

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: "$ ",
                    head_noise_len: 0,
                    tail_noise_len: 1
                })
            })
        );
    }

    #[test]
    fn test_open_paren() {
        let mut tokenizer = tokenize("(");

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::OpenParen,
                lexeme: Some(Lexeme {
                    inner: "(",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_close_paren() {
        let mut tokenizer = tokenize(") ");

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::CloseParen,
                lexeme: Some(Lexeme {
                    inner: ") ",
                    head_noise_len: 0,
                    tail_noise_len: 1
                })
            })
        );
    }

    #[test]
    fn test_paren_with_trailing() {
        let mut tokenizer = tokenize(" (abc ");

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::OpenParen,
                lexeme: Some(Lexeme {
                    inner: " (",
                    head_noise_len: 1,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_paren_with_leading() {
        let mut tokenizer = tokenize("$id(abc");

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Identifier,
                lexeme: Some(Lexeme {
                    inner: "$id",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::OpenParen,
                lexeme: Some(Lexeme {
                    inner: "(",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_reserved_not_keyword() {
        let mut tokenizer = tokenize("0$x");

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: "0$x",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_reserved_two_strings() {
        let mut tokenizer = tokenize(r#""a""b""#);

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: r#""a""b""#,
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
    }

    #[test]
    fn test_module() {
        let mut tokenizer = tokenize(r#"(module )"#);

        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::OpenParen,
                lexeme: Some(Lexeme {
                    inner: "(",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Keyword(Keyword::Module),
                lexeme: Some(Lexeme {
                    inner: "module ",
                    head_noise_len: 0,
                    tail_noise_len: 1
                })
            })
        );
        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::CloseParen,
                lexeme: Some(Lexeme {
                    inner: ")",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
        assert_eq!(
            tokenizer.next(),
            Some(Token {
                ty: TokenTy::Eof,
                lexeme: Some(Lexeme {
                    inner: "",
                    head_noise_len: 0,
                    tail_noise_len: 0
                })
            })
        );
        assert_eq!(tokenizer.next(), None,);
    }
}
