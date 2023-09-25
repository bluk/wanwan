//! Lexer to tokenize WAT.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;
use core::{
    convert::Infallible,
    fmt,
    marker::PhantomData,
    str::{FromStr, Utf8Error},
};
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::fmt::{OutOfBoundsError, Read, ReadError, SliceRead};

const HORIZONTAL_TABULATION: u8 = 9;
const LINE_FEED: u8 = 10;
const CARRIAGE_RETURN: u8 = 13;

#[inline]
#[must_use]
const fn is_space(b: u8) -> bool {
    matches!(b, b' ' | HORIZONTAL_TABULATION)
}

/// Returns true if the byte is a newline character.
#[inline]
#[must_use]
const fn is_newline(b: u8) -> bool {
    matches!(b, LINE_FEED | CARRIAGE_RETURN)
}

fn skip_space<B, R>(
    input: &mut R,
    buf: &mut B,
    consume_new_lines: bool,
) -> Result<(), TokenizerError<B::Error, ReadError<R::Error>>>
where
    B: Buf,
    R: Read,
{
    'outer: loop {
        let Some(b) = input.peek() else {
            return Ok(());
        };

        if is_space(b) || is_newline(b) {
            let b = input.next().map_err(TokenizerError::Read)?;
            buf.push(b).map_err(TokenizerError::Buf)?;
            continue;
        }

        if is_newline(b) {
            if consume_new_lines {
                let b = input.next().map_err(TokenizerError::Read)?;
                buf.push(b).map_err(TokenizerError::Buf)?;
                continue;
            }

            return Ok(());
        }

        match b {
            b';' => {
                let Some(b) = input.peek2() else {
                    return Ok(());
                };
                if b != b';' {
                    return Ok(());
                }

                // Skip line comment
                loop {
                    let b = input.next().map_err(TokenizerError::Read)?;
                    buf.push(b).map_err(TokenizerError::Buf)?;

                    if b == LINE_FEED {
                        continue 'outer;
                    }
                }
            }
            b'(' => {
                let Some(b) = input.peek2() else {
                    return Ok(());
                };

                if b != b';' {
                    return Ok(());
                }

                // Skip block comment
                let mut end_delim_count = 0usize;
                loop {
                    let b = input.next().map_err(TokenizerError::Read)?;
                    buf.push(b).map_err(TokenizerError::Buf)?;

                    match b {
                        b'(' => {
                            let Some(b) = input.peek() else {
                                return Ok(());
                            };
                            if b == b';' {
                                let b = input.next().map_err(TokenizerError::Read)?;
                                buf.push(b).map_err(TokenizerError::Buf)?;

                                end_delim_count += 1;
                            }
                        }
                        b';' => {
                            let Some(b) = input.peek() else {
                                return Ok(());
                            };
                            if b == b')' {
                                let b = input.next().map_err(TokenizerError::Read)?;
                                buf.push(b).map_err(TokenizerError::Buf)?;

                                end_delim_count -= 1;
                                if end_delim_count == 0 {
                                    continue 'outer;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => return Ok(()),
        }
    }
}

#[inline]
#[must_use]
const fn is_id_char(b: u8) -> bool {
    matches!(b,
      b'0'..=b'9'
      | b'A'..=b'Z'
      | b'a'..=b'z'
      | b'!' | b'#' | b'$' | b'%' | b'&' | b'\'' | b'*' | b'+' | b'-' | b'.' | b'/'
      | b':' | b'<' | b'=' | b'>' | b'?' | b'@' | b'\\' | b'^' | b'_' | b'`' | b'|' | b'~'
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
pub enum TokenTy<K> {
    /// End of file
    Eof,
    /// Any literal terminal in the grammar
    Keyword(K),
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
    pub inner: &'a [u8],
    pub head_noise_len: usize,
    pub token_end_offset: usize,
    // tail_noise_len is inner.len() - token_end_offset
}

impl<'a> Lexeme<'a> {
    /// Returns the parsed token as a string without any spaces or comments.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying parsed token bytes are not an UTF-8 string.
    #[inline]
    pub fn as_noiseless_str(&self) -> Result<&'a str, Utf8Error> {
        core::str::from_utf8(&self.inner[self.head_noise_len..self.token_end_offset])
    }

    /// Returns the entire lexeme as a string with all spaces and comments.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not an UTF-8 string.
    pub fn as_str(&self) -> Result<&'a str, Utf8Error> {
        core::str::from_utf8(self.inner)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a, K> {
    ty: TokenTy<K>,
    lexeme: Option<Lexeme<'a>>,
}

impl<'a, K> fmt::Display for Token<'a, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(lexeme) = self.lexeme {
            return fmt::Display::fmt(lexeme.as_str().map_err(|_| fmt::Error)?, f);
        }
        Ok(())
    }
}

impl<'a, K> Token<'a, K> {
    /// Returns true if the token type is as expected and there are lexeme bytes representing the token.
    #[inline]
    #[must_use]
    pub fn is_type_present(&self, ty: &TokenTy<K>) -> bool
    where
        K: PartialEq,
    {
        self.ty == *ty && self.lexeme.is_some()
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
    pub fn as_keyword(&self) -> Option<&K> {
        match &self.ty {
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
    pub fn as_string(&self) -> Option<Result<&str, Utf8Error>> {
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
                .map(|l| l.as_noiseless_str().map(|s| &s[1..s.len() - 1])),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedLexeme {
    pub inner: Vec<u8>,
    pub head_noise_len: usize,
    pub token_end_offset: usize,
}

impl<'a> From<Lexeme<'a>> for OwnedLexeme {
    fn from(value: Lexeme<'a>) -> Self {
        OwnedLexeme {
            inner: value.inner.to_vec(),
            head_noise_len: value.head_noise_len,
            token_end_offset: value.token_end_offset,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedToken<K> {
    ty: TokenTy<K>,
    lexeme: Option<OwnedLexeme>,
}

impl<'a, K> From<Token<'a, K>> for OwnedToken<K> {
    fn from(value: Token<'a, K>) -> Self {
        Self {
            ty: value.ty,
            lexeme: value.lexeme.map(OwnedLexeme::from),
        }
    }
}

pub trait Buf {
    type Error;

    fn clear(&mut self);

    #[must_use]
    fn len(&self) -> usize;

    #[must_use]
    fn is_empty(&self) -> bool;

    /// Pushes a byte into the buffer.
    ///
    /// # Errors
    ///
    /// If the byte cannot be pushed into the buffer.
    fn push(&mut self, b: u8) -> Result<(), Self::Error>;

    #[must_use]
    fn as_slice(&self) -> &[u8];
}

impl Buf for Vec<u8> {
    type Error = Infallible;

    fn clear(&mut self) {
        (*self).clear();
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn is_empty(&self) -> bool {
        (*self).is_empty()
    }

    fn push(&mut self, b: u8) -> Result<(), Self::Error> {
        (*self).push(b);
        Ok(())
    }

    fn as_slice(&self) -> &[u8] {
        (*self).as_slice()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError<B, R> {
    Buf(B),
    Read(R),
}

#[derive(Debug)]
pub struct Tokenizer<B, R, BE, RE> {
    buf: B,
    input: R,
    held_error: Option<TokenizerError<BE, RE>>,
}

impl<B, R, BE, RE> Tokenizer<B, R, BE, RE> {
    #[inline]
    #[must_use]
    fn new(buf: B, input: R) -> Self {
        Self {
            buf,
            input,
            held_error: None,
        }
    }
}

type NextTokenResult<'a, K, BE, RE> = Result<Token<'a, K>, TokenizerError<BE, ReadError<RE>>>;

impl<B, R, BE, RE> Tokenizer<B, R, BE, ReadError<RE>>
where
    B: Buf<Error = BE>,
    R: Read<Error = RE>,
{
    fn advance<K>(
        &mut self,
        ty: TokenTy<K>,
        head_noise_len: usize,
        token_end_offset: usize,
    ) -> Token<'_, K> {
        match skip_space(&mut self.input, &mut self.buf, false) {
            Ok(()) => {}
            Err(e) => {
                self.held_error = Some(e);
            }
        }

        let token = Token {
            ty,
            lexeme: Some(Lexeme {
                inner: self.buf.as_slice(),
                head_noise_len,
                token_end_offset,
            }),
        };

        token
    }

    #[allow(clippy::too_many_lines)]
    fn next_token<K>(&mut self) -> NextTokenResult<'_, K, BE, RE>
    where
        K: FromStr,
    {
        if let Some(e) = self.held_error.take() {
            return Err(e);
        }

        self.buf.clear();

        match skip_space(&mut self.input, &mut self.buf, true) {
            Ok(()) => {}
            Err(e) => {
                self.held_error = Some(e);

                return Ok(Token {
                    ty: TokenTy::Eof,
                    lexeme: Some(Lexeme {
                        inner: self.buf.as_slice(),
                        head_noise_len: self.buf.len(),
                        token_end_offset: self.buf.len(),
                    }),
                });
            }
        }

        let head_noise_len = self.buf.len();

        macro_rules! read_next {
            () => {{
                let b = self.input.next().map_err(TokenizerError::Read)?;
                self.buf.push(b).map_err(TokenizerError::Buf)?;
                b
            }};
        }

        // XXX: What about non-ASCII characters or the question mark, comma, semicolon, or bracket?
        // Based on the token definition, tokens are separated by
        let mut is_in_string = false;
        let Some(b) = self.input.peek() else {
            match self.input.next() {
                Ok(_) => unreachable!(),
                Err(e) => {
                    self.held_error = Some(TokenizerError::Read(e));
                }
            }

            return Ok(Token {
                ty: TokenTy::Eof,
                lexeme: Some(Lexeme {
                    inner: self.buf.as_slice(),
                    head_noise_len: self.buf.len(),
                    token_end_offset: self.buf.len(),
                }),
            });
        };
        match b {
            b'(' | b')' => {
                let _ = read_next!();
            }
            _ => 'outer: loop {
                macro_rules! peek_next {
                    () => {{
                        let Some(b) = self.input.peek() else {
                            break 'outer;
                        };
                        b
                    }};
                }

                macro_rules! peek2_next {
                    () => {{
                        let Some(b) = self.input.peek2() else {
                            let _ = read_next!();
                            break 'outer;
                        };
                        b
                    }};
                }

                let b = peek_next!();

                if is_in_string && b == b'\\' {
                    let b = peek2_next!();
                    if b == b'"' {
                        let _ = read_next!();
                        let _ = read_next!();
                        continue;
                    }
                }

                if b == b'"' {
                    is_in_string = !is_in_string;
                }

                if is_in_string {
                    let _ = read_next!();
                    continue;
                }

                if is_space(b) || is_newline(b) {
                    break;
                }

                match b {
                    b';' => {
                        let b = peek2_next!();
                        if b == b';' {
                            break;
                        }
                    }
                    b'(' | b')' => {
                        break;
                    }
                    _ => {}
                }

                let _ = read_next!();
            },
        }
        let token_end_offset = self.buf.len();

        let token = &self.buf.as_slice()[head_noise_len..];
        macro_rules! ret_ty {
            ($ty:expr) => {
                return Ok(self.advance($ty, head_noise_len, token_end_offset));
            };
        }
        let mut bytes = token.iter().copied().peekable();

        macro_rules! require_next_ch_else {
            ($err_ty:expr) => {{
                let Some(b) = bytes.next() else {
                    ret_ty!($err_ty);
                };
                b
            }};
        }

        macro_rules! tokenize_hexnum_or_hexfloat {
            ($ty:expr) => {
                // hexnum or hexfloat
                let b = require_next_ch_else!(TokenTy::Reserved);
                if !b.is_ascii_hexdigit() {
                    ret_ty!(TokenTy::Reserved);
                }

                loop {
                    let b = require_next_ch_else!($ty);

                    if b.is_ascii_hexdigit() {
                        continue;
                    }

                    if b == b'_' {
                        let b = require_next_ch_else!(TokenTy::Reserved);
                        if !b.is_ascii_hexdigit() {
                            ret_ty!(TokenTy::Reserved);
                        }
                        continue;
                    }

                    if b == b'.' {
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
                    let b = require_next_ch_else!($ty);

                    if b.is_ascii_digit() {
                        continue;
                    }

                    if b == b'_' {
                        let b = require_next_ch_else!(TokenTy::Reserved);
                        if !b.is_ascii_digit() {
                            ret_ty!(TokenTy::Reserved);
                        }
                        continue;
                    }

                    if b == b'.' {
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
                let b = require_next_ch_else!($ty);
                match b {
                    b'x' => {
                        tokenize_hexnum_or_hexfloat!($ty);
                    }
                    _ if b.is_ascii_digit() || b == b'_' => {
                        tokenize_num_or_float!($ty);
                    }
                    _ => {
                        ret_ty!(TokenTy::Reserved);
                    }
                }
            };
        }

        let Some(first_byte) = bytes.next() else {
            unreachable!()
        };

        match first_byte {
            b'a'..=b'z' => {
                // Keyword
                if bytes.all(is_id_char) {
                    let Ok(s) = core::str::from_utf8(token) else {
                        ret_ty!(TokenTy::Reserved);
                    };

                    let Ok(keyword) = s.parse::<K>() else {
                        ret_ty!(TokenTy::Reserved);
                    };

                    ret_ty!(TokenTy::Keyword(keyword));
                }

                ret_ty!(TokenTy::Reserved);
            }
            b'$' => {
                // The first character must exist
                let first_byte = require_next_ch_else!(TokenTy::Reserved);
                if is_id_char(first_byte) && bytes.all(is_id_char) {
                    ret_ty!(TokenTy::Identifier);
                }

                ret_ty!(TokenTy::Reserved);
            }
            b'"' => {
                // String
                loop {
                    let b = require_next_ch_else!(TokenTy::Reserved);

                    if b.is_ascii_control() {
                        ret_ty!(TokenTy::Reserved);
                    }

                    match b {
                        b'"' => {
                            if bytes.next().is_some() {
                                ret_ty!(TokenTy::Reserved);
                            }

                            ret_ty!(TokenTy::String);
                        }
                        b'\\' => {
                            let b = require_next_ch_else!(TokenTy::Reserved);
                            match b {
                                b't' | b'n' | b'r' | b'"' | b'\'' | b'\\' => {}
                                b'u' => {
                                    // TODO: Need to detect the next few characters are a valid Unicode hexadecimal number
                                    todo!()
                                }
                                _ => {
                                    if !b.is_ascii_hexdigit() {
                                        ret_ty!(TokenTy::Reserved);
                                    }
                                    let Some(b) = bytes.next() else {
                                        ret_ty!(TokenTy::Reserved);
                                    };
                                    if !b.is_ascii_hexdigit() {
                                        ret_ty!(TokenTy::Reserved);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            b'(' => {
                debug_assert_eq!(token.len(), 1);
                ret_ty!(TokenTy::OpenParen);
            }
            b')' => {
                debug_assert_eq!(token.len(), 1);
                ret_ty!(TokenTy::CloseParen);
            }
            b'+' | b'-' => {
                // a signed number or floating point
                let b = require_next_ch_else!(TokenTy::Reserved);
                match b {
                    b'0' => {
                        tokenize_hex_or_non_hex_value!(TokenTy::SignedInt);
                    }
                    _ if b.is_ascii_digit() => {
                        // XXX: Order matters
                        debug_assert_ne!(b, b'0');

                        tokenize_num_or_float!(TokenTy::SignedInt);
                    }
                    _ => {
                        ret_ty!(TokenTy::Reserved);
                    }
                }
            }
            b'0' => {
                tokenize_hex_or_non_hex_value!(TokenTy::UnsignedInt);
            }
            _ if first_byte.is_ascii_digit() => {
                // XXX: Order matters
                debug_assert_ne!(first_byte, b'0');

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

/// An [`Iterator`] for [`OwnedToken`]s.
#[derive(Debug)]
pub struct OwnedIter<B, R, K, BE, RE> {
    tokenizer: Tokenizer<B, R, BE, RE>,
    k_ty: PhantomData<K>,
}

impl<B, R, K, BE, RE> OwnedIter<B, R, K, BE, RE> {
    pub fn new(tokenizer: Tokenizer<B, R, BE, RE>) -> Self {
        Self {
            tokenizer,
            k_ty: PhantomData,
        }
    }
}

impl<B, R, K, BE, RE> Iterator for OwnedIter<B, R, K, BE, ReadError<RE>>
where
    R: Read<Error = RE>,
    B: Buf<Error = BE>,
    K: FromStr,
{
    type Item = Result<OwnedToken<K>, TokenizerError<BE, ReadError<RE>>>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.tokenizer.next_token().map(|t| OwnedToken {
            ty: t.ty,
            lexeme: t.lexeme.map(|l| OwnedLexeme {
                inner: l.inner.to_vec(),
                head_noise_len: l.head_noise_len,
                token_end_offset: l.token_end_offset,
            }),
        }))
    }
}

/// Tokenizes a string.
#[must_use]
pub fn from_str(
    input: &str,
) -> Tokenizer<Vec<u8>, SliceRead<'_>, Infallible, ReadError<OutOfBoundsError>> {
    Tokenizer::new(Vec::new(), SliceRead::new(input.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::{
        string::{String, ToString},
        vec,
    };
    use core::iter;
    #[cfg(feature = "std")]
    use std::{
        string::{String, ToString},
        vec,
    };

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
    const INTERNAL_UNDERSCORE_COUNT_INVALID_VALUES: &[usize] = &[0, 0, 0, 0, 0, 0, 0, 1, 1, 2];
    const LEADING_UNDERSCORE_COUNT_INVALID_VALUES: &[usize] = &[0, 0, 0, 0, 0, 0, 0, 1, 2, 3];
    const TRAILING_UNDERSCORE_COUNT_INVALID_VALUES: &[usize] = &[0, 0, 0, 0, 0, 0, 0, 1, 2, 3];

    #[cfg(feature = "std")]
    proptest! {
        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_tokenize_unsigned(n_string in any::<u64>().prop_map(|n| n.to_string())) {
            let mut tokenizer = from_str(&n_string);

            prop_assert_eq!(
                Ok(Token {
                    ty: TokenTy::UnsignedInt,
                    lexeme: Some(Lexeme {
                        inner: n_string.as_bytes(),
                        head_noise_len: 0,
                        token_end_offset: n_string.len(),
                    })
                }),
                tokenizer.next_token::<Keyword>()
            );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_tokenize_unsigned_with_underscores(n_string in unsigned_underscored()) {
            let mut tokenizer = from_str(&n_string);

            prop_assert_eq!(
                Ok(Token {
                    ty: TokenTy::UnsignedInt,
                    lexeme: Some(Lexeme {
                        inner: n_string.as_bytes(),
                        head_noise_len: 0,
                        token_end_offset: n_string.len(),
                    })
                }),
                tokenizer.next_token::<Keyword>()
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
            let mut tokenizer = from_str(&n_string);

            assert_eq!(
                Ok(Token {
                    ty: if is_valid { TokenTy::UnsignedInt } else { TokenTy::Reserved },
                    lexeme: Some(Lexeme {
                        inner: n_string.as_bytes(),
                        head_noise_len: 0,
                        token_end_offset: n_string.len(),
                    })
                }),
                tokenizer.next_token::<Keyword>()
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
            let mut tokenizer = from_str(&n_string);

            prop_assert_eq!(
                Ok(Token {
                    ty: if is_valid { TokenTy::UnsignedInt } else { TokenTy::Reserved },
                    lexeme: Some(Lexeme {
                        inner: n_string.as_bytes(),
                        head_noise_len: 0,
                        token_end_offset: n_string.len(),
                    })
                }),
                tokenizer.next_token::<Keyword>()
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
            let mut tokenizer = from_str(&n_string);

            assert_eq!(
                Ok(Token {
                    ty: if is_valid { TokenTy::UnsignedInt } else { TokenTy::Reserved },
                    lexeme: Some(Lexeme {
                        inner: n_string.as_bytes(),
                        head_noise_len: 0,
                        token_end_offset: n_string.len(),
                    })
                }),
                tokenizer.next_token::<Keyword>()
            );
        }
    }

    #[test]
    fn test_string() {
        let mut tokenizer = from_str(r#""hello world""#);

        let s = r#""hello world""#;
        assert_eq!(
            Ok(Token {
                ty: TokenTy::String,
                lexeme: Some(Lexeme {
                    inner: s.as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: s.len(),
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_string_no_quote_end() {
        let mut tokenizer = from_str(r#""hello world"#);

        let s = r#""hello world"#;
        assert_eq!(
            Ok(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: s.as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: s.len()
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_string_escaped_quote() {
        let mut tokenizer = from_str(r#""hello\" \" world \" ""#);

        let s = r#""hello\" \" world \" ""#;
        assert_eq!(
            Ok(Token {
                ty: TokenTy::String,
                lexeme: Some(Lexeme {
                    inner: s.as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: s.len()
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_identifier() {
        let mut tokenizer = from_str("$a");

        assert_eq!(
            Ok(Token {
                ty: TokenTy::Identifier,
                lexeme: Some(Lexeme {
                    inner: "$a".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 2
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_dollar_sign_eof() {
        let mut tokenizer = from_str("$");

        assert_eq!(
            Ok(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: "$".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 1
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_dollar_sign_then_not_id_char() {
        let mut tokenizer = from_str("$ ");

        assert_eq!(
            Ok(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: "$ ".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 1
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_open_paren() {
        let mut tokenizer = from_str("(");

        assert_eq!(
            Ok(Token {
                ty: TokenTy::OpenParen,
                lexeme: Some(Lexeme {
                    inner: "(".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 1
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_close_paren() {
        let mut tokenizer = from_str(") ");

        assert_eq!(
            Ok(Token {
                ty: TokenTy::CloseParen,
                lexeme: Some(Lexeme {
                    inner: ") ".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 1
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_paren_with_trailing() {
        let mut tokenizer = from_str(" (abc ");

        assert_eq!(
            Ok(Token {
                ty: TokenTy::OpenParen,
                lexeme: Some(Lexeme {
                    inner: " (".as_bytes(),
                    head_noise_len: 1,
                    token_end_offset: 2
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_paren_with_leading() {
        let mut tokenizer = from_str("$id(abc");

        assert_eq!(
            Ok(Token {
                ty: TokenTy::Identifier,
                lexeme: Some(Lexeme {
                    inner: "$id".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 3
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
        assert_eq!(
            Ok(Token {
                ty: TokenTy::OpenParen,
                lexeme: Some(Lexeme {
                    inner: "(".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 1
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
        assert_eq!(
            Ok(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: "abc".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 3
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_reserved_not_keyword() {
        let mut tokenizer = from_str("0$x");

        assert_eq!(
            Ok(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: "0$x".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 3,
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_reserved_two_strings() {
        let mut tokenizer = from_str(r#""a""b""#);

        let s = r#""a""b""#;
        assert_eq!(
            Ok(Token {
                ty: TokenTy::Reserved,
                lexeme: Some(Lexeme {
                    inner: s.as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: s.len(),
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
    }

    #[test]
    fn test_module() {
        let mut tokenizer = from_str(r#"(module )"#);

        assert_eq!(
            Ok(Token {
                ty: TokenTy::OpenParen,
                lexeme: Some(Lexeme {
                    inner: "(".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 1
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
        assert_eq!(
            Ok(Token {
                ty: TokenTy::Keyword(Keyword::Module),
                lexeme: Some(Lexeme {
                    inner: "module ".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 6
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
        assert_eq!(
            Ok(Token {
                ty: TokenTy::CloseParen,
                lexeme: Some(Lexeme {
                    inner: ")".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 1,
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
        assert_eq!(
            Ok(Token {
                ty: TokenTy::Eof,
                lexeme: Some(Lexeme {
                    inner: "".as_bytes(),
                    head_noise_len: 0,
                    token_end_offset: 0,
                })
            }),
            tokenizer.next_token::<Keyword>()
        );
        assert_eq!(
            Err(TokenizerError::Read(ReadError::new(OutOfBoundsError, true))),
            tokenizer.next_token::<Keyword>()
        );
    }
}
