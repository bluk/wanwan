//! Text format

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::borrow::Cow;
#[cfg(feature = "std")]
use std::borrow::Cow;

use self::lexer::{
    Buf, Keyword, Lexeme, OwnedIter, OwnedLexeme, OwnedToken, Token, TokenizerError,
};

use super::{Read, ReadError};

/// Recommended extension for files containing Wasm modules in text format.
pub const EXTENSION: &str = "wat";

pub mod lexer;

#[derive(Debug, Clone, PartialEq, Eq)]
enum ModuleParserState {
    Start,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ModuleEventTy {
    OpenParen,
    CloseParen,
    Module,
    Unexpected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenState<K> {
    Present(OwnedToken<K>),
    Missing,
    Unexpected(OwnedToken<K>),
}

#[derive(Debug)]
struct ModuleEvent {
    ty: ModuleEventTy,
    token: TokenState<Keyword>,
}

type TokError<BE, RE> = TokenizerError<BE, ReadError<RE>>;
type OwnedTokResult<BE, RE> = Result<OwnedToken<Keyword>, TokError<BE, RE>>;
type OptionOwnedTokResult<BE, RE> = Option<OwnedTokResult<BE, RE>>;
type OptionRefOwnedTok<'a> = Option<&'a OwnedToken<Keyword>>;

#[derive(Debug)]
struct ModuleParser<B, R>
where
    B: Buf,
    R: Read,
{
    tokenizer: OwnedIter<B, R, Keyword, B::Error, ReadError<R::Error>>,
    state: ModuleParserState,
    peeked_token: OptionOwnedTokResult<B::Error, R::Error>,
    // Option<<OwnedIter<B, R, Keyword, B::Error, ReadError<R::Error>> as Iterator>::Item>,
    is_errored: bool,
}

impl<B, R> ModuleParser<B, R>
where
    B: Buf,
    R: Read,
{
    pub fn new(tokenizer: lexer::Tokenizer<B, R, B::Error, ReadError<R::Error>>) -> Self {
        Self {
            tokenizer: OwnedIter::new(tokenizer),
            state: ModuleParserState::Start,
            peeked_token: None,
            is_errored: false,
        }
    }
}

impl<B, R> ModuleParser<B, R>
where
    B: Buf,
    R: Read,
{
    fn next(&mut self) -> OptionOwnedTokResult<B::Error, R::Error> {
        match self.peeked_token.take() {
            Some(r) => Some(r),
            None => self.tokenizer.next(),
        }
    }

    fn peek(&mut self) -> OptionRefOwnedTok<'_> {
        // XXX: take() to avoid borrow checker shenagians
        if let Some(peeked) = self.peeked_token.take() {
            self.peeked_token = Some(peeked);
            match &self.peeked_token {
                Some(Ok(t)) => return Some(t),
                Some(Err(_)) => return None,
                None => unreachable!(),
            }
        }

        match self.tokenizer.next()? {
            Ok(t) => {
                self.peeked_token = Some(Ok(t));
                match &self.peeked_token {
                    Some(Ok(t)) => Some(t),
                    Some(Err(_)) | None => unreachable!(),
                }
            }
            Err(e) => {
                self.peeked_token = Some(Err(e));
                None
            }
        }
    }
}

impl<B, R> Iterator for ModuleParser<B, R>
where
    B: Buf,
    R: Read,
{
    type Item = Result<ModuleEvent, TokenizerError<B::Error, ReadError<R::Error>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_errored {
            return None;
        }

        let peek = match self.peek() {
            Some(t) => t,
            None => match self.next() {
                Some(Err(e)) => {
                    self.is_errored = true;
                    return Some(Err(e));
                }
                None => {
                    return None;
                }
                Some(Ok(_)) => unreachable!(),
            },
        };

        match self.state {
            ModuleParserState::Start => {
                // let token = self.tokenizer.next()?;
                // if !token.is_open_paren() {
                //     self.peeked_token = Some(token);
                // }
            }
        }

        unsafe {
            Some(Ok(ModuleEvent {
                ty: ModuleEventTy::OpenParen,
                token: TokenState::Present(self.next().unwrap_unchecked().unwrap_unchecked()),
            }))
        }

        // Some(Ok(ModuleEvent {
        //     ty: ModuleEventTy::OpenParen,
        //     token: TokenState::Present(OwnedToken {
        //         ty: lexer::TokenTy::OpenParen,
        //         lexeme: Some(OwnedLexeme {
        //             inner: "(".as_bytes().to_vec(),
        //             head_noise_len: 0,
        //             token_end_offset: 1,
        //         }),
        //     }),
        // }))
    }
}

#[cfg(test)]
mod tests {}
