//! Text format

/// Recommended extension for files containing Wasm modules in text format.
pub const EXTENSION: &str = "wat";

pub mod lexer;

// #[derive(Debug)]
// struct Parser<R> {
//     tokenizer: lexer::Tokenizer<R>,
// }
