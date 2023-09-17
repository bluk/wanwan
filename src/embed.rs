//! Embedding API as described in Appendix A.1

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{
    fmt,
    string::{String, ToString},
    vec::Vec,
};
#[cfg(feature = "std")]
use std::{
    error, fmt,
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    fmt::binary::{self},
    module::{ty::ExternTy, Module},
};

#[derive(Debug)]
enum InnerError {
    DecodeError(String, u64),
}

impl<E> From<binary::Error<E>> for InnerError
where
    E: fmt::Display,
{
    fn from(value: binary::Error<E>) -> Self {
        Self::DecodeError(value.to_string(), value.pos())
    }
}

/// Generic error
#[derive(Debug)]
pub struct Error {
    inner: InnerError,
}

impl From<InnerError> for Error {
    fn from(value: InnerError) -> Self {
        Self { inner: value }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            InnerError::DecodeError(error, _) => f.write_str(error),
        }
    }
}

#[cfg(feature = "std")]
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

/// Decodes a Wasm module in binary format.
///
/// # Errors
///
/// Returns an error if the data is invalid or if a problem occurs when reading.
#[inline]
pub fn module_decode(bytes: &[u8]) -> Result<Module, Error> {
    binary::from_slice(bytes).map_err(|e| Error::from(InnerError::from(e)))
}

/// Validates a Wasm module.
///
/// # Errors
///
/// Returns an error if the module is invalid.
#[inline]
pub fn module_validate(m: Module) -> Result<Module, Error> {
    // Module is validated during decoding
    Ok(m)
}

/// Returns a module's imports.
#[must_use]
pub fn module_imports(m: &Module) -> Vec<(String, String, ExternTy)> {
    m.import_external_tys()
        .map(|(module, name, ty)| (module.to_string(), name.to_string(), ty))
        .collect()
}

/// Returns a module's exports.
#[must_use]
pub fn module_exports(m: &Module) -> Vec<(String, ExternTy)> {
    m.export_external_tys()
        .map(|(name, ty)| (name.to_string(), ty))
        .collect()
}
