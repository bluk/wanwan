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
    exec::{
        self,
        val::{ExternVal, Ref, Val},
        FuncAddr, GlobalAddr, HostcodeError, MemAddr, ModuleInst, Store, TableAddr,
    },
    fmt::binary::{self},
    module::{
        ty::{ExternTy, FuncTy, GlobalTy, MemTy, TableTy},
        Module,
    },
};

#[derive(Debug)]
enum InnerError {
    Decode(String, u64),
    Exec(exec::Error),
    UnknownExport,
}

impl<E> From<binary::Error<E>> for InnerError
where
    E: fmt::Display,
{
    fn from(value: binary::Error<E>) -> Self {
        Self::Decode(value.to_string(), value.pos())
    }
}

impl From<exec::Error> for InnerError {
    fn from(value: exec::Error) -> Self {
        Self::Exec(value)
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

impl From<exec::Error> for Error {
    fn from(value: exec::Error) -> Self {
        Self {
            inner: InnerError::Exec(value),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            InnerError::Decode(error, _) => f.write_str(error),
            InnerError::Exec(error) => fmt::Display::fmt(error, f),
            InnerError::UnknownExport => f.write_str("unknown export"),
        }
    }
}

#[cfg(feature = "std")]
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.inner {
            InnerError::Decode(_, _) | InnerError::UnknownExport => None,
            InnerError::Exec(error) => Some(error),
        }
    }
}

/// Returns an empty store.
#[must_use]
pub const fn store_init() -> Store {
    Store::new()
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

/// Return the value (address) of an export.
///
/// # Errors
///
/// If there is no export with the given name
pub fn instance_export(m: &ModuleInst, name: &str) -> Result<ExternVal, Error> {
    m.exports()
        .iter()
        .find_map(|e| (e.name == name).then_some(e.value))
        .ok_or(Error::from(InnerError::UnknownExport))
}

/// Allocate a host function in the store and return the address.
pub fn func_alloc<F>(mut s: Store, ft: FuncTy, code: F) -> (Store, FuncAddr)
where
    F: FnMut(&[Val]) -> Result<Vec<Val>, HostcodeError> + 'static,
{
    let a = s.host_func_alloc(ft, code);
    (s, a)
}

/// Return the function type for a given address.
///
/// # Panics
///
/// Panics if the address is invalid.
#[must_use]
pub fn func_type(s: &Store, a: FuncAddr) -> FuncTy {
    s.func_ty(a).cloned().unwrap()
}

/// Allocate a table in the store and return the address.
#[must_use]
pub fn table_alloc(mut s: Store, ty: TableTy, r: Ref) -> (Store, TableAddr) {
    let a = s.table_alloc(ty, r);
    (s, a)
}

/// Return the table type for a given address.
///
/// # Panics
///
/// Panics if the address is invalid.
#[must_use]
pub fn table_type(s: &Store, a: TableAddr) -> TableTy {
    s.table_ty(a).unwrap()
}

/// Read the value in a table given an address and index.
///
/// # Errors
///
/// Return an error if the index is invalid.
pub fn table_read(s: &Store, a: TableAddr, i: u32) -> Result<Ref, Error> {
    Ok(s.table_read(a, i)?)
}

/// Write the given value in a table given an address and index.
///
/// # Errors
///
/// Return an error if the index is invalid.
pub fn table_write(mut s: Store, a: TableAddr, i: u32, r: Ref) -> Result<Store, Error> {
    s.table_write(a, i, r)?;
    Ok(s)
}

/// Return the size of a table.
#[must_use]
pub fn table_size(s: &Store, a: TableAddr) -> u32 {
    s.table_size(a)
}

/// Grows the size of a table.
///
/// # Errors
///
/// If the address is invalid or if n is less than the current table size.
pub fn table_grow(mut s: Store, a: TableAddr, n: u32, r: Ref) -> Result<Store, Error> {
    s.table_grow(a, n, r)?;
    Ok(s)
}

/// Allocate memory in the store and return the address.
#[must_use]
pub fn mem_alloc(mut s: Store, ty: MemTy) -> (Store, MemAddr) {
    let a = s.mem_alloc(ty);
    (s, a)
}

/// Return the memory type for a given address.
///
/// # Panics
///
/// Panics if the address is invalid.
#[must_use]
pub fn mem_type(s: &Store, a: MemAddr) -> MemTy {
    s.mem_ty(a).unwrap()
}

/// Read the value in a memory given an address and index.
///
/// # Errors
///
/// Return an error if the index is invalid.
pub fn mem_read(s: &Store, a: MemAddr, i: u32) -> Result<u8, Error> {
    Ok(s.mem_read(a, i)?)
}

/// Write the given value in a memory given an address and index.
///
/// # Errors
///
/// Return an error if the index is invalid.
pub fn mem_write(mut s: Store, a: MemAddr, i: u32, b: u8) -> Result<Store, Error> {
    s.mem_write(a, i, b)?;
    Ok(s)
}

/// Return the size of a memory.
#[must_use]
pub fn mem_size(s: &Store, a: MemAddr) -> u32 {
    s.mem_size(a)
}

/// Grows the size of a memory.
///
/// # Errors
///
/// If the address is invalid or if n is less than the current table size.
pub fn mem_grow(mut s: Store, a: MemAddr, n: u32) -> Result<Store, Error> {
    s.mem_grow(a, n)?;
    Ok(s)
}

/// Allocate a global in the store and return the address.
#[must_use]
pub fn global_alloc(mut s: Store, ty: GlobalTy, val: Val) -> (Store, GlobalAddr) {
    let a = s.global_alloc(ty, val);
    (s, a)
}

/// Return the global type for a given address.
///
/// # Panics
///
/// Panics if the address is invalid.
#[must_use]
pub fn global_type(s: &Store, a: GlobalAddr) -> GlobalTy {
    s.global_ty(a).unwrap()
}

/// Read the value in a global given an address.
///
/// # Panics
///
/// Panics if the address is invalid.
#[must_use]
pub fn global_read(s: &Store, a: GlobalAddr) -> Val {
    s.global_read(a).unwrap()
}

/// Write the given value in a global given an address.
///
/// # Errors
///
/// Return an error if the index is invalid.
pub fn global_write(mut s: Store, a: GlobalAddr, val: Val) -> Result<Store, Error> {
    s.global_write(a, val)?;
    Ok(s)
}
