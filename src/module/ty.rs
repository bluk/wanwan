//! Types describe the kind of objects in a program.

#![allow(clippy::module_name_repetitions)]

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::exec::val::{self, Num, Ref, Val};

/// Number type
///
/// Used to represent function parameter types and return types, local and
/// global variable types, and block parameter and return types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumTy {
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 32-bit floating point number
    F32,
    /// 64-bit floating point number
    F64,
}

impl NumTy {
    pub(crate) const fn default_value(self) -> Num {
        match self {
            NumTy::I32 => Num::I32(0),
            NumTy::I64 => Num::I64(0),
            NumTy::F32 => Num::F32(0.0),
            NumTy::F64 => Num::F64(0.0),
        }
    }
}

/// Vector type
///
/// Used to represent function parameter types and return types, local and
/// global variable types, and block parameter and return types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecTy {
    /// 128-bit vector of packed data
    V128,
}

impl VecTy {
    pub(crate) const fn default_value(self) -> val::Vec {
        match self {
            VecTy::V128 => val::Vec::V128(0),
        }
    }
}

/// Reference type
///
/// Used to declare what type of reference type a `Table` stores and what
/// type of reference an `Element` segment has.
///
/// Used to represent function parameter types and return types, local and
/// global variable types, and block parameter and return types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefTy {
    /// Function reference
    FuncRef,
    /// Embedder external reference
    ExternRef,
}

impl RefTy {
    pub(crate) const fn default_value(self) -> Ref {
        Ref::Null(self)
    }
}

/// Value type
///
/// Used to represent function parameter types and return types, local and
/// global variable types, and block parameter and return types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValTy {
    /// Number type
    Num(NumTy),
    /// Vector type
    Vec(VecTy),
    /// Reference type
    Ref(RefTy),
}

impl ValTy {
    pub(crate) const fn default_value(self) -> Val {
        match self {
            ValTy::Num(n) => Val::Num(n.default_value()),
            ValTy::Vec(v) => Val::Vec(v.default_value()),
            ValTy::Ref(r) => Val::Ref(r.default_value()),
        }
    }
}

/// Used to list the types for function parameters and results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultTy(pub Vec<ValTy>);

/// Function signatures consisting of a parameter type list and a result type
/// list.
///
/// Functions may contain the same signature but with different function names
/// and implementations. Due to the limited number of types in Wasm, there is a
/// a good possibilty that function types are re-used in a module.
///
/// Listed in the [`SectionId::Type`] section. Indexed by [`TypeIndex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncTy {
    /// Parameter types
    rt1: ResultTy,
    /// Result types
    rt2: ResultTy,
}

impl FuncTy {
    #[inline]
    #[must_use]
    pub fn new(params: ResultTy, ret: ResultTy) -> Self {
        Self {
            rt1: params,
            rt2: ret,
        }
    }

    /// Returns the param value types.
    #[inline]
    #[must_use]
    pub fn params(&self) -> &[ValTy] {
        &self.rt1.0
    }

    /// Returns the return value types.
    #[inline]
    #[must_use]
    pub fn ret(&self) -> &[ValTy] {
        &self.rt2.0
    }
}

#[must_use]
pub(crate) fn is_compatible(expected: &[ValTy], actual: &[Val]) -> bool {
    if expected.len() != actual.len() {
        return false;
    }

    expected
        .iter()
        .copied()
        .zip(actual.iter().map(|val| ValTy::from(*val)))
        .all(|(expected, actual)| expected == actual)
}

/// Minimum and optional maximum limits of `Memory` and `Table`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Minimum
    pub min: u32,
    /// Optional maximum. If no maximum, the size is unlimited.
    pub max: Option<u32>,
}

/// Memory type describes the limits of the linear memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemTy {
    /// Units are in page size
    pub lim: Limits,
}

/// Table type describes the type of element stored in the table and the size limit of the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableTy {
    /// Element reference type
    pub elem_ty: RefTy,
    /// Units are number of entries
    pub lim: Limits,
}

/// Mutablity of a variable described by [`GlobalTy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mut {
    /// Immutable
    Const,
    /// Mutable
    Var,
}

/// Global type used in a [`super::Global`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalTy {
    /// Mutablity
    pub m: Mut,
    /// Value type
    pub t: ValTy,
}

/// Type information for an [`super::Import`] or [`super::Export`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternTy {
    Func(FuncTy),
    Table(TableTy),
    Mem(MemTy),
    Global(GlobalTy),
}
