//! Values

use crate::module::ty::{NumTy, RefTy, ValTy, VecTy};

use super::{ExternAddr, FuncAddr, GlobalAddr, MemAddr, TableAddr};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl From<i32> for Num {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

impl From<i64> for Num {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<f32> for Num {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<f64> for Num {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl From<bool> for Num {
    fn from(value: bool) -> Self {
        if value {
            Self::I32(1)
        } else {
            Self::I32(0)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vec {
    V128(i128),
}

impl From<i128> for Vec {
    fn from(value: i128) -> Self {
        Self::V128(value)
    }
}

impl Default for Vec {
    fn default() -> Self {
        Self::V128(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ref {
    Null(RefTy),
    Func(FuncAddr),
    Extern(ExternAddr),
}

impl From<FuncAddr> for Ref {
    fn from(value: FuncAddr) -> Self {
        Self::Func(value)
    }
}

impl From<ExternAddr> for Ref {
    fn from(value: ExternAddr) -> Self {
        Self::Extern(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Val {
    Num(Num),
    Vec(Vec),
    Ref(Ref),
}

impl From<i32> for Val {
    fn from(value: i32) -> Self {
        Self::Num(Num::I32(value))
    }
}

impl From<i64> for Val {
    fn from(value: i64) -> Self {
        Self::Num(Num::I64(value))
    }
}

impl From<f32> for Val {
    fn from(value: f32) -> Self {
        Self::Num(Num::F32(value))
    }
}

impl From<f64> for Val {
    fn from(value: f64) -> Self {
        Self::Num(Num::F64(value))
    }
}

impl From<bool> for Val {
    fn from(value: bool) -> Self {
        if value {
            Self::Num(Num::I32(1))
        } else {
            Self::Num(Num::I32(0))
        }
    }
}

impl From<i128> for Val {
    fn from(value: i128) -> Self {
        Self::Vec(Vec::V128(value))
    }
}

impl From<FuncAddr> for Val {
    fn from(value: FuncAddr) -> Self {
        Self::Ref(Ref::Func(value))
    }
}

impl From<ExternAddr> for Val {
    fn from(value: ExternAddr) -> Self {
        Self::Ref(Ref::Extern(value))
    }
}

impl From<Val> for ValTy {
    fn from(value: Val) -> Self {
        match value {
            Val::Num(n) => match n {
                Num::I32(_) => ValTy::Num(NumTy::I32),
                Num::I64(_) => ValTy::Num(NumTy::I64),
                Num::F32(_) => ValTy::Num(NumTy::F32),
                Num::F64(_) => ValTy::Num(NumTy::F64),
            },
            Val::Vec(v) => match v {
                Vec::V128(_) => ValTy::Vec(VecTy::V128),
            },
            Val::Ref(r) => match r {
                Ref::Null(r) => ValTy::Ref(r),
                Ref::Func(_) => ValTy::Ref(RefTy::FuncRef),
                Ref::Extern(_) => ValTy::Ref(RefTy::ExternRef),
            },
        }
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy)]
pub enum ExternVal {
    Func(FuncAddr),
    Table(TableAddr),
    Mem(MemAddr),
    Global(GlobalAddr),
}
