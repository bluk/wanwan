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

impl From<Num> for NumTy {
    fn from(value: Num) -> Self {
        match value {
            Num::I32(_) => NumTy::I32,
            Num::I64(_) => NumTy::I64,
            Num::F32(_) => NumTy::F32,
            Num::F64(_) => NumTy::F64,
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

impl From<Vec> for VecTy {
    fn from(value: Vec) -> Self {
        match value {
            Vec::V128(_) => VecTy::V128,
        }
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

impl From<Ref> for RefTy {
    fn from(value: Ref) -> Self {
        match value {
            Ref::Null(ty) => ty,
            Ref::Func(_) => RefTy::FuncRef,
            Ref::Extern(_) => RefTy::ExternRef,
        }
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

impl From<Num> for Val {
    fn from(value: Num) -> Self {
        Self::Num(value)
    }
}

impl From<Vec> for Val {
    fn from(value: Vec) -> Self {
        Self::Vec(value)
    }
}

impl From<Ref> for Val {
    fn from(value: Ref) -> Self {
        Self::Ref(value)
    }
}

impl From<Val> for ValTy {
    fn from(value: Val) -> Self {
        match value {
            Val::Num(n) => ValTy::Num(NumTy::from(n)),
            Val::Vec(v) => ValTy::Vec(VecTy::from(v)),
            Val::Ref(r) => ValTy::Ref(RefTy::from(r)),
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

impl From<FuncAddr> for ExternVal {
    fn from(value: FuncAddr) -> Self {
        Self::Func(value)
    }
}

impl From<TableAddr> for ExternVal {
    fn from(value: TableAddr) -> Self {
        Self::Table(value)
    }
}

impl From<MemAddr> for ExternVal {
    fn from(value: MemAddr) -> Self {
        Self::Mem(value)
    }
}

impl From<GlobalAddr> for ExternVal {
    fn from(value: GlobalAddr) -> Self {
        Self::Global(value)
    }
}
