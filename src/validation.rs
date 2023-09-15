//! Validation for a module.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{fmt, vec, vec::Vec};
#[cfg(feature = "std")]
use std::{error, fmt, vec, vec::Vec};

use crate::module::{
    ty::{self, FuncTy, GlobalTy, MemoryTy, NumTy, RefTy, TableTy, ValTy, VecTy},
    DataIndex, Elem, ElementIndex, FuncIndex, Global, GlobalIndex, ImportGlobalIndex, LabelIndex,
    LocalIndex, Mem, MemIndex, Table, TableIndex, TypeIndex,
};

/// Value type
///
/// Used to represent function parameter types and return types, local and
/// global variable types, and block parameter and return types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpdTy {
    /// Number type
    Num(NumTy),
    /// Vector type
    Vec(VecTy),
    /// Reference type
    Ref(RefTy),
    /// Unknown type
    Unknown,
}

impl From<ValTy> for OpdTy {
    fn from(value: ValTy) -> Self {
        match value {
            ty::ValTy::Num(n) => OpdTy::Num(n),
            ty::ValTy::Vec(v) => OpdTy::Vec(v),
            ty::ValTy::Ref(r) => OpdTy::Ref(r),
        }
    }
}

impl OpdTy {
    /// Return true if the type is a numeric type, false otherwise.
    #[inline]
    #[must_use]
    pub(crate) fn is_num(self) -> bool {
        match self {
            OpdTy::Num(_) | OpdTy::Unknown => true,
            OpdTy::Vec(_) | OpdTy::Ref(_) => false,
        }
    }

    /// Return true if the type is a vector type, false otherwise.
    #[inline]
    #[must_use]
    pub(crate) fn is_vec(self) -> bool {
        match self {
            OpdTy::Vec(_) | OpdTy::Unknown => true,
            OpdTy::Num(_) | OpdTy::Ref(_) => false,
        }
    }

    /// Return true if the type is a refrence type, false otherwise.
    #[inline]
    #[must_use]
    pub(crate) fn is_ref(self) -> bool {
        match self {
            OpdTy::Ref(_) | OpdTy::Unknown => true,
            OpdTy::Num(_) | OpdTy::Vec(_) => false,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CtrlFrame {
    pub(crate) op_code: u8,
    pub(crate) start_tys: Vec<OpdTy>,
    pub(crate) end_tys: Vec<OpdTy>,
    pub(crate) height: usize,
    pub(crate) unreachable: bool,
}

impl CtrlFrame {
    fn label_tys(&self) -> &[OpdTy] {
        const LOOP_OP_CODE: u8 = 0x03;
        if self.op_code == LOOP_OP_CODE {
            &self.start_tys
        } else {
            &self.end_tys
        }
    }
}

type ValStack = Vec<OpdTy>;
type CtrlStack = Vec<CtrlFrame>;

#[derive(Debug, PartialEq, Eq)]
pub enum ExprError {
    /// Underflow current block
    Underflow,
    /// Popped type does not match expectations
    UnexpectedTy,
    /// Popped too many frames
    EmptyFrames,
    InvalidStack,
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprError::Underflow => f.write_str("underflow"),
            ExprError::UnexpectedTy => f.write_str("unexpected type popped"),
            ExprError::EmptyFrames => f.write_str("empty frames"),
            ExprError::InvalidStack => f.write_str("invalid value stack"),
        }
    }
}

#[cfg(feature = "std")]
impl error::Error for ExprError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

#[derive(Debug, Default)]
pub(crate) struct ExprContext {
    vals: ValStack,
    ctrls: CtrlStack,
}

impl ExprContext {
    #[inline]
    #[must_use]
    fn ctrl_frames_len(&self) -> usize {
        self.ctrls.len()
    }

    fn push_val(&mut self, ty: OpdTy) {
        self.vals.push(ty);
    }

    fn pop_val(&mut self) -> Result<OpdTy, ExprError> {
        let cur_frame = self.ctrls.last().unwrap();
        if self.vals.len() == cur_frame.height {
            if cur_frame.unreachable {
                return Ok(OpdTy::Unknown);
            }
            return Err(ExprError::Underflow);
        }
        Ok(self.vals.pop().unwrap())
    }

    fn pop_expect_val(&mut self, expect: OpdTy) -> Result<OpdTy, ExprError> {
        let actual = self.pop_val()?;
        match (actual, expect) {
            (OpdTy::Num(_) | OpdTy::Unknown, OpdTy::Num(_))
            | (OpdTy::Vec(_) | OpdTy::Unknown, OpdTy::Vec(_))
            | (OpdTy::Ref(_) | OpdTy::Unknown, OpdTy::Ref(_))
            | (OpdTy::Num(_) | OpdTy::Vec(_) | OpdTy::Ref(_) | OpdTy::Unknown, OpdTy::Unknown) => {
                Ok(actual)
            }
            (OpdTy::Num(_) | OpdTy::Ref(_), OpdTy::Vec(_))
            | (OpdTy::Num(_) | OpdTy::Vec(_), OpdTy::Ref(_))
            | (OpdTy::Vec(_) | OpdTy::Ref(_), OpdTy::Num(_)) => Err(ExprError::UnexpectedTy),
        }
    }

    fn push_vals(&mut self, tys: &[OpdTy]) {
        for ty in tys {
            self.push_val(*ty);
        }
    }

    fn pop_expect_vals(&mut self, tys: &[OpdTy]) -> Result<Vec<OpdTy>, ExprError> {
        let mut popped = Vec::with_capacity(tys.len());
        for ty in tys.iter().rev() {
            popped.insert(0, self.pop_expect_val(*ty)?);
        }
        Ok(popped)
    }

    fn push_ctrl(&mut self, op_code: u8, start_tys: Vec<OpdTy>, end_tys: Vec<OpdTy>) {
        self.push_vals(&start_tys);
        let frame = CtrlFrame {
            op_code,
            start_tys,
            end_tys,
            height: self.vals.len(),
            unreachable: false,
        };
        self.ctrls.push(frame);
    }

    fn pop_ctrl(&mut self) -> Result<CtrlFrame, ExprError> {
        let Some(frame) = self.ctrls.last() else {
            return Err(ExprError::EmptyFrames);
        };
        let height = frame.height;
        let end_tys = frame.end_tys.clone();
        self.pop_expect_vals(&end_tys)?;
        if self.vals.len() != height {
            return Err(ExprError::InvalidStack);
        }
        let frame = self.ctrls.pop().unwrap();
        Ok(frame)
    }

    fn unreachable(&mut self) -> Result<(), ExprError> {
        let Some(frame) = self.ctrls.last_mut() else {
            return Err(ExprError::EmptyFrames);
        };
        if self.vals.len() < frame.height {
            return Err(ExprError::InvalidStack);
        }
        self.vals.truncate(frame.height);
        frame.unreachable = true;
        Ok(())
    }

    fn label_tys(&self, label: LabelIndex) -> &[OpdTy] {
        // TODO: Potential crash here if label is too much
        let frame = &self.ctrls[self.ctrls.len() - usize::try_from(label.0).unwrap() - 1];
        frame.label_tys()
    }
}

#[derive(Debug)]
pub(crate) struct FuncExprValidator {
    ctx: ExprContext,
}

impl FuncExprValidator {
    #[inline]
    #[must_use]
    pub(crate) fn new(func_ty: &FuncTy, locals: &[ValTy]) -> Self {
        let mut ctx = ExprContext::default();
        // TODO: Opcode is wrong
        let mut new_locals = func_ty
            .rt1
            .0
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<OpdTy>>();
        new_locals.extend(locals.iter().copied().map(OpdTy::from));
        ctx.push_ctrl(
            0,
            new_locals,
            func_ty
                .rt2
                .0
                .iter()
                .copied()
                .map(OpdTy::from)
                .collect::<Vec<_>>(),
        );

        Self { ctx }
    }

    #[inline]
    #[must_use]
    pub(crate) fn ctrl_frames_len(&self) -> usize {
        self.ctx.ctrl_frames_len()
    }

    #[inline]
    pub(crate) fn local_idx(&self, x: LocalIndex) -> Option<OpdTy> {
        let frame = self.ctx.ctrls.get(0)?;
        frame.start_tys.get(usize::try_from(x.0).unwrap()).copied()
    }

    #[inline]
    pub(crate) fn ret(&mut self) -> Result<(), ExprError> {
        let Some(frame) = self.ctx.ctrls.get(0) else {
            panic!();
        };

        let ret_tys = frame.end_tys.clone();

        self.pop_expect_vals(&ret_tys)?;
        self.unreachable()?;

        Ok(())
    }

    #[inline]
    pub(crate) fn push_val(&mut self, ty: OpdTy) {
        self.ctx.push_val(ty);
    }

    #[inline]
    pub(crate) fn pop_val(&mut self) -> Result<OpdTy, ExprError> {
        self.ctx.pop_val()
    }

    #[inline]
    pub(crate) fn pop_expect_val(&mut self, expect: OpdTy) -> Result<OpdTy, ExprError> {
        self.ctx.pop_expect_val(expect)
    }

    #[inline]
    pub(crate) fn push_vals(&mut self, tys: &[OpdTy]) {
        self.ctx.push_vals(tys);
    }

    #[inline]
    pub(crate) fn pop_expect_vals(&mut self, tys: &[OpdTy]) -> Result<Vec<OpdTy>, ExprError> {
        self.ctx.pop_expect_vals(tys)
    }

    #[inline]
    pub(crate) fn push_ctrl(&mut self, op_code: u8, start_tys: Vec<OpdTy>, end_tys: Vec<OpdTy>) {
        self.ctx.push_ctrl(op_code, start_tys, end_tys);
    }

    #[inline]
    pub(crate) fn pop_ctrl(&mut self) -> Result<CtrlFrame, ExprError> {
        self.ctx.pop_ctrl()
    }

    #[inline]
    pub(crate) fn unreachable(&mut self) -> Result<(), ExprError> {
        self.ctx.unreachable()
    }

    #[inline]
    #[must_use]
    pub(crate) fn label_tys(&self, label: LabelIndex) -> &[OpdTy] {
        self.ctx.label_tys(label)
    }
}

#[derive(Debug)]
pub(crate) struct ConstExprValidator {
    ctx: ExprContext,
}

impl ConstExprValidator {
    #[inline]
    #[must_use]
    pub(crate) fn new(expected_ty: OpdTy) -> Self {
        let mut ctx = ExprContext::default();
        ctx.push_ctrl(0x02, Vec::new(), vec![expected_ty]);

        Self { ctx }
    }

    #[inline]
    pub(crate) fn push_val(&mut self, ty: OpdTy) {
        self.ctx.push_val(ty);
    }

    #[inline]
    pub(crate) fn push_vals(&mut self, tys: &[OpdTy]) {
        self.ctx.push_vals(tys);
    }

    #[inline]
    pub(crate) fn pop_ctrl(&mut self) -> Result<CtrlFrame, ExprError> {
        self.ctx.pop_ctrl()
    }
}

pub(crate) trait TypesContext {
    fn is_type_valid(&self, idx: TypeIndex) -> bool;

    fn func_ty(&self, idx: TypeIndex) -> Option<&FuncTy>;
}

macro_rules! impl_types_context {
    ($name:ty) => {
        impl<'a> TypesContext for $name {
            fn is_type_valid(&self, idx: TypeIndex) -> bool {
                usize::try_from(idx.0).unwrap() < self.types.len()
            }

            fn func_ty(&self, idx: TypeIndex) -> Option<&FuncTy> {
                self.types.get(usize::try_from(idx.0).unwrap())
            }
        }
    };
}

pub(crate) trait ImportGlobalsContext {
    fn import_global_ty(&self, idx: ImportGlobalIndex) -> Option<&GlobalTy>;
}

macro_rules! impl_import_globals_context {
    ($name:ty) => {
        impl<'a> ImportGlobalsContext for $name {
            fn import_global_ty(&self, idx: ImportGlobalIndex) -> Option<&GlobalTy> {
                self.import_globals.get(usize::try_from(idx.0).unwrap())
            }
        }
    };
}

pub(crate) trait FunctionsContext {
    fn imported_funcs_len(&self) -> usize;

    fn is_func_valid(&self, idx: FuncIndex) -> bool;

    fn type_index(&self, idx: FuncIndex) -> Option<TypeIndex>;
}

macro_rules! impl_funcs_context {
    ($name:ty) => {
        impl<'a> FunctionsContext for $name {
            fn imported_funcs_len(&self) -> usize {
                self.import_funcs.len()
            }

            fn is_func_valid(&self, idx: FuncIndex) -> bool {
                usize::try_from(idx.0).unwrap() < self.import_funcs.len() + self.type_idxs.len()
            }

            fn type_index(&self, idx: FuncIndex) -> Option<TypeIndex> {
                let idx = usize::try_from(idx.0).unwrap();
                if idx < self.import_funcs.len() {
                    return self.import_funcs.get(idx).copied();
                }

                let idx = idx - self.import_funcs.len();
                self.type_idxs.get(idx).copied()
            }
        }
    };
}

pub(crate) trait TablesContext {
    fn is_table_valid(&self, idx: TableIndex) -> bool;

    fn table_ty(&self, idx: TableIndex) -> Option<TableTy>;
}

macro_rules! impl_tables_context {
    ($name:ty) => {
        impl<'a> TablesContext for $name {
            fn is_table_valid(&self, idx: TableIndex) -> bool {
                usize::try_from(idx.0).unwrap() < self.import_tables.len() + self.tables.len()
            }

            fn table_ty(&self, idx: TableIndex) -> Option<TableTy> {
                let idx = usize::try_from(idx.0).unwrap();
                if idx < self.import_tables.len() {
                    return self.import_tables.get(idx).copied();
                }

                let idx = idx - self.import_tables.len();
                self.tables.get(idx).map(|t| t.ty)
            }
        }
    };
}

pub(crate) trait MemsContext {
    fn is_mem_valid(&self, idx: MemIndex) -> bool;
}

macro_rules! impl_mems_context {
    ($name:ty) => {
        impl<'a> MemsContext for $name {
            fn is_mem_valid(&self, idx: MemIndex) -> bool {
                usize::try_from(idx.0).unwrap() < self.import_mems.len() + self.mems.len()
            }
        }
    };
}

pub(crate) trait GlobalsContext {
    fn is_global_valid(&self, idx: GlobalIndex) -> bool;

    fn global_ty(&self, idx: GlobalIndex) -> Option<&GlobalTy>;
}

macro_rules! impl_globals_context {
    ($name:ty) => {
        impl<'a> GlobalsContext for $name {
            fn is_global_valid(&self, idx: GlobalIndex) -> bool {
                usize::try_from(idx.0).unwrap() < self.import_globals.len() + self.globals.len()
            }

            fn global_ty(&self, idx: GlobalIndex) -> Option<&GlobalTy> {
                let idx = usize::try_from(idx.0).unwrap();
                if idx < self.import_globals.len() {
                    return self.import_globals.get(idx);
                }

                let idx = idx - self.import_globals.len();
                let g = self.globals.get(idx)?;
                Some(&g.ty)
            }
        }
    };
}

pub(crate) trait ElementsContext {
    fn is_elem_valid(&self, idx: ElementIndex) -> bool;

    fn elem(&self, idx: ElementIndex) -> Option<&Elem>;
}

macro_rules! impl_elems_context {
    ($name:ty) => {
        impl<'a> ElementsContext for $name {
            fn is_elem_valid(&self, idx: ElementIndex) -> bool {
                usize::try_from(idx.0).unwrap() < self.elems.len()
            }

            fn elem(&self, idx: ElementIndex) -> Option<&Elem> {
                let idx = usize::try_from(idx.0).unwrap();
                let elem = self.elems.get(idx)?;
                Some(elem)
            }
        }
    };
}

pub(crate) trait DataContext {
    fn is_data_valid(&self, idx: DataIndex) -> bool;
}

macro_rules! impl_data_context {
    ($name:ty) => {
        impl<'a> DataContext for $name {
            fn is_data_valid(&self, idx: DataIndex) -> bool {
                let Some(data_len) = self.data_len else {
                    return true;
                };
                idx.0 < data_len
            }
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ImportSectionValidator<'a> {
    pub types: &'a [FuncTy],
}

impl_types_context!(ImportSectionValidator<'a>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct FunctionSectionValidator<'a> {
    pub types: &'a [FuncTy],
}

impl_types_context!(FunctionSectionValidator<'a>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct GlobalSectionValidator<'a> {
    pub import_funcs: &'a [TypeIndex],
    pub import_globals: &'a [GlobalTy],
    pub type_idxs: &'a [TypeIndex],
}

impl_import_globals_context!(GlobalSectionValidator<'a>);
impl_funcs_context!(GlobalSectionValidator<'a>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExportsSectionValidator<'a> {
    pub import_funcs: &'a [TypeIndex],
    pub import_tables: &'a [TableTy],
    pub import_mems: &'a [MemoryTy],
    pub import_globals: &'a [GlobalTy],
    pub type_idxs: &'a [TypeIndex],
    pub tables: &'a [Table],
    pub mems: &'a [Mem],
    pub globals: &'a [Global],
}

impl_funcs_context!(ExportsSectionValidator<'a>);
impl_tables_context!(ExportsSectionValidator<'a>);
impl_mems_context!(ExportsSectionValidator<'a>);
impl_globals_context!(ExportsSectionValidator<'a>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct StartSectionValidator<'a> {
    pub types: &'a [FuncTy],
    pub import_funcs: &'a [TypeIndex],
    pub type_idxs: &'a [TypeIndex],
}

impl_types_context!(StartSectionValidator<'a>);
impl_funcs_context!(StartSectionValidator<'a>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct ElementsSectionValidator<'a> {
    pub import_funcs: &'a [TypeIndex],
    pub import_tables: &'a [TableTy],
    pub import_globals: &'a [GlobalTy],
    pub type_idxs: &'a [TypeIndex],
    pub tables: &'a [Table],
}

impl_import_globals_context!(ElementsSectionValidator<'a>);
impl_funcs_context!(ElementsSectionValidator<'a>);
impl_tables_context!(ElementsSectionValidator<'a>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct CodeSectionValidator<'a> {
    pub types: &'a [FuncTy],
    pub import_funcs: &'a [TypeIndex],
    pub import_mems: &'a [MemoryTy],
    pub import_tables: &'a [TableTy],
    pub import_globals: &'a [GlobalTy],
    pub type_idxs: &'a [TypeIndex],
    pub tables: &'a [Table],
    pub mems: &'a [Mem],
    pub globals: &'a [Global],
    pub elems: &'a [Elem],
    pub data_len: Option<u32>,
}

impl_types_context!(CodeSectionValidator<'a>);
impl_funcs_context!(CodeSectionValidator<'a>);
impl_tables_context!(CodeSectionValidator<'a>);
impl_mems_context!(CodeSectionValidator<'a>);
impl_globals_context!(CodeSectionValidator<'a>);
impl_elems_context!(CodeSectionValidator<'a>);
impl_data_context!(CodeSectionValidator<'a>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct DataSectionValidator<'a> {
    pub import_funcs: &'a [TypeIndex],
    pub import_mems: &'a [MemoryTy],
    pub import_globals: &'a [GlobalTy],
    pub type_idxs: &'a [TypeIndex],
    pub mems: &'a [Mem],
}

impl_import_globals_context!(DataSectionValidator<'a>);
impl_funcs_context!(DataSectionValidator<'a>);
impl_mems_context!(DataSectionValidator<'a>);
