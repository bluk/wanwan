//! Abstract model of a WebAssembly module.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{string::String, vec::Vec};
#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

use self::{
    instr::{ConstExpr, Expr},
    ty::{FuncTy, GlobalTy, MemoryTy, RefTy, TableTy, ValTy},
};

pub mod instr;
pub mod ty;

/// Index for a [`FuncTy`] in the [`SectionId::Type`] section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeIndex(pub u32);

/// Index for a global variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportGlobalIndex(pub u32);

/// Index for a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FuncIndex(pub u32);

/// Index for a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableIndex(pub u32);

/// Index for memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemIndex(pub u32);

/// Index for a global variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalIndex(pub u32);

/// Index for an element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementIndex(pub u32);

/// Index for a data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataIndex(pub u32);

/// Index for a local variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalIndex(pub u32);

/// Index for a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelIndex(pub u32);

/// Functions
#[derive(Debug, Clone, PartialEq)]
pub struct Func {
    pub ty: TypeIndex,
    pub locals: Vec<ValTy>,
    pub body: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Table {
    pub ty: TableTy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mem {
    pub ty: MemoryTy,
}

/// Global
#[derive(Debug, Clone, PartialEq)]
pub struct Global {
    pub ty: GlobalTy,
    pub init: ConstExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ElementSegmentMode {
    /// Elements can be copied via [`Instr::TableInit`] instruction.
    Passive,
    /// Elements can be copied during instantiation.
    Active(TableIndex, ConstExpr),
    /// Used with [`Instr::RefFunc`].
    Declarative,
}

/// Initializes a table.
#[derive(Debug, Clone, PartialEq)]
pub struct Elem {
    pub ty: RefTy,
    pub init: Vec<ConstExpr>,
    pub mode: ElementSegmentMode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataMode {
    Passive,
    Active(MemIndex, ConstExpr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Data {
    pub init: Vec<u8>,
    pub mode: DataMode,
}

/// Export description
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportDesc {
    /// Function index
    Func(FuncIndex),
    /// Table type
    Table(TableIndex),
    /// Memory type
    Mem(MemIndex),
    /// Global type
    Global(GlobalIndex),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Export {
    pub name: String,
    pub desc: ExportDesc,
}

/// Import description
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportDesc {
    /// Function index
    Func(TypeIndex),
    /// Table type
    Table(TableTy),
    /// Memory type
    Mem(MemoryTy),
    /// Global type
    Global(GlobalTy),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    /// Module
    pub module: String,
    /// Name
    pub name: String,
    /// Description
    pub desc: ImportDesc,
}

/// Decoded module
#[derive(Debug)]
pub struct Module {
    /// Function types
    pub types: Vec<FuncTy>,
    /// Functions
    pub funcs: Vec<Func>,
    /// Tables
    pub tables: Vec<Table>,
    /// Memory
    pub mems: Vec<Mem>,
    /// Globals
    pub globals: Vec<Global>,
    /// Element segments
    pub elems: Vec<Elem>,
    /// Data
    pub datas: Vec<Data>,
    /// Start function index
    pub start: Option<FuncIndex>,
    /// Imports
    pub imports: Vec<Import>,
    /// Exports
    pub exports: Vec<Export>,
}
