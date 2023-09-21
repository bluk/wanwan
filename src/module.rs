//! Abstract model of a WebAssembly module.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{string::String, vec::Vec};
use core::num::TryFromIntError;
#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

use self::{
    instr::{ConstExpr, Expr},
    ty::{ExternTy, FuncTy, GlobalTy, MemTy, RefTy, TableTy, ValTy},
};

pub mod instr;
pub mod ty;

macro_rules! impl_index {
    ($ty:ty) => {
        impl $ty {
            #[inline]
            #[must_use]
            pub(crate) const fn new(idx: u32) -> Self {
                Self(idx)
            }
        }

        impl From<$ty> for u32 {
            fn from(value: $ty) -> Self {
                value.0
            }
        }

        impl TryFrom<$ty> for usize {
            type Error = TryFromIntError;

            fn try_from(value: $ty) -> Result<Self, Self::Error> {
                usize::try_from(value.0)
            }
        }
    };
}

/// Index for a [`FuncTy`] in the [`Types`] section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeIndex(u32);

impl_index!(TypeIndex);

/// Index for a global variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportGlobalIndex(u32);

impl_index!(ImportGlobalIndex);

/// Index for a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FuncIndex(u32);

impl_index!(FuncIndex);

/// Index for a table.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TableIndex(u32);

impl_index!(TableIndex);

/// Index for memory.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MemIndex(u32);

impl_index!(MemIndex);

/// Index for a global variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalIndex(u32);

impl_index!(GlobalIndex);

/// Index for an element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementIndex(u32);

impl_index!(ElementIndex);

/// Index for a data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataIndex(u32);

impl_index!(DataIndex);

/// Index for a local variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalIndex(u32);

impl_index!(LocalIndex);

/// Index for a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelIndex(u32);

impl_index!(LabelIndex);

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
    pub ty: MemTy,
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
    Mem(MemTy),
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

/// Function signatures in a module
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Types(Vec<FuncTy>);

impl Types {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_types(types: Vec<FuncTy>) -> Self {
        Self(types)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[FuncTy] {
        self.0.as_slice()
    }

    /// Returns true if the type index is valid
    #[inline]
    #[must_use]
    pub fn is_index_valid(&self, idx: TypeIndex) -> bool {
        let Ok(idx) = usize::try_from(idx.0) else {
            return false;
        };
        idx < self.0.len()
    }

    /// Returns the function type (signature) given a type index.
    ///
    /// # Panics
    ///
    /// If the index is not valid, then panics.
    #[must_use]
    pub fn func_ty(&self, idx: TypeIndex) -> Option<&FuncTy> {
        self.0.get(usize::try_from(idx.0).unwrap())
    }

    #[inline]
    #[must_use]
    pub(crate) fn into_inner(self) -> Vec<FuncTy> {
        self.0
    }
}

/// Functions defined in a module
///
/// # Important
///
/// Imported functions are not included. To resolve a [`FuncIndex`], the
/// imported functions must be prepended to the functions defined in the module.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Funcs(Vec<Func>);

impl Funcs {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_funcs(funcs: Vec<Func>) -> Self {
        Self(funcs)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Func] {
        self.0.as_slice()
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Tables(Vec<Table>);

impl Tables {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_tables(tables: Vec<Table>) -> Self {
        Self(tables)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Table] {
        self.0.as_slice()
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Mems(Vec<Mem>);

impl Mems {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_mems(mems: Vec<Mem>) -> Self {
        Self(mems)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Mem] {
        self.0.as_slice()
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Globals(Vec<Global>);

impl Globals {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_globals(globals: Vec<Global>) -> Self {
        Self(globals)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Global] {
        self.0.as_slice()
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Elems(Vec<Elem>);

impl Elems {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_elems(elems: Vec<Elem>) -> Self {
        Self(elems)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Elem] {
        self.0.as_slice()
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Datas(Vec<Data>);

impl Datas {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_datas(datas: Vec<Data>) -> Self {
        Self(datas)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Data] {
        self.0.as_slice()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct StartFunc(Option<FuncIndex>);

impl StartFunc {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(None)
    }

    #[inline]
    #[must_use]
    pub const fn with_start_func(start_func: Option<FuncIndex>) -> Self {
        Self(start_func)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Imports(Vec<Import>);

impl Imports {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_imports(imports: Vec<Import>) -> Self {
        Self(imports)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Import] {
        self.0.as_slice()
    }

    pub fn funcs_iter(&self) -> impl Iterator<Item = TypeIndex> + '_ {
        self.0.iter().filter_map(|im| match im.desc {
            ImportDesc::Func(idx) => Some(idx),
            ImportDesc::Table(_) | ImportDesc::Mem(_) | ImportDesc::Global(_) => None,
        })
    }

    pub fn tables_iter(&self) -> impl Iterator<Item = TableTy> + '_ {
        self.0.iter().filter_map(|im| match im.desc {
            ImportDesc::Table(ty) => Some(ty),
            ImportDesc::Func(_) | ImportDesc::Mem(_) | ImportDesc::Global(_) => None,
        })
    }

    pub fn mems_iter(&self) -> impl Iterator<Item = MemTy> + '_ {
        self.0.iter().filter_map(|im| match im.desc {
            ImportDesc::Mem(ty) => Some(ty),
            ImportDesc::Func(_) | ImportDesc::Table(_) | ImportDesc::Global(_) => None,
        })
    }

    pub fn globals_iter(&self) -> impl Iterator<Item = GlobalTy> + '_ {
        self.0.iter().filter_map(|im| match im.desc {
            ImportDesc::Global(ty) => Some(ty),
            ImportDesc::Func(_) | ImportDesc::Table(_) | ImportDesc::Mem(_) => None,
        })
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Exports(Vec<Export>);

impl Exports {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    #[must_use]
    pub const fn with_exports(exports: Vec<Export>) -> Self {
        Self(exports)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Export] {
        self.0.as_slice()
    }
}

#[must_use]
fn type_index(imports: &Imports, funcs: &Funcs, idx: FuncIndex) -> Option<TypeIndex> {
    let idx = usize::try_from(idx.0).unwrap();
    let import_funcs_count = imports.funcs_iter().count();
    if idx < import_funcs_count {
        return imports.funcs_iter().nth(idx);
    }

    let idx = idx - import_funcs_count;
    funcs.as_slice().get(idx).map(|f| f.ty)
}

#[must_use]
pub(crate) fn table_ty(imports: &Imports, tables: &Tables, idx: TableIndex) -> Option<TableTy> {
    let idx = usize::try_from(idx.0).unwrap();
    let import_tables_count = imports.tables_iter().count();
    if idx < import_tables_count {
        return imports.tables_iter().nth(idx);
    }

    let idx = idx - import_tables_count;
    tables.as_slice().get(idx).map(|t| t.ty)
}

#[must_use]
pub(crate) fn mem_ty(imports: &Imports, mems: &Mems, idx: MemIndex) -> Option<MemTy> {
    let idx = usize::try_from(idx.0).unwrap();
    let import_mems_count = imports.mems_iter().count();
    if idx < import_mems_count {
        return imports.mems_iter().nth(idx);
    }

    let idx = idx - import_mems_count;
    mems.as_slice().get(idx).map(|m| m.ty)
}

#[must_use]
pub(crate) fn global_ty(
    imports: &Imports,
    globals: &Globals,
    idx: GlobalIndex,
) -> Option<GlobalTy> {
    let idx = usize::try_from(idx.0).unwrap();
    let import_globals_count = imports.globals_iter().count();
    if idx < import_globals_count {
        return imports.globals_iter().nth(idx);
    }

    let idx = idx - import_globals_count;
    globals.as_slice().get(idx).map(|g| g.ty)
}

/// Decoded module
#[derive(Debug)]
pub struct Module {
    pub(crate) types: Types,
    funcs: Funcs,
    tables: Tables,
    mems: Mems,
    globals: Globals,
    elems: Elems,
    datas: Datas,
    start: StartFunc,
    imports: Imports,
    exports: Exports,
}

impl Module {
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[must_use]
    pub(crate) const fn new(
        types: Types,
        funcs: Funcs,
        tables: Tables,
        mems: Mems,
        globals: Globals,
        elems: Elems,
        datas: Datas,
        start: StartFunc,
        imports: Imports,
        exports: Exports,
    ) -> Self {
        Self {
            types,
            funcs,
            tables,
            mems,
            globals,
            elems,
            datas,
            start,
            imports,
            exports,
        }
    }

    #[inline]
    #[must_use]
    pub fn func_tys(&self) -> &[FuncTy] {
        self.types.as_slice()
    }

    /// Returns the function type (signature) given a type index.
    #[inline]
    #[must_use]
    pub fn func_ty(&self, idx: TypeIndex) -> Option<&FuncTy> {
        self.types.func_ty(idx)
    }

    #[inline]
    #[must_use]
    pub fn funcs(&self) -> &[Func] {
        self.funcs.as_slice()
    }

    #[must_use]
    pub fn type_index(&self, idx: FuncIndex) -> Option<TypeIndex> {
        type_index(&self.imports, &self.funcs, idx)
    }

    #[inline]
    #[must_use]
    pub fn tables(&self) -> &[Table] {
        self.tables.as_slice()
    }

    #[must_use]
    pub fn table_ty(&self, idx: TableIndex) -> Option<TableTy> {
        table_ty(&self.imports, &self.tables, idx)
    }

    #[inline]
    #[must_use]
    pub fn mems(&self) -> &[Mem] {
        self.mems.as_slice()
    }

    #[must_use]
    pub fn mem_ty(&self, idx: MemIndex) -> Option<MemTy> {
        mem_ty(&self.imports, &self.mems, idx)
    }

    #[inline]
    #[must_use]
    pub fn globals(&self) -> &[Global] {
        self.globals.as_slice()
    }

    #[must_use]
    pub fn global_ty(&self, idx: GlobalIndex) -> Option<GlobalTy> {
        global_ty(&self.imports, &self.globals, idx)
    }

    #[inline]
    #[must_use]
    pub fn elems(&self) -> &[Elem] {
        self.elems.as_slice()
    }

    #[inline]
    #[must_use]
    pub fn datas(&self) -> &[Data] {
        self.datas.as_slice()
    }

    #[inline]
    #[must_use]
    pub fn start(&self) -> Option<FuncIndex> {
        self.start.0
    }

    #[inline]
    #[must_use]
    pub fn imports(&self) -> &[Import] {
        self.imports.as_slice()
    }

    pub(crate) fn import_external_tys(&self) -> impl Iterator<Item = (&str, &str, ExternTy)> + '_ {
        self.imports().iter().map(|im| {
            let ty = match im.desc {
                ImportDesc::Func(idx) => {
                    // Index should have been validated
                    let func_ty = self.func_ty(idx).unwrap();
                    ExternTy::Func(func_ty.clone())
                }
                ImportDesc::Table(t) => ExternTy::Table(t),
                ImportDesc::Mem(m) => ExternTy::Mem(m),
                ImportDesc::Global(g) => ExternTy::Global(g),
            };

            (im.module.as_str(), im.name.as_str(), ty)
        })
    }

    #[inline]
    #[must_use]
    pub fn exports(&self) -> &[Export] {
        self.exports.as_slice()
    }

    pub(crate) fn export_external_tys(&self) -> impl Iterator<Item = (&str, ExternTy)> + '_ {
        self.exports().iter().map(|ex| {
            let ty = match ex.desc {
                ExportDesc::Func(idx) => {
                    // Index should have been validated
                    let func_ty = self.func_ty(self.type_index(idx).unwrap()).unwrap();
                    ExternTy::Func(func_ty.clone())
                }
                ExportDesc::Table(idx) => {
                    // Index should have been validated
                    ExternTy::Table(self.table_ty(idx).unwrap())
                }
                ExportDesc::Mem(idx) => {
                    // Index should have been validated
                    ExternTy::Mem(self.mem_ty(idx).unwrap())
                }
                ExportDesc::Global(idx) => {
                    // Index should have been validated
                    ExternTy::Global(self.global_ty(idx).unwrap())
                }
            };

            (ex.name.as_str(), ty)
        })
    }
}
