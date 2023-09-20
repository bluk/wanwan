//! Execution related code.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{boxed::Box, fmt, string::String, vec, vec::Vec};
use core::ops::Deref;
#[cfg(feature = "std")]
use std::{
    boxed::Box,
    error, fmt,
    string::String,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
    vec,
    vec::Vec,
};

use crate::module::{
    instr::{Const, ConstExpr, ConstInstr},
    ty::{FuncTy, GlobalTy, MemTy, Mut, RefTy, TableTy},
    DataIndex, ElementIndex, Func, FuncIndex, GlobalIndex, ImportGlobalIndex, MemIndex, TableIndex,
};

use self::val::{ExternVal, Ref, Val};

pub mod val;

const PAGE_SIZE: usize = 2usize.pow(16);

#[derive(Debug)]
enum InnerError {
    InvalidIndex,
    CouldNotGrowTable,
    CouldNotGrowMemory,
    NotMutable,
}

/// Generic error
#[derive(Debug)]
pub(crate) struct Error {
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
            InnerError::InvalidIndex => f.write_str("invalid index"),
            InnerError::CouldNotGrowTable => f.write_str("could not grow table"),
            InnerError::CouldNotGrowMemory => f.write_str("could not grow memory"),
            InnerError::NotMutable => f.write_str("not mutable value"),
        }
    }
}

#[cfg(feature = "std")]
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

#[derive(Debug)]
pub(crate) struct Trap;

/// Stores state for an instantiated Wasm program.
///
/// The store is somewhat analogous to a process's memory. For instance, a
/// decoded function is read from a validated module and placed somewhere in
/// the store and assigned an address.
///
/// The consumer of the API creates a new store. If the Wasm module declares any
/// imports, then the API consumer must allocate function(s), table(s),
/// memories, and/or globals in the store before instantiating the module. As
/// each instance is allocated, the API consumer should keep the returned
/// addresses. Finally, when instantiating the module, the addresses for the
/// instances matching the import's types are passed along with the validated module.
#[derive(Debug, Default)]
pub struct Store {
    funcs: Vec<FuncInst>,
    tables: Vec<TableInst>,
    mems: Vec<MemInst>,
    globals: Vec<GlobalInst>,
    elems: Vec<ElemInst>,
    datas: Vec<DataInst>,
}

impl Store {
    #[inline]
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            funcs: Vec::new(),
            tables: Vec::new(),
            mems: Vec::new(),
            globals: Vec::new(),
            elems: Vec::new(),
            datas: Vec::new(),
        }
    }

    pub(crate) fn host_func_alloc<F>(&mut self, ty: FuncTy, code: F) -> FuncAddr
    where
        F: FnMut(&[Val]) -> Result<Vec<Val>, HostcodeError> + 'static,
    {
        let a = FuncAddr(self.funcs.len());
        self.funcs.push(FuncInst::Host {
            ty,
            hostcode: Box::new(code),
        });
        a
    }

    #[cfg(feature = "std")]
    #[must_use]
    pub(crate) fn func_alloc(&mut self, ty: FuncTy, module: ModuleInst, code: Func) -> FuncAddr {
        let a = FuncAddr(self.funcs.len());
        self.funcs.push(FuncInst::Module { ty, module, code });
        a
    }

    #[must_use]
    pub(crate) fn func_ty(&self, addr: FuncAddr) -> Option<&FuncTy> {
        #[cfg(feature = "std")]
        {
            self.funcs.get(addr.0).map(|f| match f {
                FuncInst::Host { ty, hostcode: _ }
                | FuncInst::Module {
                    ty,
                    module: _,
                    code: _,
                } => ty,
            })
        }
        #[cfg(all(feature = "alloc", not(feature = "std")))]
        {
            self.funcs.get(addr.0).map(|f| match f {
                FuncInst::Host { ty, hostcode: _ } => ty,
            })
        }
    }

    pub(crate) fn table_alloc(&mut self, ty: TableTy, r: Ref) -> TableAddr {
        let a = TableAddr(self.tables.len());
        let min = usize::try_from(ty.lim.min).unwrap();
        let mut elem = Vec::with_capacity(min);
        elem.resize(min, r);
        self.tables.push(TableInst { ty, elem });
        a
    }

    #[must_use]
    pub(crate) fn table_ty(&self, addr: TableAddr) -> Option<TableTy> {
        self.tables.get(addr.0).map(|t| t.ty)
    }

    pub(crate) fn table_read(&self, addr: TableAddr, i: u32) -> Result<Ref, Error> {
        Ok(self
            .tables
            .get(addr.0)
            .and_then(|t| t.elem.get(usize::try_from(i).unwrap()).copied())
            .ok_or(InnerError::InvalidIndex)?)
    }

    pub(crate) fn table_write(&mut self, addr: TableAddr, i: u32, r: Ref) -> Result<(), Error> {
        let table = self
            .tables
            .get_mut(addr.0)
            .ok_or(InnerError::InvalidIndex)?;
        let val = table
            .elem
            .get_mut(usize::try_from(i).unwrap())
            .ok_or(InnerError::InvalidIndex)?;
        *val = r;
        Ok(())
    }

    #[must_use]
    pub(crate) fn table_size(&self, addr: TableAddr) -> u32 {
        u32::try_from(self.tables.get(addr.0).map(|t| t.elem.len()).unwrap()).unwrap()
    }

    pub(crate) fn table_grow(&mut self, addr: TableAddr, n: u32, r: Ref) -> Result<(), Error> {
        let table = self
            .tables
            .get_mut(addr.0)
            .ok_or(InnerError::CouldNotGrowTable)?;
        if let Some(max) = table.ty.lim.max {
            if n > max {
                return Err(InnerError::CouldNotGrowTable)?;
            }
        }
        let n = usize::try_from(n).unwrap();
        if n < table.elem.len() {
            return Err(InnerError::CouldNotGrowTable)?;
        }
        table.elem.resize(n, r);
        Ok(())
    }

    pub(crate) fn table_init<M>(
        &mut self,
        module_inst: &M,
        table_idx: TableIndex,
        elem_idx: ElementIndex,
        n: i32,
        s: i32,
        d: i32,
    ) -> Result<(), Trap>
    where
        M: Deref<Target = ModuleInstInternal>,
    {
        let mut n = usize::try_from(u32::from_ne_bytes(n.to_ne_bytes())).unwrap();
        let mut s = usize::try_from(u32::from_ne_bytes(s.to_ne_bytes())).unwrap();
        let mut d = usize::try_from(u32::from_ne_bytes(d.to_ne_bytes())).unwrap();

        let table_addr = module_inst.table_addr(table_idx).unwrap();
        let table = &mut self.tables[usize::from(table_addr)];

        let elem_addr = module_inst.elem_addr(elem_idx).unwrap();
        let elem = &mut self.elems[usize::from(elem_addr)];

        if s + n > elem.elem.len() || d + n > table.elem.len() {
            return Err(Trap);
        }

        while n > 0 {
            table.elem[d] = elem.elem[s];

            d += 1;
            s += 1;
            n -= 1;
        }

        Ok(())
    }

    #[must_use]
    pub(crate) fn mem_alloc(&mut self, ty: MemTy) -> MemAddr {
        let a = MemAddr(self.mems.len());
        let min = usize::try_from(ty.lim.min).unwrap();
        let elem = vec![0; min * PAGE_SIZE];
        self.mems.push(MemInst { ty, elem });
        a
    }

    #[must_use]
    pub(crate) fn mem_ty(&self, addr: MemAddr) -> Option<MemTy> {
        self.mems.get(addr.0).map(|t| t.ty)
    }

    pub(crate) fn mem_read(&self, addr: MemAddr, i: u32) -> Result<u8, Error> {
        Ok(self
            .mems
            .get(addr.0)
            .and_then(|m| m.elem.get(usize::try_from(i).unwrap()).copied())
            .ok_or(InnerError::InvalidIndex)?)
    }

    pub(crate) fn mem_write(&mut self, addr: MemAddr, i: u32, b: u8) -> Result<(), Error> {
        let mem = self.mems.get_mut(addr.0).ok_or(InnerError::InvalidIndex)?;
        let val = mem
            .elem
            .get_mut(usize::try_from(i).unwrap())
            .ok_or(InnerError::InvalidIndex)?;
        *val = b;
        Ok(())
    }

    #[must_use]
    pub(crate) fn mem_size(&self, addr: MemAddr) -> u32 {
        u32::try_from(self.mems.get(addr.0).map(|m| m.elem.len()).unwrap()).unwrap()
    }

    pub(crate) fn mem_grow(&mut self, addr: MemAddr, n: u32) -> Result<(), Error> {
        let mem = self
            .mems
            .get_mut(addr.0)
            .ok_or(InnerError::CouldNotGrowMemory)?;
        if let Some(max) = mem.ty.lim.max {
            if n > max {
                return Err(InnerError::CouldNotGrowMemory)?;
            }
        }
        let n = usize::try_from(n).unwrap();
        if n * PAGE_SIZE < mem.elem.len() {
            return Err(InnerError::CouldNotGrowMemory)?;
        }
        mem.elem.resize(n * PAGE_SIZE, 0);
        Ok(())
    }

    pub(crate) fn mem_init<M>(
        &mut self,
        module_inst: &M,
        data_idx: DataIndex,
        n: i32,
        s: i32,
        d: i32,
    ) -> Result<(), Trap>
    where
        M: Deref<Target = ModuleInstInternal>,
    {
        let mut n = usize::try_from(u32::from_ne_bytes(n.to_ne_bytes())).unwrap();
        let mut s = usize::try_from(u32::from_ne_bytes(s.to_ne_bytes())).unwrap();
        let mut d = usize::try_from(u32::from_ne_bytes(d.to_ne_bytes())).unwrap();

        let mem_addr = module_inst.mem_addr(MemIndex::new(0)).unwrap();
        let mem = &mut self.mems[usize::from(mem_addr)];

        let data_addr = module_inst.data_addr(data_idx).unwrap();
        let data = &mut self.datas[usize::from(data_addr)];

        if s + n > data.data.len() || d + n > mem.elem.len() {
            return Err(Trap);
        }

        while n > 0 {
            mem.elem[d] = data.data[s];

            d += 1;
            s += 1;
            n -= 1;
        }

        Ok(())
    }

    #[must_use]
    pub(crate) fn global_alloc(&mut self, ty: GlobalTy, value: Val) -> GlobalAddr {
        let a = GlobalAddr(self.globals.len());
        self.globals.push(GlobalInst { ty, value });
        a
    }

    #[must_use]
    pub(crate) fn global_ty(&self, addr: GlobalAddr) -> Option<GlobalTy> {
        self.globals.get(addr.0).map(|t| t.ty)
    }

    #[must_use]
    pub(crate) fn global_read(&self, addr: GlobalAddr) -> Option<Val> {
        self.globals.get(addr.0).map(|g| g.value)
    }

    pub(crate) fn global_write(&mut self, addr: GlobalAddr, val: Val) -> Result<(), Error> {
        let g = self
            .globals
            .get_mut(addr.0)
            .ok_or(InnerError::InvalidIndex)?;
        match g.ty.m {
            Mut::Const => {
                return Err(InnerError::NotMutable)?;
            }
            Mut::Var => {}
        }
        g.value = val;
        Ok(())
    }

    #[must_use]
    pub(crate) fn elem_alloc(&mut self, ty: RefTy, elem: Vec<Ref>) -> ElemAddr {
        let a = ElemAddr(self.elems.len());
        self.elems.push(ElemInst { ty, elem });
        a
    }

    pub(crate) fn elem_drop<M>(&mut self, module_inst: &M, idx: ElementIndex)
    where
        M: Deref<Target = ModuleInstInternal>,
    {
        let addr = module_inst.elem_addr(idx).unwrap();
        // TODO: "Drop" the element segment from the store
        self.elems[usize::from(addr)].elem = Vec::new();
    }

    #[must_use]
    pub(crate) fn data_alloc(&mut self, data: Vec<u8>) -> DataAddr {
        let a = DataAddr(self.datas.len());
        self.datas.push(DataInst { data });
        a
    }

    pub(crate) fn data_drop<M>(&mut self, module_inst: &M, idx: DataIndex)
    where
        M: Deref<Target = ModuleInstInternal>,
    {
        let addr = module_inst.data_addr(idx).unwrap();
        // TODO: "Drop" the element segment from the store
        self.datas[usize::from(addr)].data = Vec::new();
    }
}

macro_rules! impl_addr {
    ($ty:ty) => {
        impl From<$ty> for usize {
            fn from(value: $ty) -> Self {
                value.0
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FuncAddr(usize);

impl_addr!(FuncAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableAddr(usize);

impl_addr!(TableAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemAddr(usize);

impl_addr!(MemAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalAddr(usize);

impl_addr!(GlobalAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ElemAddr(usize);

impl_addr!(ElemAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DataAddr(usize);

impl_addr!(DataAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternAddr(usize);

impl_addr!(ExternAddr);

#[derive(Debug)]
pub(crate) struct ModuleInstInternal {
    func_addrs: Vec<FuncAddr>,
    table_addrs: Vec<TableAddr>,
    mem_addrs: Vec<MemAddr>,
    global_addrs: Vec<GlobalAddr>,
    elem_addrs: Vec<ElemAddr>,
    data_addrs: Vec<DataAddr>,
    exports: Vec<ExportInst>,
}

impl ModuleInstInternal {
    #[inline]
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            func_addrs: Vec::new(),
            table_addrs: Vec::new(),
            mem_addrs: Vec::new(),
            global_addrs: Vec::new(),
            elem_addrs: Vec::new(),
            data_addrs: Vec::new(),
            exports: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn push_func(&mut self, addr: FuncAddr) {
        self.func_addrs.push(addr);
    }

    #[inline]
    pub(crate) fn push_table(&mut self, addr: TableAddr) {
        self.table_addrs.push(addr);
    }

    #[inline]
    pub(crate) fn push_mem(&mut self, addr: MemAddr) {
        self.mem_addrs.push(addr);
    }

    #[inline]
    pub(crate) fn push_global(&mut self, addr: GlobalAddr) {
        self.global_addrs.push(addr);
    }

    #[inline]
    pub(crate) fn push_elem(&mut self, addr: ElemAddr) {
        self.elem_addrs.push(addr);
    }

    #[inline]
    pub(crate) fn push_data(&mut self, addr: DataAddr) {
        self.data_addrs.push(addr);
    }

    #[inline]
    pub(crate) fn push_export(&mut self, export: ExportInst) {
        self.exports.push(export);
    }

    #[inline]
    #[must_use]
    pub(crate) fn func_addr(&self, idx: FuncIndex) -> Option<FuncAddr> {
        self.func_addrs.get(usize::try_from(idx).unwrap()).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn import_global_addr(&self, idx: ImportGlobalIndex) -> Option<GlobalAddr> {
        self.global_addrs
            .get(usize::try_from(idx).unwrap())
            .copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn table_addr(&self, idx: TableIndex) -> Option<TableAddr> {
        self.table_addrs.get(usize::try_from(idx).unwrap()).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn global_addr(&self, idx: GlobalIndex) -> Option<GlobalAddr> {
        self.global_addrs
            .get(usize::try_from(idx).unwrap())
            .copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn elem_addr(&self, idx: ElementIndex) -> Option<ElemAddr> {
        self.elem_addrs.get(usize::try_from(idx).unwrap()).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn mem_addr(&self, idx: MemIndex) -> Option<MemAddr> {
        self.mem_addrs.get(usize::try_from(idx).unwrap()).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn data_addr(&self, idx: DataIndex) -> Option<DataAddr> {
        self.data_addrs.get(usize::try_from(idx).unwrap()).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn exports(&self) -> &[ExportInst] {
        self.exports.as_slice()
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct ModuleInst {
    internal: Arc<RwLock<ModuleInstInternal>>,
}

#[cfg(feature = "std")]
impl ModuleInst {
    #[inline]
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            internal: Arc::new(RwLock::new(ModuleInstInternal::new())),
        }
    }

    pub(crate) fn read<R, F>(&self, f: F) -> R
    where
        F: FnOnce(RwLockReadGuard<'_, ModuleInstInternal>) -> R,
    {
        let inst = self.internal.read().unwrap();
        f(inst)
    }

    pub(crate) fn write<R, F>(&self, f: F) -> R
    where
        F: FnOnce(RwLockWriteGuard<'_, ModuleInstInternal>) -> R,
    {
        let inst = self.internal.write().unwrap();
        f(inst)
    }
}

#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy)]
pub struct HostcodeError;

impl HostcodeError {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl fmt::Display for HostcodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("hostcode error")
    }
}

#[cfg(feature = "std")]
impl error::Error for HostcodeError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

pub type Hostcode = Box<dyn FnMut(&[Val]) -> Result<Vec<Val>, HostcodeError>>;

pub(crate) enum FuncInst {
    #[cfg(feature = "std")]
    Module {
        ty: FuncTy,
        module: ModuleInst,
        code: Func,
    },
    Host {
        ty: FuncTy,
        hostcode: Hostcode,
    },
}

impl fmt::Debug for FuncInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Module { ty, module, code } => f
                .debug_struct("Module")
                .field("ty", ty)
                .field("module", module)
                .field("code", code)
                .finish(),
            Self::Host { ty, hostcode: _ } => f
                .debug_struct("Host")
                .field("ty", ty)
                .finish_non_exhaustive(),
        }
    }
}

impl FuncInst {
    fn ty(&self) -> &FuncTy {
        #[cfg(feature = "std")]
        {
            match self {
                FuncInst::Module {
                    ty,
                    module: _,
                    code: _,
                }
                | FuncInst::Host { ty, hostcode: _ } => ty,
            }
        }
        #[cfg(all(feature = "alloc", not(feature = "std")))]
        {
            match self {
                FuncInst::Host { ty, hostcode: _ } => ty,
            }
        }
    }
}

#[derive(Debug)]

pub(crate) struct TableInst {
    pub(crate) ty: TableTy,
    pub(crate) elem: Vec<Ref>,
}

#[derive(Debug)]

pub(crate) struct MemInst {
    pub(crate) ty: MemTy,
    pub(crate) elem: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GlobalInst {
    pub(crate) ty: GlobalTy,
    pub(crate) value: Val,
}

#[derive(Debug)]
pub(crate) struct ElemInst {
    pub(crate) ty: RefTy,
    pub(crate) elem: Vec<Ref>,
}

#[derive(Debug)]
pub(crate) struct DataInst {
    pub(crate) data: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct ExportInst {
    pub(crate) name: String,
    pub(crate) value: ExternVal,
}

impl ConstExpr {
    #[cfg(feature = "std")]
    pub(crate) fn eval<M>(&self, store: &Store, module_inst: &M) -> Val
    where
        M: Deref<Target = ModuleInstInternal>,
    {
        debug_assert_eq!(self.instrs.len(), 1);
        match &self.instrs[0] {
            ConstInstr::Constant(v) => match v {
                Const::I32(v) => Val::from(*v),
                Const::I64(v) => Val::from(*v),
                Const::F32(v) => Val::from(*v),
                Const::F64(v) => Val::from(*v),
            },
            ConstInstr::RefNull(v) => Val::Ref(Ref::Null(*v)),
            ConstInstr::RefFunc(idx) => {
                let addr = module_inst.func_addr(*idx).unwrap();
                Val::from(addr)
            }
            ConstInstr::GlobalGet(idx) => {
                let addr = module_inst.import_global_addr(*idx).unwrap();
                store.global_read(addr).unwrap()
            }
        }
    }
}
