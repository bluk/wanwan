//! Execution related code.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{boxed::Box, fmt, string::String, vec, vec::Vec};
#[cfg(feature = "std")]
use std::{boxed::Box, error, fmt, string::String, vec, vec::Vec};

use crate::module::ty::{FuncTy, GlobalTy, MemTy, Mut, RefTy, TableTy};

use self::val::{ExternVal, Ref, Val};

pub(crate) mod val;

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
}

impl Store {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            funcs: Vec::new(),
            tables: Vec::new(),
            mems: Vec::new(),
            globals: Vec::new(),
        }
    }

    pub fn host_func_alloc<F>(&mut self, ft: FuncTy, code: F) -> FuncAddr
    where
        F: FnMut(&[Val]) -> Result<Vec<Val>, HostcodeError> + 'static,
    {
        let a = FuncAddr::from(self.funcs.len());
        self.funcs.push(FuncInst::Host {
            ty: ft,
            hostcode: Box::new(code),
        });
        a
    }

    pub fn func_ty(&self, addr: FuncAddr) -> Option<&FuncTy> {
        self.funcs.get(addr.0).map(|f| match f {
            FuncInst::Host { ty, hostcode: _ } => ty,
        })
    }

    pub fn table_alloc(&mut self, ty: TableTy, r: Ref) -> TableAddr {
        let a = TableAddr::from(self.tables.len());
        let min = usize::try_from(ty.lim.min).unwrap();
        let mut elem = Vec::with_capacity(min);
        elem.resize(min, r);
        self.tables.push(TableInst { ty, elem });
        a
    }

    pub fn table_ty(&self, addr: TableAddr) -> Option<TableTy> {
        self.tables.get(addr.0).map(|t| t.ty)
    }

    pub fn table_read(&self, addr: TableAddr, i: u32) -> Result<Ref, Error> {
        Ok(self
            .tables
            .get(addr.0)
            .and_then(|t| t.elem.get(usize::try_from(i).unwrap()).copied())
            .ok_or(InnerError::InvalidIndex)?)
    }

    pub fn table_write(&mut self, addr: TableAddr, i: u32, r: Ref) -> Result<(), Error> {
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
    pub fn table_size(&self, addr: TableAddr) -> u32 {
        u32::try_from(self.tables.get(addr.0).map(|t| t.elem.len()).unwrap()).unwrap()
    }

    pub fn table_grow(&mut self, addr: TableAddr, n: u32, r: Ref) -> Result<(), Error> {
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

    pub fn mem_alloc(&mut self, ty: MemTy) -> MemAddr {
        let a = MemAddr::from(self.mems.len());
        let min = usize::try_from(ty.lim.min).unwrap();
        let elem = vec![0; min * PAGE_SIZE];
        self.mems.push(MemInst { ty, elem });
        a
    }

    pub fn mem_ty(&self, addr: MemAddr) -> Option<MemTy> {
        self.mems.get(addr.0).map(|t| t.ty)
    }

    pub fn mem_read(&self, addr: MemAddr, i: u32) -> Result<u8, Error> {
        Ok(self
            .mems
            .get(addr.0)
            .and_then(|m| m.elem.get(usize::try_from(i).unwrap()).copied())
            .ok_or(InnerError::InvalidIndex)?)
    }

    pub fn mem_write(&mut self, addr: MemAddr, i: u32, b: u8) -> Result<(), Error> {
        let mem = self.mems.get_mut(addr.0).ok_or(InnerError::InvalidIndex)?;
        let val = mem
            .elem
            .get_mut(usize::try_from(i).unwrap())
            .ok_or(InnerError::InvalidIndex)?;
        *val = b;
        Ok(())
    }

    #[must_use]
    pub fn mem_size(&self, addr: MemAddr) -> u32 {
        u32::try_from(self.mems.get(addr.0).map(|m| m.elem.len()).unwrap()).unwrap()
    }

    pub fn mem_grow(&mut self, addr: MemAddr, n: u32) -> Result<(), Error> {
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

    pub fn global_alloc(&mut self, ty: GlobalTy, value: Val) -> GlobalAddr {
        let a = GlobalAddr::from(self.globals.len());
        self.globals.push(GlobalInst { ty, value });
        a
    }

    pub fn global_ty(&self, addr: GlobalAddr) -> Option<GlobalTy> {
        self.globals.get(addr.0).map(|t| t.ty)
    }

    pub fn global_read(&self, addr: GlobalAddr) -> Option<Val> {
        self.globals.get(addr.0).map(|g| g.value)
    }

    pub fn global_write(&mut self, addr: GlobalAddr, val: Val) -> Result<(), Error> {
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
}

macro_rules! impl_addr {
    ($ty:ty) => {
        impl $ty {
            #[inline]
            #[must_use]
            pub const fn new(addr: usize) -> Self {
                Self(addr)
            }
        }

        impl From<usize> for $ty {
            fn from(value: usize) -> Self {
                Self(value)
            }
        }

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
pub struct ElemAddr(usize);

impl_addr!(ElemAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataAddr(usize);

impl_addr!(DataAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternAddr(usize);

impl_addr!(ExternAddr);

#[derive(Debug)]
pub struct ModuleInst {
    exports: Vec<ExportInst>,
}

impl ModuleInst {
    pub fn exports(&self) -> &[ExportInst] {
        self.exports.as_slice()
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

pub enum FuncInst {
    Host { ty: FuncTy, hostcode: Hostcode },
}

impl fmt::Debug for FuncInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host { ty, hostcode: _ } => f
                .debug_struct("Host")
                .field("ty", ty)
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Debug)]

pub struct TableInst {
    pub ty: TableTy,
    pub elem: Vec<Ref>,
}

#[derive(Debug)]

pub struct MemInst {
    pub ty: MemTy,
    pub elem: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalInst {
    pub ty: GlobalTy,
    pub value: Val,
}

#[derive(Debug)]
pub struct ElemInst {
    pub ty: RefTy,
    pub elem: Vec<Ref>,
}

#[derive(Debug)]
pub struct DataInst {
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct ExportInst {
    pub name: String,
    pub value: ExternVal,
}
