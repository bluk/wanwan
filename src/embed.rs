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
        FuncAddr, GlobalAddr, HostcodeError, MemAddr, Store, TableAddr,
    },
    fmt::binary::{self},
    module::{
        ty::{ExternTy, FuncTy, GlobalTy, MemTy, TableTy},
        ImportDesc, Module,
    },
};

#[cfg(feature = "std")]
use crate::exec::ModuleInst;

#[derive(Debug)]
pub(crate) enum InnerError {
    Decode(String, u64),
    Exec(exec::Error),
    InvalidArguments,
    InvalidImport,
    UnknownExport,
    Trap,
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

impl From<exec::Trap> for InnerError {
    fn from(_: exec::Trap) -> Self {
        Self::Trap
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

impl From<exec::Trap> for Error {
    fn from(_: exec::Trap) -> Self {
        Self {
            inner: InnerError::Trap,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            InnerError::Decode(error, _) => f.write_str(error),
            InnerError::Exec(error) => fmt::Display::fmt(error, f),
            InnerError::InvalidArguments => f.write_str("invalid arguments"),
            InnerError::InvalidImport => f.write_str("invalid import"),
            InnerError::Trap => f.write_str("trap"),
            InnerError::UnknownExport => f.write_str("unknown export"),
        }
    }
}

#[cfg(feature = "std")]
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.inner {
            InnerError::Decode(_, _)
            | InnerError::InvalidArguments
            | InnerError::InvalidImport
            | InnerError::Trap
            | InnerError::UnknownExport => None,
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

/// Instantites a Wasm module.
///
/// # Errors
///
/// Returns an error if instantiation fails.
///
/// # Panics
///
/// Panics if the import address values are not valid.
#[allow(clippy::too_many_lines)]
#[cfg(feature = "std")]
#[inline]
pub fn module_instantiate(
    mut s: Store,
    m: Module,
    imports: &[ExternVal],
) -> Result<(Store, ModuleInst), Error> {
    use crate::{
        exec::{val::Num, ExportInst},
        module::{DataIndex, DataMode, ElementIndex, ElementSegmentMode, ExportDesc},
    };

    let m = module_validate(m)?;

    if imports.len() != m.imports().len() {
        Err(InnerError::InvalidImport)?;
    }

    let moduleinst_init = ModuleInst::new();

    let mut imported_tables = Vec::new();
    let mut imported_mems = Vec::new();

    moduleinst_init.write(|mut moduleinst_init| {
        for (ext, im) in imports.iter().zip(m.imports().iter()) {
            match ext {
                ExternVal::Func(f) => {
                    let ext_ty = s.func_ty(*f).ok_or(InnerError::InvalidImport)?;
                    match im.desc {
                        ImportDesc::Func(idx) => {
                            let im_ty = m.func_ty(idx).ok_or(InnerError::InvalidImport)?;
                            if ext_ty != im_ty {
                                Err(InnerError::InvalidImport)?;
                            }

                            moduleinst_init.push_func(*f);
                        }
                        ImportDesc::Table(_) | ImportDesc::Mem(_) | ImportDesc::Global(_) => {
                            Err(InnerError::InvalidImport)?;
                        }
                    }
                }
                ExternVal::Table(t) => {
                    let ext_ty = s.table_ty(*t).ok_or(InnerError::InvalidImport)?;
                    match im.desc {
                        ImportDesc::Table(im_ty) => {
                            if ext_ty != im_ty {
                                Err(InnerError::InvalidImport)?;
                            }
                            imported_tables.push(*t);
                        }
                        ImportDesc::Func(_) | ImportDesc::Mem(_) | ImportDesc::Global(_) => {
                            Err(InnerError::InvalidImport)?;
                        }
                    }
                }
                ExternVal::Mem(m) => {
                    let ext_ty = s.mem_ty(*m).ok_or(InnerError::InvalidImport)?;
                    match im.desc {
                        ImportDesc::Mem(im_ty) => {
                            if ext_ty != im_ty {
                                Err(InnerError::InvalidImport)?;
                            }
                            imported_mems.push(*m);
                        }
                        ImportDesc::Func(_) | ImportDesc::Table(_) | ImportDesc::Global(_) => {
                            Err(InnerError::InvalidImport)?;
                        }
                    }
                }
                ExternVal::Global(g) => {
                    let ext_ty = s.global_ty(*g).ok_or(InnerError::InvalidImport)?;
                    match im.desc {
                        ImportDesc::Global(im_ty) => {
                            if ext_ty != im_ty {
                                Err(InnerError::InvalidImport)?;
                            }
                            moduleinst_init.push_global(*g);
                        }
                        ImportDesc::Func(_) | ImportDesc::Table(_) | ImportDesc::Mem(_) => {
                            Err(InnerError::InvalidImport)?;
                        }
                    }
                }
            }
        }

        Ok::<_, Error>(())
    })?;

    let module_inst = moduleinst_init;
    let m2 = module_inst.clone();
    module_inst.write(|mut module_inst| {
        for f in m.funcs().iter().cloned() {
            let ty = m.func_ty(f.ty).unwrap();
            let addr = s.func_alloc(ty.clone(), m2.clone(), f.clone());
            module_inst.push_func(addr);
        }

        for t in m.tables() {
            let addr = s.table_alloc(t.ty, Ref::Null(t.ty.elem_ty));
            module_inst.push_table(addr);
        }

        for m in m.mems() {
            let addr = s.mem_alloc(m.ty);
            module_inst.push_mem(addr);
        }

        for g in m.globals() {
            let val = g.init.eval(&s, &module_inst);
            let addr = s.global_alloc(g.ty, val);
            module_inst.push_global(addr);
        }
    });

    for e in m.elems() {
        let elem = module_inst.read(|module_inst| {
            e.init
                .iter()
                .map(|expr| expr.eval(&s, &module_inst))
                .map(|val| match val {
                    Val::Ref(r) => r,
                    Val::Num(_) | Val::Vec(_) => unreachable!(),
                })
                .collect::<Vec<_>>()
        });
        let addr = s.elem_alloc(e.ty, elem);
        module_inst.write(|mut module_inst| {
            module_inst.push_elem(addr);
        });
    }

    module_inst.write(|mut module_inst| {
        for d in m.datas() {
            let addr = s.data_alloc(d.init.clone());
            module_inst.push_data(addr);
        }

        for e in m.exports() {
            let value = match &e.desc {
                ExportDesc::Func(f) => ExternVal::Func(module_inst.func_addr(*f).unwrap()),
                ExportDesc::Table(t) => ExternVal::Table(module_inst.table_addr(*t).unwrap()),
                ExportDesc::Mem(m) => ExternVal::Mem(module_inst.mem_addr(*m).unwrap()),
                ExportDesc::Global(g) => ExternVal::Global(module_inst.global_addr(*g).unwrap()),
            };

            module_inst.push_export(ExportInst {
                name: e.name.clone(),
                value,
            });
        }
    });

    module_inst.read::<Result<_, _>, _>(|module_inst| {
        for (i, e) in m.elems().iter().enumerate() {
            let elem_idx = ElementIndex::new(u32::try_from(i).unwrap());
            match &e.mode {
                ElementSegmentMode::Active(table_idx, offset) => {
                    let n = i32::try_from(e.init.len()).unwrap();

                    let offset = offset.eval(&s, &module_inst);
                    let d = match offset {
                        Val::Num(num) => match num {
                            Num::I32(d) => d,
                            Num::I64(_) | Num::F32(_) | Num::F64(_) => unreachable!(),
                        },
                        Val::Vec(_) | Val::Ref(_) => unreachable!(),
                    };

                    s.table_init(&module_inst, *table_idx, elem_idx, n, 0, d)?;
                    s.elem_drop(&module_inst, elem_idx);
                }
                ElementSegmentMode::Declarative => {
                    s.elem_drop(&module_inst, elem_idx);
                }
                ElementSegmentMode::Passive => {}
            }
        }

        for (i, d) in m.datas().iter().enumerate() {
            let data_idx = DataIndex::new(u32::try_from(i).unwrap());
            match &d.mode {
                DataMode::Passive => {}
                DataMode::Active(mem_idx, offset) => {
                    assert_eq!(u32::from(*mem_idx), 0);
                    let n = i32::try_from(d.init.len()).unwrap();
                    let d = match offset.eval(&s, &module_inst) {
                        Val::Num(num) => match num {
                            Num::I32(d) => d,
                            Num::I64(_) | Num::F32(_) | Num::F64(_) => unreachable!(),
                        },
                        Val::Vec(_) | Val::Ref(_) => unreachable!(),
                    };
                    s.mem_init(&module_inst, data_idx, n, 0, d)?;
                    s.data_drop(&module_inst, data_idx);
                }
            }
        }

        Ok::<_, Error>(())
    })?;

    let start_func = m.start();

    module_inst.write(|mut module_inst| {
        module_inst.set_types(m.types.into_inner());
    });

    if let Some(start_func) = start_func {
        let start_func_addr = module_inst
            .read(|module_inst| module_inst.func_addr(start_func))
            .unwrap();
        s.eval(start_func_addr, &[])?;
    }

    Ok((s, module_inst))
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
#[cfg(feature = "std")]
pub fn instance_export(m: &ModuleInst, name: &str) -> Result<ExternVal, Error> {
    m.read(|m| {
        m.exports()
            .iter()
            .find_map(|e| (e.name == name).then_some(e.value))
            .ok_or(Error::from(InnerError::UnknownExport))
    })
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

/// Invokes a function.
///
/// # Errors
///
/// If the function invocation traps.
///
/// # Panics
///
/// If the function address is invalid.
#[cfg(feature = "std")]
pub fn func_invoke(mut s: Store, a: FuncAddr, values: &[Val]) -> Result<(Store, Vec<Val>), Error> {
    let result = s.eval(a, values)?;
    Ok((s, result))
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
