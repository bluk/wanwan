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

use crate::{
    embed,
    module::{
        instr::{self, Const, ConstExpr, ConstInstr, Instr},
        ty::{self, FuncTy, GlobalTy, MemTy, Mut, RefTy, TableTy, ValTy},
        DataIndex, ElementIndex, Func, FuncIndex, GlobalIndex, ImportGlobalIndex, LabelIndex,
        MemIndex, TableIndex, TypeIndex,
    },
};

use self::val::{ExternVal, Num, Ref, Val};

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

    #[must_use]
    pub(crate) fn func(&self, addr: FuncAddr) -> Option<&FuncInst> {
        self.funcs.get(addr.0)
    }

    #[must_use]
    pub(crate) fn func_mut(&mut self, addr: FuncAddr) -> Option<&mut FuncInst> {
        self.funcs.get_mut(addr.0)
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

    #[allow(clippy::too_many_lines)]
    #[cfg(feature = "std")]
    pub(crate) fn eval(
        &mut self,
        addr: FuncAddr,
        params: &[Val],
    ) -> Result<Vec<Val>, embed::Error> {
        let mut frames: Vec<Activation> = Vec::new();
        let mut values: ValStack = ValStack::new();
        let mut labels: Vec<Label> = Vec::new();

        values.push_params(params);
        func_call(self, addr, &mut frames, &mut labels, &mut values)?;

        'call: loop {
            let frame = frames.last().unwrap();

            let instrs = match &self.funcs[usize::from(frame.func_addr)] {
                FuncInst::Module {
                    ty: _,
                    module: _,
                    code,
                } => &code.body.instrs,
                FuncInst::Host { ty: _, hostcode: _ } => unreachable!(),
            };
            let mut instr_idx = frame.instr_idx;

            loop {
                let Some(instr) = instrs.get(instr_idx) else {
                    break;
                };

                match instr {
                    Instr::Control(instr) => match instr {
                        instr::Control::Unreachable => Err(Trap)?,
                        instr::Control::Nop => {
                            // no-op
                        }
                        instr::Control::Block { bt, end_idx } => match bt {
                            instr::BlockTy::Val(ty) => {
                                labels.push(Label {
                                    values_height: values.len(),
                                    arity: ty.is_some().then_some(1).unwrap_or_default(),
                                    end_instr_idx: *end_idx,
                                    next_instr_idx: None,
                                    br_instr_idx: *end_idx,
                                });
                            }
                            instr::BlockTy::Index(ty_idx) => {
                                let frame = frames.last().unwrap();
                                frame.module.read(|module_inst| {
                                    let func_ty = module_inst.func_ty(*ty_idx).unwrap();

                                    let params = values.pop_params(func_ty.params().len());

                                    if !ty::is_compatible(func_ty.params(), &params) {
                                        return Err(Trap);
                                    }

                                    labels.push(Label {
                                        values_height: values.len(),
                                        arity: func_ty.ret().len(),
                                        end_instr_idx: *end_idx,
                                        next_instr_idx: None,
                                        br_instr_idx: *end_idx,
                                    });

                                    values.push_params(&params);

                                    Ok(())
                                })?;
                            }
                        },
                        instr::Control::Loop {
                            bt,
                            start_idx,
                            end_idx,
                        } => match bt {
                            instr::BlockTy::Val(_) => {
                                labels.push(Label {
                                    values_height: values.len(),
                                    arity: 0,
                                    end_instr_idx: *end_idx,
                                    next_instr_idx: None,
                                    br_instr_idx: *start_idx,
                                });
                            }
                            instr::BlockTy::Index(ty_idx) => {
                                let frame = frames.last().unwrap();
                                frame.module.read(|module_inst| {
                                    let func_ty = module_inst.func_ty(*ty_idx).unwrap();

                                    let params = values.pop_params(func_ty.params().len());

                                    if !ty::is_compatible(func_ty.params(), &params) {
                                        return Err(Trap);
                                    }

                                    labels.push(Label {
                                        values_height: values.len(),
                                        arity: func_ty.ret().len(),
                                        end_instr_idx: *end_idx,
                                        next_instr_idx: None,
                                        br_instr_idx: *start_idx,
                                    });

                                    values.push_params(&params);

                                    Ok(())
                                })?;
                            }
                        },
                        instr::Control::If {
                            bt,
                            then_end_idx,
                            el_end_idx,
                        } => {
                            let c = values.pop_i32();

                            let c = c != 0;
                            if !c {
                                instr_idx = *then_end_idx;
                            }

                            match bt {
                                instr::BlockTy::Val(ty) => {
                                    labels.push(Label {
                                        values_height: values.len(),
                                        arity: ty.is_some().then_some(1).unwrap_or_default(),
                                        end_instr_idx: if c { *then_end_idx } else { *el_end_idx },
                                        next_instr_idx: Some(*el_end_idx),
                                        br_instr_idx: *el_end_idx,
                                    });
                                }
                                instr::BlockTy::Index(ty_idx) => {
                                    let frame = frames.last().unwrap();
                                    frame.module.read(|module_inst| {
                                        let func_ty = module_inst.func_ty(*ty_idx).unwrap();

                                        let params = values.pop_params(func_ty.params().len());

                                        if !ty::is_compatible(func_ty.params(), &params) {
                                            return Err(Trap);
                                        }

                                        labels.push(Label {
                                            values_height: values.len(),
                                            arity: func_ty.ret().len(),
                                            end_instr_idx: if c {
                                                *then_end_idx
                                            } else {
                                                *el_end_idx
                                            },
                                            next_instr_idx: None,
                                            br_instr_idx: *el_end_idx,
                                        });

                                        values.push_params(&params);

                                        Ok(())
                                    })?;
                                }
                            }
                        }
                        instr::Control::Br(l) => {
                            branch(*l, &mut labels, &mut values, &mut instr_idx);
                        }
                        instr::Control::BrIf(l) => {
                            let c = values.pop_i32();

                            if c != 0 {
                                branch(*l, &mut labels, &mut values, &mut instr_idx);
                            }
                        }
                        instr::Control::BrTable { table, idx } => {
                            let c = values.pop_i32();
                            let c = usize::try_from(c).unwrap();
                            if c < table.len() {
                                branch(table[c], &mut labels, &mut values, &mut instr_idx);
                            } else {
                                branch(*idx, &mut labels, &mut values, &mut instr_idx);
                            }
                        }
                        instr::Control::Return => {
                            let frame = frames.last().unwrap();
                            let params = values.pop_ret(frame.arity);

                            while values.len() > frame.values_height {
                                let _ = values.pop();
                            }
                            while labels.len() > frame.labels_height {
                                let _ = labels.pop();
                            }
                            frames.pop();
                            values.push_ret(&params);
                            continue 'call;
                        }
                        instr::Control::Call(idx) => {
                            let idx = *idx;
                            let cur_frame = frames.last_mut().unwrap();
                            cur_frame.instr_idx = instr_idx + 1;

                            let func_addr = cur_frame
                                .module
                                .read(|module_inst| module_inst.func_addr(idx))
                                .unwrap();
                            func_call(self, func_addr, &mut frames, &mut labels, &mut values)?;
                            continue 'call;
                        }
                        instr::Control::CallIndirect { y, x } => {
                            let frame = frames.last().unwrap();
                            let func_addr = frame.module.read(|module_inst| {
                                let t_a = module_inst.table_addr(*x).unwrap();
                                let tab = &self.tables[usize::from(t_a)];
                                let ft_expect = module_inst.func_ty(*y).unwrap();
                                let i = values.pop_i32();
                                let i = usize::try_from(i).unwrap();
                                let Some(r) = tab.elem.get(i) else {
                                    return Err(Trap);
                                };

                                let a = match r {
                                    Ref::Null(_) => {
                                        return Err(Trap);
                                    }
                                    Ref::Extern(_) => {
                                        unreachable!()
                                    }
                                    Ref::Func(a) => a,
                                };
                                let f = &self.funcs[usize::from(*a)];
                                let ft_actual = f.ty();
                                if ft_actual != ft_expect {
                                    return Err(Trap);
                                }

                                Ok(*a)
                            })?;

                            func_call(self, func_addr, &mut frames, &mut labels, &mut values)?;
                            continue 'call;
                        }
                    },
                    Instr::Ref(_) => todo!(),
                    Instr::Parametric(_) => todo!(),
                    Instr::Var(_) => todo!(),
                    Instr::Table(_) => todo!(),
                    Instr::Mem(_) => todo!(),
                    Instr::Num(instr) => match instr {
                        instr::Num::Constant(c) => match c {
                            Const::I32(val) => {
                                values.push(Val::from(*val));
                            }
                            Const::I64(val) => {
                                values.push(Val::from(*val));
                            }
                            Const::F32(val) => {
                                values.push(Val::from(*val));
                            }
                            Const::F64(val) => {
                                values.push(Val::from(*val));
                            }
                        },
                        instr::Num::Int(_, _) => todo!(),
                        instr::Num::Float(_, _) => todo!(),
                        instr::Num::Conversion(_) => todo!(),
                    },
                }

                instr_idx += 1;

                let label = labels.last().unwrap();
                if instr_idx == label.end_instr_idx {
                    // Pop the label
                    debug_assert_eq!(values.len(), label.values_height + label.arity);
                    let last_label = labels.pop().unwrap();
                    if let Some(idx) = last_label.next_instr_idx {
                        instr_idx = idx;
                    }
                    debug_assert!(!labels.is_empty());
                }
            }

            let label = labels.last().unwrap();
            debug_assert_eq!(instr_idx, label.end_instr_idx);
            debug_assert_eq!(values.len(), label.values_height + label.arity);
            let last_label = labels.pop().unwrap();
            debug_assert_eq!(last_label.next_instr_idx, None);

            let last_frame = frames.last().unwrap();
            debug_assert_eq!(labels.len(), last_frame.labels_height);
            debug_assert_eq!(values.len(), last_frame.values_height + last_frame.arity);

            let ret = values.pop_ret(last_frame.arity);

            let func_ty = self.func(last_frame.func_addr).unwrap().ty();
            if !ty::is_compatible(func_ty.ret(), &ret) {
                return Err(Trap)?;
            }

            frames.pop();

            values.push_ret(&ret);

            if frames.is_empty() {
                break;
            }
        }

        let mut values = values.into_inner();
        values.reverse();
        Ok(values)
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
    types: Vec<FuncTy>,
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
            types: Vec::new(),
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
    pub(crate) fn set_types(&mut self, types: Vec<FuncTy>) {
        self.types = types;
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
    pub(crate) fn func_ty(&self, idx: TypeIndex) -> Option<&FuncTy> {
        self.types.get(usize::try_from(idx).unwrap())
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

#[derive(Debug)]
struct ValStack {
    stack: Vec<Val>,
}

impl ValStack {
    #[inline]
    #[must_use]
    const fn new() -> Self {
        Self { stack: Vec::new() }
    }

    #[must_use]
    fn len(&self) -> usize {
        self.stack.len()
    }

    #[inline]
    #[must_use]
    fn pop(&mut self) -> Option<Val> {
        self.stack.pop()
    }

    #[inline]
    #[must_use]
    fn pop_i32(&mut self) -> i32 {
        let Some(Val::Num(Num::I32(c))) = self.stack.pop() else {
            unreachable!()
        };
        c
    }

    #[inline]
    #[must_use]
    fn pop_params(&mut self, n: usize) -> Vec<Val> {
        let mut params = self.stack.split_off(self.stack.len() - n);
        params.reverse();
        params
    }

    fn pop_ret(&mut self, n: usize) -> Vec<Val> {
        self.pop_params(n)
    }

    #[inline]
    fn push(&mut self, param: Val) {
        self.stack.push(param);
    }

    #[inline]
    fn push_params(&mut self, params: &[Val]) {
        self.stack.extend(params.iter().copied().rev());
    }

    #[inline]
    fn push_ret(&mut self, ret: &[Val]) {
        self.push_params(ret);
    }

    #[inline]
    #[must_use]
    fn into_inner(self) -> Vec<Val> {
        self.stack
    }
}

#[derive(Debug)]
struct Label {
    values_height: usize,
    arity: usize,
    /// If the instruction counter reaches this instruction, then pop the label.
    end_instr_idx: usize,
    /// The instruction index to jump to once end_instr_idx is reached if not the same index
    next_instr_idx: Option<usize>,
    /// If a break occurs, then use this instruction to move to.
    br_instr_idx: usize,
}

#[cfg(feature = "std")]
#[derive(Debug)]
struct Activation {
    values_height: usize,
    labels_height: usize,
    locals: Vec<Val>,
    module: ModuleInst,
    func_addr: FuncAddr,
    instr_idx: usize,
    arity: usize,
}

fn branch(
    label_idx: LabelIndex,
    labels: &mut Vec<Label>,
    values: &mut ValStack,
    instr_idx: &mut usize,
) {
    let mut l = usize::try_from(label_idx).unwrap();
    debug_assert!(l < labels.len());
    let n = labels[labels.len() - l - 1].arity;
    let params = values.pop_params(n);

    while l + 1 > 0 {
        let label = labels.pop().unwrap();
        debug_assert!(values.len() >= label.values_height);
        while values.len() > label.values_height {
            let _ = values.pop();
        }

        l -= 1;
    }

    values.push_params(&params);
    let label = labels.last().unwrap();
    *instr_idx = label.br_instr_idx;
}

#[cfg(feature = "std")]
fn func_call(
    store: &mut Store,
    addr: FuncAddr,
    frames: &mut Vec<Activation>,
    labels: &mut Vec<Label>,
    values: &mut ValStack,
) -> Result<(), embed::Error> {
    match store.func_mut(addr).unwrap() {
        FuncInst::Module { ty, module, code } => {
            let n = ty.params().len();
            let params = values.pop_params(n);

            if !ty::is_compatible(ty.params(), &params) {
                unreachable!()
            }

            let mut locals = Vec::new();
            locals.extend(params.iter().rev());
            locals.extend(code.locals.iter().copied().map(ValTy::default_value));

            frames.push(Activation {
                values_height: values.len(),
                labels_height: labels.len(),
                locals,
                module: module.clone(),
                func_addr: addr,
                instr_idx: 0,
                arity: ty.ret().len(),
            });

            labels.push(Label {
                values_height: values.len(),
                arity: ty.ret().len(),
                end_instr_idx: code.body.instrs.len(),
                next_instr_idx: None,
                br_instr_idx: code.body.instrs.len(),
            });
        }
        FuncInst::Host { ty, hostcode } => {
            let n = ty.params().len();
            let params = values.pop_params(n);

            if !ty::is_compatible(ty.params(), &params) {
                unreachable!()
            }

            match hostcode(&params) {
                Ok(res) => {
                    if !ty::is_compatible(ty.ret(), &res) {
                        Err(Trap)?;
                    }

                    values.push_ret(&res);
                }
                Err(_) => {
                    // TODO: Return the right error
                    Err(Trap)?;
                }
            }
        }
    }

    Ok(())
}
