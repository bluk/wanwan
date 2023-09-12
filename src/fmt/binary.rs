//! WebAssembly Binary format
//!
//! # See Also
//!
//! * [Binary Format spec](https://www.w3.org/TR/wasm-core-2/binary/index.html)

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{
    fmt,
    string::{FromUtf8Error, String},
    vec,
    vec::Vec,
};
#[cfg(feature = "std")]
use std::{
    error, fmt, io,
    string::{FromUtf8Error, String},
    vec,
    vec::Vec,
};

use crate::{
    module::{
        instr::{BlockTy, ConstExpr, ConstInstr, Expr, Instr, MemArg},
        ty::{
            FuncTy, GlobalTy, Limits, MemoryTy, Mut, NumTy, RefTy, ResultTy, TableTy, ValTy, VecTy,
        },
        Data, DataIndex, DataMode, Elem, ElementIndex, ElementSegmentMode, Export, ExportDesc,
        FuncIndex, Global, GlobalIndex, Import, ImportDesc, ImportGlobalIndex, LabelIndex,
        LocalIndex, Mem, MemIndex, Module, Table, TableIndex, TypeIndex,
    },
    validation::{
        CodeSectionValidator, ConstExprValidator, DataContext, DataSectionValidator,
        ElementsContext, ElementsSectionValidator, ExportsSectionValidator, ExprError,
        FunctionSectionValidator, FunctionsContext, GlobalSectionValidator, GlobalsContext,
        ImportGlobalsContext, ImportSectionValidator, MemsContext, OpdTy, StartSectionValidator,
        TablesContext, TypesContext,
    },
};

mod leb128;

/// Recommended extension for files containing Wasm modules in binary format.
pub const EXTENSION: &str = "wasm";

/// Recommended media type for Wasm modules in binary format.
pub const MEDIA_TYPE: &str = "application/wasm";

#[inline]
fn expect_bytes<R>(reader: &mut R, expected: &[u8]) -> Result<(), DecodeError<R::Error>>
where
    R: Read,
{
    for e in expected {
        if *e != reader.next()? {
            return Err(DecodeError::InvalidPreamble);
        }
    }

    Ok(())
}

fn decode_vec<R, T, F>(reader: &mut R, f: F) -> Result<Vec<T>, DecodeError<R::Error>>
where
    R: Read,
    F: Fn(&mut R) -> Result<T, DecodeError<R::Error>>,
{
    let mut n = decode_u32(reader)?;
    let mut x = Vec::with_capacity(usize::try_from(n).unwrap());
    while n > 0 {
        x.push(f(reader)?);
        n -= 1;
    }

    Ok(x)
}

#[inline]
fn decode_bytes_vec<R>(reader: &mut R) -> Result<Vec<u8>, DecodeError<R::Error>>
where
    R: Read,
{
    decode_vec(reader, |r| Ok(r.next()?))
}

use leb128::{decode_s32, decode_s33_block_ty, decode_s64, decode_u32};

#[inline]
fn decode_f32<R>(reader: &mut R) -> Result<f32, DecodeError<R::Error>>
where
    R: Read,
{
    let bytes = [
        reader.next()?,
        reader.next()?,
        reader.next()?,
        reader.next()?,
    ];

    Ok(f32::from_le_bytes(bytes))
}

#[inline]
fn decode_f64<R>(reader: &mut R) -> Result<f64, DecodeError<R::Error>>
where
    R: Read,
{
    let bytes = [
        reader.next()?,
        reader.next()?,
        reader.next()?,
        reader.next()?,
        reader.next()?,
        reader.next()?,
        reader.next()?,
        reader.next()?,
    ];

    Ok(f64::from_le_bytes(bytes))
}

#[inline]
fn decode_name<R>(reader: &mut R) -> Result<String, DecodeError<R::Error>>
where
    R: Read,
{
    decode_bytes_vec(reader).and_then(|v| Ok(String::from_utf8(v)?))
}

impl RefTy {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        Ok(match reader.next()? {
            0x70 => Self::FuncRef,
            0x6f => Self::ExternRef,
            _ => return Err(DecodeError::InvalidRefTy),
        })
    }
}

impl ValTy {
    #[inline]
    fn decode_u8<R>(ty: u8) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        Ok(match ty {
            0x7f => Self::Num(NumTy::I32),
            0x7e => Self::Num(NumTy::I64),
            0x7d => Self::Num(NumTy::F32),
            0x7c => Self::Num(NumTy::F64),
            0x7b => Self::Vec(VecTy::V128),
            0x70 => Self::Ref(RefTy::FuncRef),
            0x6f => Self::Ref(RefTy::ExternRef),
            _ => return Err(DecodeError::InvalidValueTy),
        })
    }

    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        Self::decode_u8::<R>(reader.next()?)
    }
}

impl ResultTy {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        decode_vec(reader, ValTy::decode).map(Self)
    }
}

impl FuncTy {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        if reader.next()? != 0x60 {
            return Err(DecodeError::InvalidFuncTy);
        }

        let rt1 = ResultTy::decode(reader)?;
        let rt2 = ResultTy::decode(reader)?;

        Ok(Self { rt1, rt2 })
    }
}

impl Limits {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        // XXX: Should have a maximum limit supported by the runtime and ensure
        // both the minimum and maximum are not larger.

        let max_present = match reader.next()? {
            0x00 => false,
            0x01 => true,
            _ => {
                return Err(DecodeError::InvalidLimits);
            }
        };

        let n = decode_u32(reader)?;
        let m = max_present.then(|| decode_u32(reader)).transpose()?;

        if let Some(m) = m {
            if m > n {
                return Err(DecodeError::InvalidLimits);
            }
        }

        Ok(Self { min: n, max: m })
    }
}

impl MemoryTy {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        let lim = Limits::decode(reader)?;
        if lim.min > 2u32.pow(16) {
            return Err(DecodeError::InvalidMemoryTy);
        }

        if let Some(max) = lim.max {
            if max > 2u32.pow(16) {
                return Err(DecodeError::InvalidMemoryTy);
            }
        }

        Ok(Self { lim })
    }
}

impl TableTy {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        let elem_ty = RefTy::decode(reader)?;
        let lim = Limits::decode(reader)?;

        Ok(Self { elem_ty, lim })
    }
}

impl Mut {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        match reader.next()? {
            0x00 => Ok(Self::Const),
            0x01 => Ok(Self::Var),
            _ => Err(DecodeError::InvalidMut),
        }
    }
}

impl GlobalTy {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        let t = ValTy::decode(reader)?;
        let m = Mut::decode(reader)?;

        Ok(Self { m, t })
    }
}

impl BlockTy {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext,
    {
        let peek = reader.peek()?;
        if peek == 0x40 {
            reader.next()?;
            return Ok(Self::Val(None));
        }

        if let Ok(val_ty) = ValTy::decode_u8::<R>(peek) {
            reader.next()?;
            return Ok(Self::Val(Some(val_ty)));
        }

        let ty_idx = decode_s33_block_ty(reader)
            .and_then(|ty_idx| u32::try_from(ty_idx).map_err(|_| DecodeError::InvalidNum))
            .map(TypeIndex)?;

        if !ctx.is_type_valid(ty_idx) {
            return Err(DecodeError::InvalidTypeIndex);
        }

        Ok(Self::Index(ty_idx))
    }
}

impl MemArg {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        let a = decode_u32(reader)?;
        let o = decode_u32(reader)?;
        Ok(Self {
            align: a,
            offset: o,
        })
    }
}

const OP_CODE_END: u8 = 0x0b;
const OP_CODE_ELSE: u8 = 0x05;

impl Instr {
    #[allow(clippy::too_many_lines)]
    fn decode_with_op_code<R, C>(
        op_code: u8,
        reader: &mut R,
        ctx: &C,
    ) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext
            + FunctionsContext
            + TablesContext
            + GlobalsContext
            + ElementsContext
            + DataContext,
    {
        use crate::module::instr::{
            Const, Control, CvtOp, FBinOp, FRelOp, FUnOp, FloatTy, IBinOp, IRelOp, ITestOp, IUnOp,
            IntTy, Mem, Num, NumFOp, NumIOp, Parametric, Ref, SignExtension, StorageSize, Table,
            Variable,
        };

        Ok(match op_code {
            // Control Instructions
            0x00 => Self::Control(Control::Unreachable),
            0x01 => Self::Control(Control::Nop),
            0x02 => {
                let bt = BlockTy::decode(reader, ctx)?;
                let (instrs, _) = Self::decode_until(reader, &[OP_CODE_END], ctx)?;
                Self::Control(Control::Block { bt, instrs })
            }
            0x03 => {
                let bt = BlockTy::decode(reader, ctx)?;
                let (instrs, _) = Self::decode_until(reader, &[OP_CODE_END], ctx)?;
                Self::Control(Control::Loop { bt, instrs })
            }
            0x04 => {
                let bt = BlockTy::decode(reader, ctx)?;
                let (then, end) = Self::decode_until(reader, &[OP_CODE_ELSE, OP_CODE_END], ctx)?;
                let el = if end == OP_CODE_END {
                    Vec::new()
                } else {
                    let (instrs, _) = Self::decode_until(reader, &[OP_CODE_END], ctx)?;
                    instrs
                };
                Self::Control(Control::If { bt, then, el })
            }
            0x0c => {
                let l = LabelIndex::decode(reader)?;
                Self::Control(Control::Br(l))
            }
            0x0d => {
                let l = LabelIndex::decode(reader)?;
                Self::Control(Control::BrIf(l))
            }
            0x0e => {
                let table = decode_vec(reader, LabelIndex::decode)?;
                let idx = LabelIndex::decode(reader)?;
                Self::Control(Control::BrTable { table, idx })
            }
            0x0f => Self::Control(Control::Return),
            0x10 => {
                let idx = FuncIndex::decode(reader, ctx)?;
                Self::Control(Control::Call(idx))
            }
            0x11 => {
                let y = TypeIndex::decode(reader, ctx)?;
                let x = TableIndex::decode(reader, ctx)?;
                Self::Control(Control::CallIndirect { y, x })
            }

            // Reference Instructions
            0xd0 => {
                let t = RefTy::decode(reader)?;
                Self::Ref(Ref::RefNull(t))
            }
            0xd1 => Self::Ref(Ref::RefIsNull),
            0xd2 => {
                let x = FuncIndex::decode(reader, ctx)?;
                Self::Ref(Ref::RefFunc(x))
            }

            // Parametric Instructions
            0x1a => Self::Parametric(Parametric::Drop),
            0x1b => Self::Parametric(Parametric::Select(None)),
            0x1c => {
                let t = decode_vec(reader, ValTy::decode)?;
                Self::Parametric(Parametric::Select(Some(t)))
            }

            // Variable Instructions
            0x20 => {
                let x = LocalIndex::decode(reader)?;
                Self::Var(Variable::LocalGet(x))
            }
            0x21 => {
                let x = LocalIndex::decode(reader)?;
                Self::Var(Variable::LocalSet(x))
            }
            0x22 => {
                let x = LocalIndex::decode(reader)?;
                Self::Var(Variable::LocalTee(x))
            }
            0x23 => {
                let x = GlobalIndex::decode(reader, ctx)?;
                Self::Var(Variable::GlobalGet(x))
            }
            0x24 => {
                let x = GlobalIndex::decode(reader, ctx)?;
                Self::Var(Variable::GlobalSet(x))
            }

            // Table Instructions
            0x25 => {
                let x = TableIndex::decode(reader, ctx)?;
                Self::Table(Table::TableGet(x))
            }
            0x26 => {
                let x = TableIndex::decode(reader, ctx)?;
                Self::Table(Table::TableSet(x))
            }

            // Memory Instructions
            0x28 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(NumTy::I32, m, None))
            }
            0x29 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(NumTy::I64, m, None))
            }
            0x2A => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(NumTy::F32, m, None))
            }
            0x2B => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(NumTy::F64, m, None))
            }
            0x2C => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I32,
                    m,
                    Some((StorageSize::Size8, SignExtension::Signed)),
                ))
            }
            0x2D => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I32,
                    m,
                    Some((StorageSize::Size8, SignExtension::Unsigned)),
                ))
            }
            0x2E => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I32,
                    m,
                    Some((StorageSize::Size16, SignExtension::Signed)),
                ))
            }
            0x2F => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I32,
                    m,
                    Some((StorageSize::Size16, SignExtension::Unsigned)),
                ))
            }
            0x30 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I64,
                    m,
                    Some((StorageSize::Size8, SignExtension::Signed)),
                ))
            }
            0x31 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I64,
                    m,
                    Some((StorageSize::Size8, SignExtension::Unsigned)),
                ))
            }
            0x32 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I64,
                    m,
                    Some((StorageSize::Size16, SignExtension::Signed)),
                ))
            }
            0x33 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I64,
                    m,
                    Some((StorageSize::Size16, SignExtension::Unsigned)),
                ))
            }
            0x34 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I64,
                    m,
                    Some((StorageSize::Size32, SignExtension::Signed)),
                ))
            }
            0x35 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Load(
                    NumTy::I64,
                    m,
                    Some((StorageSize::Size32, SignExtension::Unsigned)),
                ))
            }
            0x36 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::I32, m, None))
            }
            0x37 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::I64, m, None))
            }
            0x38 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::F32, m, None))
            }
            0x39 => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::F64, m, None))
            }
            0x3A => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::I32, m, Some(StorageSize::Size8)))
            }
            0x3B => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::I32, m, Some(StorageSize::Size16)))
            }
            0x3C => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::I64, m, Some(StorageSize::Size8)))
            }
            0x3D => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::I64, m, Some(StorageSize::Size16)))
            }
            0x3E => {
                let m = MemArg::decode(reader)?;
                Self::Mem(Mem::Store(NumTy::I64, m, Some(StorageSize::Size32)))
            }
            0x3F => {
                match reader.next()? {
                    0x00 => {}
                    _ => return Err(DecodeError::InvalidInstr),
                }
                Self::Mem(Mem::MemorySize)
            }
            0x40 => {
                match reader.next()? {
                    0x00 => {}
                    _ => return Err(DecodeError::InvalidInstr),
                }
                Self::Mem(Mem::MemoryGrow)
            }

            // Numeric Instructions
            0x41 => {
                let n = decode_s32(reader)?;
                Self::Num(Num::Constant(Const::I32(n)))
            }
            0x42 => {
                let n = decode_s64(reader)?;
                Self::Num(Num::Constant(Const::I64(n)))
            }
            0x43 => {
                let n = decode_f32(reader)?;
                Self::Num(Num::Constant(Const::F32(n)))
            }
            0x44 => {
                let n = decode_f64(reader)?;
                Self::Num(Num::Constant(Const::F64(n)))
            }
            // I32 Test Operations
            0x45 => Self::Num(Num::Int(IntTy::I32, NumIOp::Test(ITestOp::Eqz))),

            // I32 Comparision operations
            0x46 => Self::Num(Num::Int(IntTy::I32, NumIOp::Rel(IRelOp::Eq))),
            0x47 => Self::Num(Num::Int(IntTy::I32, NumIOp::Rel(IRelOp::Ne))),
            0x48 => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Rel(IRelOp::Lt(SignExtension::Signed)),
            )),
            0x49 => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Rel(IRelOp::Lt(SignExtension::Unsigned)),
            )),
            0x4a => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Rel(IRelOp::Gt(SignExtension::Signed)),
            )),
            0x4b => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Rel(IRelOp::Gt(SignExtension::Unsigned)),
            )),
            0x4c => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Rel(IRelOp::Le(SignExtension::Signed)),
            )),
            0x4d => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Rel(IRelOp::Le(SignExtension::Unsigned)),
            )),
            0x4e => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Rel(IRelOp::Ge(SignExtension::Signed)),
            )),
            0x4f => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Rel(IRelOp::Ge(SignExtension::Unsigned)),
            )),

            // I64 Test Operations
            0x50 => Self::Num(Num::Int(IntTy::I64, NumIOp::Test(ITestOp::Eqz))),

            // I64 Comparision operations
            0x51 => Self::Num(Num::Int(IntTy::I64, NumIOp::Rel(IRelOp::Eq))),
            0x52 => Self::Num(Num::Int(IntTy::I64, NumIOp::Rel(IRelOp::Ne))),
            0x53 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Rel(IRelOp::Lt(SignExtension::Signed)),
            )),
            0x54 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Rel(IRelOp::Lt(SignExtension::Unsigned)),
            )),
            0x55 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Rel(IRelOp::Gt(SignExtension::Signed)),
            )),
            0x56 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Rel(IRelOp::Gt(SignExtension::Unsigned)),
            )),
            0x57 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Rel(IRelOp::Le(SignExtension::Signed)),
            )),
            0x58 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Rel(IRelOp::Le(SignExtension::Unsigned)),
            )),
            0x59 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Rel(IRelOp::Ge(SignExtension::Signed)),
            )),
            0x5a => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Rel(IRelOp::Ge(SignExtension::Unsigned)),
            )),

            // F32 Comparision operations
            0x5b => Self::Num(Num::Float(FloatTy::F32, NumFOp::Rel(FRelOp::Eq))),
            0x5c => Self::Num(Num::Float(FloatTy::F32, NumFOp::Rel(FRelOp::Ne))),
            0x5d => Self::Num(Num::Float(FloatTy::F32, NumFOp::Rel(FRelOp::Lt))),
            0x5e => Self::Num(Num::Float(FloatTy::F32, NumFOp::Rel(FRelOp::Gt))),
            0x5f => Self::Num(Num::Float(FloatTy::F32, NumFOp::Rel(FRelOp::Le))),
            0x60 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Rel(FRelOp::Ge))),

            // F64 Comparision operations
            0x61 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Rel(FRelOp::Eq))),
            0x62 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Rel(FRelOp::Ne))),
            0x63 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Rel(FRelOp::Lt))),
            0x64 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Rel(FRelOp::Gt))),
            0x65 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Rel(FRelOp::Le))),
            0x66 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Rel(FRelOp::Ge))),

            // I32 remaining operations
            0x67 => Self::Num(Num::Int(IntTy::I32, NumIOp::Unary(IUnOp::Clz))),
            0x68 => Self::Num(Num::Int(IntTy::I32, NumIOp::Unary(IUnOp::Ctz))),
            0x69 => Self::Num(Num::Int(IntTy::I32, NumIOp::Unary(IUnOp::PopCnt))),
            0x6a => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::Add))),
            0x6b => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::Sub))),
            0x6c => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::Mul))),
            0x6d => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Binary(IBinOp::Div(SignExtension::Signed)),
            )),
            0x6e => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Binary(IBinOp::Div(SignExtension::Unsigned)),
            )),
            0x6f => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Binary(IBinOp::Rem(SignExtension::Signed)),
            )),
            0x70 => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Binary(IBinOp::Rem(SignExtension::Unsigned)),
            )),
            0x71 => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::And))),
            0x72 => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::Or))),
            0x73 => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::Xor))),
            0x74 => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::Shl))),
            0x75 => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Binary(IBinOp::Shr(SignExtension::Signed)),
            )),
            0x76 => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Binary(IBinOp::Shr(SignExtension::Unsigned)),
            )),
            0x77 => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::Rotl))),
            0x78 => Self::Num(Num::Int(IntTy::I32, NumIOp::Binary(IBinOp::Rotr))),

            // I64 remaining operations
            0x79 => Self::Num(Num::Int(IntTy::I64, NumIOp::Unary(IUnOp::Clz))),
            0x7a => Self::Num(Num::Int(IntTy::I64, NumIOp::Unary(IUnOp::Ctz))),
            0x7b => Self::Num(Num::Int(IntTy::I64, NumIOp::Unary(IUnOp::PopCnt))),
            0x7c => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::Add))),
            0x7d => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::Sub))),
            0x7e => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::Mul))),
            0x7f => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Binary(IBinOp::Div(SignExtension::Signed)),
            )),
            0x80 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Binary(IBinOp::Div(SignExtension::Unsigned)),
            )),
            0x81 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Binary(IBinOp::Rem(SignExtension::Signed)),
            )),
            0x82 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Binary(IBinOp::Rem(SignExtension::Unsigned)),
            )),
            0x83 => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::And))),
            0x84 => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::Or))),
            0x85 => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::Xor))),
            0x86 => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::Shl))),
            0x87 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Binary(IBinOp::Shr(SignExtension::Signed)),
            )),
            0x88 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Binary(IBinOp::Shr(SignExtension::Unsigned)),
            )),
            0x89 => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::Rotl))),
            0x8a => Self::Num(Num::Int(IntTy::I64, NumIOp::Binary(IBinOp::Rotr))),

            // FP32 remaining operations
            0x8b => Self::Num(Num::Float(FloatTy::F32, NumFOp::Unary(FUnOp::Abs))),
            0x8c => Self::Num(Num::Float(FloatTy::F32, NumFOp::Unary(FUnOp::Neg))),
            0x8d => Self::Num(Num::Float(FloatTy::F32, NumFOp::Unary(FUnOp::Ceil))),
            0x8e => Self::Num(Num::Float(FloatTy::F32, NumFOp::Unary(FUnOp::Floor))),
            0x8f => Self::Num(Num::Float(FloatTy::F32, NumFOp::Unary(FUnOp::Trunc))),
            0x90 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Unary(FUnOp::Nearest))),
            0x91 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Unary(FUnOp::Sqrt))),
            0x92 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Binary(FBinOp::Add))),
            0x93 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Binary(FBinOp::Sub))),
            0x94 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Binary(FBinOp::Mul))),
            0x95 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Binary(FBinOp::Div))),
            0x96 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Binary(FBinOp::Min))),
            0x97 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Binary(FBinOp::Max))),
            0x98 => Self::Num(Num::Float(FloatTy::F32, NumFOp::Binary(FBinOp::CopySign))),

            // FP64 remaining operations
            0x99 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Unary(FUnOp::Abs))),
            0x9a => Self::Num(Num::Float(FloatTy::F64, NumFOp::Unary(FUnOp::Neg))),
            0x9b => Self::Num(Num::Float(FloatTy::F64, NumFOp::Unary(FUnOp::Ceil))),
            0x9c => Self::Num(Num::Float(FloatTy::F64, NumFOp::Unary(FUnOp::Floor))),
            0x9d => Self::Num(Num::Float(FloatTy::F64, NumFOp::Unary(FUnOp::Trunc))),
            0x9e => Self::Num(Num::Float(FloatTy::F64, NumFOp::Unary(FUnOp::Nearest))),
            0x9f => Self::Num(Num::Float(FloatTy::F64, NumFOp::Unary(FUnOp::Sqrt))),
            0xa0 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Binary(FBinOp::Add))),
            0xa1 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Binary(FBinOp::Sub))),
            0xa2 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Binary(FBinOp::Mul))),
            0xa3 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Binary(FBinOp::Div))),
            0xa4 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Binary(FBinOp::Min))),
            0xa5 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Binary(FBinOp::Max))),
            0xa6 => Self::Num(Num::Float(FloatTy::F64, NumFOp::Binary(FBinOp::CopySign))),

            // Conversion
            0xa7 => Self::Num(Num::Conversion(CvtOp::I32WrapI64)),
            0xa8 => Self::Num(Num::Conversion(CvtOp::Trunc(
                IntTy::I32,
                FloatTy::F32,
                SignExtension::Signed,
            ))),
            0xa9 => Self::Num(Num::Conversion(CvtOp::Trunc(
                IntTy::I32,
                FloatTy::F32,
                SignExtension::Unsigned,
            ))),
            0xaa => Self::Num(Num::Conversion(CvtOp::Trunc(
                IntTy::I32,
                FloatTy::F64,
                SignExtension::Signed,
            ))),
            0xab => Self::Num(Num::Conversion(CvtOp::Trunc(
                IntTy::I32,
                FloatTy::F64,
                SignExtension::Unsigned,
            ))),
            0xac => Self::Num(Num::Conversion(CvtOp::I64ExtendI32(SignExtension::Signed))),
            0xad => Self::Num(Num::Conversion(CvtOp::I64ExtendI32(
                SignExtension::Unsigned,
            ))),
            0xae => Self::Num(Num::Conversion(CvtOp::Trunc(
                IntTy::I64,
                FloatTy::F32,
                SignExtension::Signed,
            ))),
            0xaf => Self::Num(Num::Conversion(CvtOp::Trunc(
                IntTy::I64,
                FloatTy::F32,
                SignExtension::Unsigned,
            ))),
            0xb0 => Self::Num(Num::Conversion(CvtOp::Trunc(
                IntTy::I64,
                FloatTy::F64,
                SignExtension::Signed,
            ))),
            0xb1 => Self::Num(Num::Conversion(CvtOp::Trunc(
                IntTy::I64,
                FloatTy::F64,
                SignExtension::Unsigned,
            ))),
            0xb2 => Self::Num(Num::Conversion(CvtOp::Convert(
                FloatTy::F32,
                IntTy::I32,
                SignExtension::Signed,
            ))),
            0xb3 => Self::Num(Num::Conversion(CvtOp::Convert(
                FloatTy::F32,
                IntTy::I32,
                SignExtension::Unsigned,
            ))),
            0xb4 => Self::Num(Num::Conversion(CvtOp::Convert(
                FloatTy::F32,
                IntTy::I64,
                SignExtension::Signed,
            ))),
            0xb5 => Self::Num(Num::Conversion(CvtOp::Convert(
                FloatTy::F32,
                IntTy::I64,
                SignExtension::Unsigned,
            ))),
            0xb6 => Self::Num(Num::Conversion(CvtOp::F32DemoteF64)),
            0xb7 => Self::Num(Num::Conversion(CvtOp::Convert(
                FloatTy::F64,
                IntTy::I32,
                SignExtension::Signed,
            ))),
            0xb8 => Self::Num(Num::Conversion(CvtOp::Convert(
                FloatTy::F64,
                IntTy::I32,
                SignExtension::Unsigned,
            ))),
            0xb9 => Self::Num(Num::Conversion(CvtOp::Convert(
                FloatTy::F64,
                IntTy::I64,
                SignExtension::Signed,
            ))),
            0xba => Self::Num(Num::Conversion(CvtOp::Convert(
                FloatTy::F64,
                IntTy::I64,
                SignExtension::Unsigned,
            ))),
            0xbb => Self::Num(Num::Conversion(CvtOp::F64PromoteF32)),
            0xbc => Self::Num(Num::Conversion(CvtOp::I32ReinterpretF32)),
            0xbd => Self::Num(Num::Conversion(CvtOp::I64ReinterpretF64)),
            0xbe => Self::Num(Num::Conversion(CvtOp::F32ReinterpretI32)),
            0xbf => Self::Num(Num::Conversion(CvtOp::F64ReinterpretI64)),
            0xc0 => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Unary(IUnOp::Extend(StorageSize::Size8)),
            )),
            0xc1 => Self::Num(Num::Int(
                IntTy::I32,
                NumIOp::Unary(IUnOp::Extend(StorageSize::Size16)),
            )),
            0xc2 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Unary(IUnOp::Extend(StorageSize::Size8)),
            )),
            0xc3 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Unary(IUnOp::Extend(StorageSize::Size16)),
            )),
            0xc4 => Self::Num(Num::Int(
                IntTy::I64,
                NumIOp::Unary(IUnOp::Extend(StorageSize::Size32)),
            )),

            // Extended Instructions
            0xfc => match decode_u32(reader)? {
                // Numeric Instructions
                0 => Self::Num(Num::Conversion(CvtOp::TruncSat(
                    IntTy::I32,
                    FloatTy::F32,
                    SignExtension::Signed,
                ))),
                1 => Self::Num(Num::Conversion(CvtOp::TruncSat(
                    IntTy::I32,
                    FloatTy::F32,
                    SignExtension::Unsigned,
                ))),
                2 => Self::Num(Num::Conversion(CvtOp::TruncSat(
                    IntTy::I32,
                    FloatTy::F64,
                    SignExtension::Signed,
                ))),
                3 => Self::Num(Num::Conversion(CvtOp::TruncSat(
                    IntTy::I32,
                    FloatTy::F64,
                    SignExtension::Unsigned,
                ))),
                4 => Self::Num(Num::Conversion(CvtOp::TruncSat(
                    IntTy::I64,
                    FloatTy::F32,
                    SignExtension::Signed,
                ))),
                5 => Self::Num(Num::Conversion(CvtOp::TruncSat(
                    IntTy::I64,
                    FloatTy::F32,
                    SignExtension::Unsigned,
                ))),
                6 => Self::Num(Num::Conversion(CvtOp::TruncSat(
                    IntTy::I64,
                    FloatTy::F64,
                    SignExtension::Signed,
                ))),
                7 => Self::Num(Num::Conversion(CvtOp::TruncSat(
                    IntTy::I64,
                    FloatTy::F64,
                    SignExtension::Unsigned,
                ))),

                // Memory Instructions
                8 => {
                    let x = DataIndex::decode(reader, ctx)?;
                    match reader.next()? {
                        0x00 => {}
                        _ => return Err(DecodeError::InvalidInstr),
                    }
                    Self::Mem(Mem::MemoryInit(x))
                }
                9 => {
                    let x = DataIndex::decode(reader, ctx)?;
                    Self::Mem(Mem::DataDrop(x))
                }
                10 => {
                    match reader.next()? {
                        0x00 => {}
                        _ => return Err(DecodeError::InvalidInstr),
                    }
                    match reader.next()? {
                        0x00 => {}
                        _ => return Err(DecodeError::InvalidInstr),
                    }

                    Self::Mem(Mem::MemoryCopy)
                }
                11 => {
                    match reader.next()? {
                        0x00 => {}
                        _ => return Err(DecodeError::InvalidInstr),
                    }
                    Self::Mem(Mem::MemoryFill)
                }

                // Table Instructions
                12 => {
                    let y = ElementIndex::decode(reader, ctx)?;
                    let x = TableIndex::decode(reader, ctx)?;
                    Self::Table(Table::TableInit { elem: y, table: x })
                }
                13 => {
                    let x = ElementIndex::decode(reader, ctx)?;
                    Self::Table(Table::ElemDrop(x))
                }
                14 => {
                    let x = TableIndex::decode(reader, ctx)?;
                    let y = TableIndex::decode(reader, ctx)?;
                    Self::Table(Table::TableCopy { x, y })
                }
                15 => {
                    let x = TableIndex::decode(reader, ctx)?;
                    Self::Table(Table::TableGrow(x))
                }
                16 => {
                    let x = TableIndex::decode(reader, ctx)?;
                    Self::Table(Table::TableSize(x))
                }
                17 => {
                    let x = TableIndex::decode(reader, ctx)?;
                    Self::Table(Table::TableFill(x))
                }
                _ => return Err(DecodeError::InvalidInstr),
            },
            0xfd => {
                decode_u32(reader)?;
                // TODO: vector instructions
                unimplemented!("vector instructions")
            }
            _ => return Err(DecodeError::InvalidInstr),
        })
    }

    fn decode_until<R, C>(
        reader: &mut R,
        terms: &[u8],
        ctx: &C,
    ) -> Result<(Vec<Self>, u8), DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext
            + FunctionsContext
            + TablesContext
            + GlobalsContext
            + ElementsContext
            + DataContext,
    {
        debug_assert!(!terms.is_empty());

        let mut instrs = Vec::new();
        let mut op_code;

        loop {
            op_code = reader.next()?;
            if terms.contains(&op_code) {
                break;
            }

            let instr = Self::decode_with_op_code(op_code, reader, ctx)?;
            instrs.push(instr);
        }

        Ok((instrs, op_code))
    }
}

impl ConstInstr {
    fn decode_with_op_code<R, C>(
        op_code: u8,
        reader: &mut R,
        ctx: &C,
        validator: &mut ConstExprValidator,
    ) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: ImportGlobalsContext + FunctionsContext,
    {
        use crate::module::instr::Const;

        Ok(match op_code {
            // Reference Instructions
            0xd0 => {
                let t = RefTy::decode(reader)?;

                validator.push_val(OpdTy::Ref(t));

                Self::RefNull(t)
            }
            0xd2 => {
                let x = FuncIndex::decode(reader, ctx)?;

                validator.push_val(OpdTy::Ref(RefTy::FuncRef));

                Self::RefFunc(x)
            }

            // Variable Instructions
            0x23 => {
                let x = ImportGlobalIndex::decode(reader, ctx)?;

                // XXX: This is checking for the import twice since decode above does it.
                let global_ty = ctx.import_global_ty(x).unwrap();
                validator.push_val(global_ty.t.into());

                Self::GlobalGet(x)
            }

            // Numeric Instructions
            0x41 => {
                let n = decode_s32(reader)?;

                validator.push_val(OpdTy::Num(NumTy::I32));

                Self::Constant(Const::I32(n))
            }
            0x42 => {
                let n = decode_s64(reader)?;

                validator.push_val(OpdTy::Num(NumTy::I64));

                Self::Constant(Const::I64(n))
            }
            0x43 => {
                let n = decode_f32(reader)?;

                validator.push_val(OpdTy::Num(NumTy::F32));

                Self::Constant(Const::F32(n))
            }
            0x44 => {
                let n = decode_f64(reader)?;

                validator.push_val(OpdTy::Num(NumTy::F64));

                Self::Constant(Const::F64(n))
            }
            _ => return Err(DecodeError::InvalidConstInstr),
        })
    }

    fn decode_until<R, C>(
        reader: &mut R,
        ctx: &C,
        validator: &mut ConstExprValidator,
    ) -> Result<Vec<Self>, DecodeError<R::Error>>
    where
        R: Read,
        C: ImportGlobalsContext + FunctionsContext,
    {
        let mut instrs = Vec::new();
        let mut op_code;

        loop {
            op_code = reader.next()?;
            if op_code == OP_CODE_END {
                validator.end()?;
                break;
            }

            let instr = Self::decode_with_op_code(op_code, reader, ctx, validator)?;
            instrs.push(instr);
        }

        Ok(instrs)
    }
}

impl Expr {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext
            + FunctionsContext
            + TablesContext
            + GlobalsContext
            + ElementsContext
            + DataContext,
    {
        let (instrs, _) = Instr::decode_until(reader, &[OP_CODE_END], ctx)?;
        Ok(Self { instrs })
    }
}

impl ConstExpr {
    fn decode<R, C>(
        reader: &mut R,
        ctx: &C,
        expected_ty: OpdTy,
    ) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: ImportGlobalsContext + FunctionsContext,
    {
        let mut validator = ConstExprValidator::new(expected_ty);
        let instrs = ConstInstr::decode_until(reader, ctx, &mut validator)?;
        debug_assert_eq!(validator.into_final_ty(), vec![expected_ty]);

        Ok(Self { instrs })
    }
}

impl TypeIndex {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext,
    {
        let idx = Self(decode_u32(reader)?);
        if !ctx.is_type_valid(idx) {
            return Err(DecodeError::InvalidTypeIndex);
        }

        Ok(idx)
    }
}

impl ImportGlobalIndex {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: ImportGlobalsContext,
    {
        let idx = Self(decode_u32(reader)?);
        let Some(import_ty) = ctx.import_global_ty(idx) else {
            return Err(DecodeError::InvalidGlobalIndex);
        };
        match import_ty.m {
            Mut::Const => {}
            Mut::Var => {
                return Err(DecodeError::InvalidGlobalIndex);
            }
        }

        Ok(idx)
    }
}

impl FuncIndex {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: FunctionsContext,
    {
        let idx = Self(decode_u32(reader)?);
        if !ctx.is_func_valid(idx) {
            return Err(DecodeError::InvalidFuncIndex);
        }

        Ok(idx)
    }
}

impl TableIndex {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TablesContext,
    {
        let idx = Self(decode_u32(reader)?);
        if !ctx.is_table_valid(idx) {
            return Err(DecodeError::InvalidTableIndex);
        }

        Ok(idx)
    }
}

impl MemIndex {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: MemsContext,
    {
        let idx = Self(decode_u32(reader)?);
        if !ctx.is_mem_valid(idx) {
            return Err(DecodeError::InvalidMemIndex);
        }

        Ok(idx)
    }
}

impl GlobalIndex {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: GlobalsContext,
    {
        let idx = Self(decode_u32(reader)?);
        if !ctx.is_global_valid(idx) {
            return Err(DecodeError::InvalidGlobalIndex);
        }

        Ok(idx)
    }
}

impl ElementIndex {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: ElementsContext,
    {
        let idx = Self(decode_u32(reader)?);
        if !ctx.is_elem_valid(idx) {
            return Err(DecodeError::InvalidElementIndex);
        }

        Ok(idx)
    }
}

impl DataIndex {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: DataContext,
    {
        let idx = Self(decode_u32(reader)?);
        if !ctx.is_data_valid(idx) {
            return Err(DecodeError::InvalidDataIndex);
        }

        Ok(idx)
    }
}

macro_rules! decode_idx {
    ($t:ty) => {
        impl $t {
            fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
            where
                R: Read,
            {
                Ok(Self(decode_u32(reader)?))
            }
        }
    };
}

decode_idx!(LocalIndex);
decode_idx!(LabelIndex);

impl ImportDesc {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext,
    {
        match reader.next()? {
            0x00 => Ok(Self::Func(TypeIndex::decode(reader, ctx)?)),
            0x01 => Ok(Self::Table(TableTy::decode(reader)?)),
            0x02 => Ok(Self::Mem(MemoryTy::decode(reader)?)),
            0x03 => Ok(Self::Global(GlobalTy::decode(reader)?)),
            _ => Err(DecodeError::InvalidImportDesc),
        }
    }
}

impl Import {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext,
    {
        let module = decode_name(reader)?;
        let nm = decode_name(reader)?;
        let d = ImportDesc::decode(reader, ctx)?;

        Ok(Self {
            module,
            name: nm,
            desc: d,
        })
    }
}

impl Global {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: ImportGlobalsContext + FunctionsContext,
    {
        let gt = GlobalTy::decode(reader)?;
        let e = ConstExpr::decode(reader, ctx, gt.t.into())?;

        Ok(Self { ty: gt, init: e })
    }
}

impl ExportDesc {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: FunctionsContext + TablesContext + MemsContext + GlobalsContext,
    {
        match reader.next()? {
            0x00 => Ok(Self::Func(FuncIndex::decode(reader, ctx)?)),
            0x01 => Ok(Self::Table(TableIndex::decode(reader, ctx)?)),
            0x02 => Ok(Self::Mem(MemIndex::decode(reader, ctx)?)),
            0x03 => Ok(Self::Global(GlobalIndex::decode(reader, ctx)?)),
            _ => Err(DecodeError::InvalidExportDesc),
        }
    }
}

impl Export {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: FunctionsContext + TablesContext + MemsContext + GlobalsContext,
    {
        let nm = decode_name(reader)?;
        let d = ExportDesc::decode(reader, ctx)?;

        Ok(Self { name: nm, desc: d })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ElemKind;

impl ElemKind {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        let b = reader.next()?;
        if b != 0x00 {
            return Err(DecodeError::InvalidElementSegment);
        }
        Ok(ElemKind)
    }
}

impl Elem {
    #[allow(clippy::too_many_lines)]
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: ImportGlobalsContext + FunctionsContext + TablesContext,
    {
        // XXX: Could optimize ConstExpr to hold single instruction instead of Vec<ConstInstr>

        match decode_u32(reader)? {
            0 => {
                let e = ConstExpr::decode(reader, ctx, OpdTy::Num(NumTy::I32))?;
                let y = decode_vec(reader, |r| FuncIndex::decode(r, ctx))?;
                let x = TableIndex(0);
                if !ctx.is_table_valid(x) {
                    return Err(DecodeError::InvalidTableIndex);
                }
                Ok(Self {
                    ty: RefTy::FuncRef,
                    init: y
                        .into_iter()
                        .map(|idx| ConstExpr {
                            instrs: vec![ConstInstr::RefFunc(idx)],
                        })
                        .collect(),
                    mode: ElementSegmentMode::Active(x, e),
                })
            }
            1 => {
                let _ = ElemKind::decode(reader)?;
                let y = decode_vec(reader, |r| FuncIndex::decode(r, ctx))?;
                Ok(Self {
                    ty: RefTy::FuncRef,
                    init: y
                        .into_iter()
                        .map(|idx| ConstExpr {
                            instrs: vec![ConstInstr::RefFunc(idx)],
                        })
                        .collect(),
                    mode: ElementSegmentMode::Passive,
                })
            }
            2 => {
                let x = TableIndex::decode(reader, ctx)?;
                let e = ConstExpr::decode(reader, ctx, OpdTy::Num(NumTy::I32))?;
                let _ = ElemKind::decode(reader)?;
                let y = decode_vec(reader, |r| FuncIndex::decode(r, ctx))?;
                Ok(Self {
                    ty: RefTy::FuncRef,
                    init: y
                        .into_iter()
                        .map(|idx| ConstExpr {
                            instrs: vec![ConstInstr::RefFunc(idx)],
                        })
                        .collect(),
                    mode: ElementSegmentMode::Active(x, e),
                })
            }
            3 => {
                let _ = ElemKind::decode(reader)?;
                let y = decode_vec(reader, |r| FuncIndex::decode(r, ctx))?;
                Ok(Self {
                    ty: RefTy::FuncRef,
                    init: y
                        .into_iter()
                        .map(|idx| ConstExpr {
                            instrs: vec![ConstInstr::RefFunc(idx)],
                        })
                        .collect(),
                    mode: ElementSegmentMode::Declarative,
                })
            }
            4 => {
                let e = ConstExpr::decode(reader, ctx, OpdTy::Num(NumTy::I32))?;
                let el = decode_vec(reader, |r| {
                    ConstExpr::decode(r, ctx, OpdTy::Ref(RefTy::FuncRef))
                })?;
                let x = TableIndex(0);
                if !ctx.is_table_valid(x) {
                    return Err(DecodeError::InvalidTableIndex);
                }
                Ok(Self {
                    ty: RefTy::FuncRef,
                    init: el,
                    mode: ElementSegmentMode::Active(x, e),
                })
            }
            5 => {
                let et = RefTy::decode(reader)?;
                let el = decode_vec(reader, |r| ConstExpr::decode(r, ctx, OpdTy::Ref(et)))?;
                Ok(Self {
                    ty: et,
                    init: el,
                    mode: ElementSegmentMode::Passive,
                })
            }
            6 => {
                let x = TableIndex::decode(reader, ctx)?;
                let e = ConstExpr::decode(reader, ctx, OpdTy::Num(NumTy::I32))?;
                let et = RefTy::decode(reader)?;
                let el = decode_vec(reader, |r| ConstExpr::decode(r, ctx, OpdTy::Ref(et)))?;
                Ok(Self {
                    ty: et,
                    init: el,
                    mode: ElementSegmentMode::Active(x, e),
                })
            }
            7 => {
                let et = RefTy::decode(reader)?;
                let el = decode_vec(reader, |r| ConstExpr::decode(r, ctx, OpdTy::Ref(et)))?;
                Ok(Self {
                    ty: et,
                    init: el,
                    mode: ElementSegmentMode::Declarative,
                })
            }
            _ => Err(DecodeError::InvalidElementSegment),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Locals {
    n: u32,
    t: ValTy,
}

impl Locals {
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        let n = decode_u32(reader)?;
        let t = ValTy::decode(reader)?;
        Ok(Self { n, t })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Func {
    t: Vec<ValTy>,
    e: Expr,
}

impl Func {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext
            + FunctionsContext
            + TablesContext
            + GlobalsContext
            + ElementsContext
            + DataContext,
    {
        // XXX: Can optimize this by decoding into the concatenated Vec immediately

        let locals = decode_vec(reader, Locals::decode)?;
        let e = Expr::decode(reader, ctx)?;

        let capacity = usize::try_from(locals.iter().map(|l| u64::from(l.n)).sum::<u64>()).unwrap();
        let mut t = Vec::with_capacity(capacity);
        for l in locals {
            t.resize(t.len() + usize::try_from(l.n).unwrap(), l.t);
        }

        Ok(Self { t, e })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Code {
    code: Func,
}

impl Code {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: TypesContext
            + FunctionsContext
            + TablesContext
            + GlobalsContext
            + ElementsContext
            + DataContext,
    {
        let size = decode_u32(reader)?;
        let expected_pos = reader.pos() + u64::from(size);

        let code = Func::decode(reader, ctx)?;

        // Early detection of error
        if reader.pos() != expected_pos {
            return Err(DecodeError::InvalidModule);
        }

        Ok(Self { code })
    }
}

impl Data {
    fn decode<R, C>(reader: &mut R, ctx: &C) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
        C: ImportGlobalsContext + FunctionsContext + MemsContext,
    {
        match decode_u32(reader)? {
            0 => {
                let e = ConstExpr::decode(reader, ctx, OpdTy::Num(NumTy::I32))?;
                let b = decode_bytes_vec(reader)?;
                let x = MemIndex(0);
                if !ctx.is_mem_valid(x) {
                    return Err(DecodeError::InvalidMemIndex);
                }
                Ok(Self {
                    init: b,
                    mode: DataMode::Active(x, e),
                })
            }
            1 => {
                let b = decode_bytes_vec(reader)?;
                Ok(Self {
                    init: b,
                    mode: DataMode::Passive,
                })
            }
            2 => {
                let x = MemIndex::decode(reader, ctx)?;
                let e = ConstExpr::decode(reader, ctx, OpdTy::Num(NumTy::I32))?;
                let b = decode_bytes_vec(reader)?;
                Ok(Self {
                    init: b,
                    mode: DataMode::Active(x, e),
                })
            }
            _ => Err(DecodeError::InvalidData),
        }
    }
}

/// Section index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SectionIndex(u32);

/// Section Ids
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SectionId {
    /// Custom
    Custom,
    /// Type
    Type,
    /// Import
    Import,
    /// Function
    Function,
    /// Table
    Table,
    /// Memory
    Memory,
    /// Global
    Global,
    /// Export
    Export,
    /// Start
    Start,
    /// Element
    Element,
    /// Code
    Code,
    /// Data
    Data,
    /// Data Count
    DataCount,
    /// Unknown section
    Unknown(u8),
}

impl From<u8> for SectionId {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Custom,
            1 => Self::Type,
            2 => Self::Import,
            3 => Self::Function,
            4 => Self::Table,
            5 => Self::Memory,
            6 => Self::Global,
            7 => Self::Export,
            8 => Self::Start,
            9 => Self::Element,
            10 => Self::Code,
            11 => Self::Data,
            12 => Self::DataCount,
            _ => Self::Unknown(value),
        }
    }
}

impl SectionId {
    #[must_use]
    pub(crate) fn is_valid_order(last: Option<SectionId>, cur: SectionId) -> bool {
        use SectionId::{
            Code, Custom, Data, DataCount, Element, Export, Function, Global, Import, Memory,
            Start, Table, Type, Unknown,
        };

        const ORDER: &[SectionId] = &[
            Type, Import, Function, Table, Memory, Global, Export, Start, Element, DataCount, Code,
            Data,
        ];

        let Some(last) = last else {
            return true;
        };

        match last {
            Type | Import | Function | Table | Memory | Global | Export | Start | Element
            | Code | Data | DataCount => {}
            Custom | Unknown(_) => unreachable!(),
        }

        match cur {
            Custom | Unknown(_) => {
                return true;
            }
            Type | Import | Function | Table | Memory | Global | Export | Start | Element
            | Code | Data | DataCount => {}
        }

        let last_pos = ORDER.iter().position(|id| *id == last);
        let cur_pos = ORDER.iter().position(|id| *id == cur);

        last_pos <= cur_pos
    }
}

/// Magic number in preamble
pub const MAGIC: &[u8] = &[0x00, 0x61, 0x73, 0x6D];

/// Version field
pub const VERSION: &[u8] = &[0x01, 0x00, 0x00, 0x00];

impl Module {
    #[allow(clippy::too_many_lines)]
    fn decode<R>(reader: &mut R) -> Result<Self, DecodeError<R::Error>>
    where
        R: Read,
    {
        expect_bytes(reader, MAGIC)?;
        expect_bytes(reader, VERSION)?;

        let mut last_non_custom_sec_id = None;

        let mut types = Vec::new();
        let mut imports = Vec::new();
        let mut import_funcs = Vec::new();
        let mut import_tables = Vec::new();
        let mut import_mems = Vec::new();
        let mut import_globals = Vec::new();
        let mut type_idxs = Vec::new();
        let mut tables = Vec::new();
        let mut mems = Vec::new();
        let mut globals = Vec::new();
        let mut exports = Vec::new();
        let mut start = None;
        let mut elems = Vec::new();
        let mut datacount = None;
        let mut code = Vec::new();
        let mut datas = Vec::new();

        loop {
            let sec_id = match reader.next() {
                Ok(k) => SectionId::from(k),
                Err(e) if e.is_eof() => break,
                Err(e) => return Err(DecodeError::from(e)),
            };
            let sec_size = decode_u32(reader)?;
            if !SectionId::is_valid_order(last_non_custom_sec_id, sec_id) {
                return Err(DecodeError::InvalidSection);
            }

            let expected_pos = reader.pos() + u64::from(sec_size);

            match sec_id {
                SectionId::Unknown(_) => return Err(DecodeError::InvalidSection),
                SectionId::Custom => {
                    reader.skip(u64::from(sec_size))?;
                    continue;
                }
                SectionId::Type => {
                    types = decode_vec(reader, FuncTy::decode)?;
                }
                SectionId::Import => {
                    let ctx = ImportSectionValidator { types: &types };
                    imports = decode_vec(reader, |r| Import::decode(r, &ctx))?;

                    import_funcs = Vec::new();
                    import_tables = Vec::new();
                    import_mems = Vec::new();
                    import_globals = Vec::new();

                    for i in &imports {
                        match i.desc {
                            ImportDesc::Func(ty_idx) => {
                                import_funcs.push(ty_idx);
                            }
                            ImportDesc::Table(tbl) => {
                                import_tables.push(tbl);
                            }
                            ImportDesc::Mem(m) => {
                                import_mems.push(m);
                            }
                            ImportDesc::Global(g) => {
                                import_globals.push(g);
                            }
                        }
                    }
                }
                SectionId::Function => {
                    let ctx = FunctionSectionValidator { types: &types };
                    type_idxs = decode_vec(reader, |r| TypeIndex::decode(r, &ctx))?;
                }
                SectionId::Table => {
                    tables = decode_vec(reader, TableTy::decode)?
                        .into_iter()
                        .map(|tt| Table { ty: tt })
                        .collect();
                }
                SectionId::Memory => {
                    mems = decode_vec(reader, MemoryTy::decode)?
                        .into_iter()
                        .map(|mt| Mem { ty: mt })
                        .collect();
                    if mems.len() > 1 {
                        return Err(DecodeError::InvalidMemCount);
                    }
                }
                SectionId::Global => {
                    let ctx = GlobalSectionValidator {
                        import_funcs: &import_funcs,
                        import_globals: &import_globals,
                        type_idxs: &type_idxs,
                    };
                    globals = decode_vec(reader, |r| Global::decode(r, &ctx))?;
                }
                SectionId::Export => {
                    let ctx = ExportsSectionValidator {
                        import_funcs: &import_funcs,
                        import_tables: &import_tables,
                        import_mems: &import_mems,
                        import_globals: &import_globals,
                        type_idxs: &type_idxs,
                        tables: &tables,
                        mems: &mems,
                        globals: &globals,
                    };
                    exports = decode_vec(reader, |r| Export::decode(r, &ctx))?;
                    // TODO: Validate all export names are unique
                }
                SectionId::Start => {
                    let ctx = StartSectionValidator {
                        types: &types,
                        import_funcs: &import_funcs,
                        type_idxs: &type_idxs,
                    };
                    let idx = FuncIndex::decode(reader, &ctx)?;
                    let Some(ty_idx) = ctx.type_index(idx) else {
                        return Err(DecodeError::InvalidStart);
                    };
                    let Some(func_ty) = ctx.func_ty(ty_idx) else {
                        return Err(DecodeError::InvalidStart);
                    };
                    if !func_ty.is_params_empty() || !func_ty.is_return_empty() {
                        return Err(DecodeError::InvalidStart);
                    }
                    start = Some(idx);
                }
                SectionId::Element => {
                    let ctx = ElementsSectionValidator {
                        import_funcs: &import_funcs,
                        import_tables: &import_tables,
                        import_globals: &import_globals,
                        type_idxs: &type_idxs,
                        tables: &tables,
                    };
                    elems = decode_vec(reader, |r| Elem::decode(r, &ctx))?;
                }
                SectionId::DataCount => {
                    datacount = Some(decode_u32(reader)?);
                }
                SectionId::Code => {
                    let ctx = CodeSectionValidator {
                        types: &types,
                        import_funcs: &import_funcs,
                        import_tables: &import_tables,
                        import_globals: &import_globals,
                        type_idxs: &type_idxs,
                        tables: &tables,
                        globals: &globals,
                        elems: &elems,
                        data_len: datacount,
                    };
                    code = decode_vec(reader, |r| Code::decode(r, &ctx))?;

                    // Early exit possible, but still need to check at the end.
                    if type_idxs.len() != code.len() {
                        return Err(DecodeError::InvalidModule);
                    }
                }
                SectionId::Data => {
                    // XXX: If data section exists, then datacount MUST exist in Wasm 2.

                    let ctx = DataSectionValidator {
                        import_funcs: &import_funcs,
                        import_mems: &import_mems,
                        import_globals: &import_globals,
                        type_idxs: &type_idxs,
                        mems: &mems,
                    };
                    datas = decode_vec(reader, |r| Data::decode(r, &ctx))?;
                    // XXX: Data count MUST match with the datas size in Wasm 2
                    if let Some(datacount) = datacount {
                        if datas.len() != usize::try_from(datacount).unwrap() {
                            return Err(DecodeError::InvalidData);
                        }
                    }
                }
            }

            last_non_custom_sec_id = Some(sec_id);

            // Early detection of error
            if reader.pos() != expected_pos {
                return Err(DecodeError::InvalidSection);
            }
        }

        if type_idxs.len() != code.len() {
            return Err(DecodeError::InvalidModule);
        }

        let funcs = type_idxs
            .into_iter()
            .zip(code)
            .map(|(ty, code)| crate::module::Func {
                ty,
                locals: code.code.t,
                body: code.code.e,
            })
            .collect();

        Ok(Self {
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
        })
    }
}

use super::{Read, ReadError};

#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
enum DecodeError<E> {
    InvalidNum,
    InvalidRefTy,
    InvalidValueTy,
    InvalidFuncTy,
    InvalidLimits,
    InvalidMemoryTy,
    InvalidInstr,
    InvalidConstInstr,
    InvalidMut,
    InvalidImportDesc,
    InvalidExportDesc,
    InvalidElementSegment,
    InvalidData,
    InvalidSection,
    InvalidTypeIndex,
    InvalidFuncIndex,
    InvalidTableIndex,
    InvalidMemIndex,
    InvalidGlobalIndex,
    InvalidElementIndex,
    InvalidDataIndex,
    InvalidPreamble,
    InvalidModule,
    InvalidMemCount,
    InvalidStart,
    InvalidExpr(ExprError),
    InvalidName(FromUtf8Error),
    Read(ReadError<E>),
}

impl<E> From<FromUtf8Error> for DecodeError<E> {
    fn from(value: FromUtf8Error) -> Self {
        Self::InvalidName(value)
    }
}

impl<E> From<ExprError> for DecodeError<E> {
    fn from(value: ExprError) -> Self {
        Self::InvalidExpr(value)
    }
}

impl<E> From<ReadError<E>> for DecodeError<E> {
    fn from(value: ReadError<E>) -> Self {
        Self::Read(value)
    }
}

/// Error during decoding
#[derive(Debug)]
pub struct Error<E> {
    /// Inner error
    inner: DecodeError<E>,
    /// Byte offset read
    pos: u64,
}

impl<E> Error<E> {
    /// Byte offset when the error occurred
    #[inline]
    #[must_use]
    pub fn pos(&self) -> u64 {
        self.pos
    }
}

impl<E> fmt::Display for Error<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            DecodeError::InvalidNum => f.write_str("invalid number"),
            DecodeError::InvalidRefTy => f.write_str("invalid reference type"),
            DecodeError::InvalidValueTy => f.write_str("invalid value type"),
            DecodeError::InvalidFuncTy => f.write_str("invalid function type"),
            DecodeError::InvalidLimits => f.write_str("invalid limits"),
            DecodeError::InvalidMemoryTy => f.write_str("invalid memory type"),
            DecodeError::InvalidInstr => f.write_str("invalid instruction"),
            DecodeError::InvalidConstInstr => f.write_str("invalid constant instruction"),
            DecodeError::InvalidMut => f.write_str("invalid mutability"),
            DecodeError::InvalidImportDesc => f.write_str("invalid import description"),
            DecodeError::InvalidExportDesc => f.write_str("invalid export description"),
            DecodeError::InvalidElementSegment => f.write_str("invalid element segment"),
            DecodeError::InvalidData => f.write_str("invalid data section"),
            DecodeError::InvalidTypeIndex => f.write_str("invalid type index"),
            DecodeError::InvalidFuncIndex => f.write_str("invalid function index"),
            DecodeError::InvalidTableIndex => f.write_str("invalid table index"),
            DecodeError::InvalidMemIndex => f.write_str("invalid memory index"),
            DecodeError::InvalidGlobalIndex => f.write_str("invalid global index"),
            DecodeError::InvalidElementIndex => f.write_str("invalid element index"),
            DecodeError::InvalidDataIndex => f.write_str("invalid data index"),
            DecodeError::InvalidSection => f.write_str("invalid section"),
            DecodeError::InvalidPreamble => f.write_str("invalid preamble"),
            DecodeError::InvalidModule => f.write_str("invalid module"),
            DecodeError::InvalidMemCount => f.write_str("greater than 1 memory"),
            DecodeError::InvalidStart => f.write_str("invalid start function"),
            DecodeError::InvalidExpr(e) => fmt::Display::fmt(e, f),
            DecodeError::InvalidName(e) => fmt::Display::fmt(e, f),
            DecodeError::Read(e) => fmt::Display::fmt(e, f),
        }
    }
}

#[cfg(feature = "std")]
impl<E> error::Error for Error<E>
where
    E: error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.inner {
            DecodeError::InvalidNum
            | DecodeError::InvalidRefTy
            | DecodeError::InvalidValueTy
            | DecodeError::InvalidFuncTy
            | DecodeError::InvalidLimits
            | DecodeError::InvalidMemoryTy
            | DecodeError::InvalidInstr
            | DecodeError::InvalidConstInstr
            | DecodeError::InvalidMut
            | DecodeError::InvalidImportDesc
            | DecodeError::InvalidExportDesc
            | DecodeError::InvalidElementSegment
            | DecodeError::InvalidData
            | DecodeError::InvalidTypeIndex
            | DecodeError::InvalidFuncIndex
            | DecodeError::InvalidTableIndex
            | DecodeError::InvalidMemIndex
            | DecodeError::InvalidGlobalIndex
            | DecodeError::InvalidElementIndex
            | DecodeError::InvalidDataIndex
            | DecodeError::InvalidSection
            | DecodeError::InvalidPreamble
            | DecodeError::InvalidModule
            | DecodeError::InvalidMemCount
            | DecodeError::InvalidStart => None,
            DecodeError::InvalidExpr(e) => Some(e),
            DecodeError::InvalidName(e) => Some(e),
            DecodeError::Read(e) => Some(e),
        }
    }
}

/// Decodes a Wasm module in binary format.
///
/// # Errors
///
/// Returns an error if the data is invalid or if a problem occurs when reading.
fn decode<R: Read>(mut reader: R) -> Result<Module, Error<R::Error>> {
    Module::decode(&mut reader).map_err(|e| Error {
        inner: e,
        pos: reader.pos(),
    })
}

/// Decodes a Wasm module in binary format.
///
/// # Errors
///
/// Returns an error if the data is invalid or if a problem occurs when reading.
#[cfg(feature = "std")]
pub fn from_slice(s: &[u8]) -> Result<Module, Error<super::OutOfBoundsError>> {
    decode(super::SliceRead::new(s))
}

/// Decodes a Wasm module in binary format.
///
/// # Errors
///
/// Returns an error if the data is invalid or if a problem occurs when reading.
#[cfg(feature = "std")]
pub fn from_reader<R>(r: R) -> Result<Module, Error<io::Error>>
where
    R: io::Read,
{
    decode(super::IoRead::new(r))
}