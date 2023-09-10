//! Instructions for code.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::{
    ty::{NumTy, RefTy, ValTy},
    DataIndex, ElementIndex, FuncIndex, GlobalIndex, LabelIndex, LocalIndex, TableIndex, TypeIndex,
};

/// Sign extension mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignExtension {
    /// Signed
    Signed,
    /// Unsigned
    Unsigned,
}

/// Storage size target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageSize {
    Size8,
    Size16,
    Size32,
}

/// Integer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntTy {
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
}

/// Floating point type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatTy {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
}

/// Numeric constant operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Const {
    /// 0x41
    I32(i32),
    /// 0x42
    I64(i64),
    /// 0x43
    F32(f32),
    /// 0x44
    F64(f64),
}

/// Integer unary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IUnOp {
    /// Count leading zeroes bits
    Clz,
    /// Count trailing zero bits
    Ctz,
    /// Return count of non-zero bits
    PopCnt,
    /// 0xc0, 0xc1, 0xc2, 0xc3, 0xc4
    Extend(StorageSize),
}

/// Integer binary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IBinOp {
    /// Add
    Add,
    /// Subtract
    Sub,
    /// Multiply
    Mul,
    /// Divide with sign extension
    Div(SignExtension),
    /// Remainder with sign extension
    Rem(SignExtension),
    /// And
    And,
    /// Or
    Or,
    /// Xor
    Xor,
    /// Shift left
    Shl,
    /// Shift right with sign extension
    Shr(SignExtension),
    /// Rotate left
    Rotl,
    /// Rotate right
    Rotr,
}

/// Floating point unary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FUnOp {
    /// Absolute value
    Abs,
    /// Negate
    Neg,
    /// Square root
    Sqrt,
    /// Ceiling
    Ceil,
    /// Floor
    Floor,
    /// Truncate
    Trunc,
    /// Nearest
    Nearest,
}

/// Floating point binary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FBinOp {
    /// Add
    Add,
    /// Subtract
    Sub,
    /// Multiply
    Mul,
    /// Divide
    Div,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// If same sign, return value as-is. If different sign, return with negated sign.
    CopySign,
}

/// Integer test operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ITestOp {
    /// Return 1 if value is zero, 0 otherwise.
    Eqz,
}

/// Integer comparision operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IRelOp {
    /// Equal
    Eq,
    /// Not Equal
    Ne,
    /// Less Than with Sign Extension
    Lt(SignExtension),
    /// Greater Than with Sign Extension
    Gt(SignExtension),
    /// Less Than or Equal with Sign Extension
    Le(SignExtension),
    /// Greater Than or Equal with Sign Extension
    Ge(SignExtension),
}

/// Floating point ccmparision operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FRelOp {
    /// Equal
    Eq,
    /// Not Equal
    Ne,
    /// Less Than
    Lt,
    /// Greater Than
    Gt,
    /// Less Than or Equal
    Le,
    /// Greater Than or Equal
    Ge,
}

/// Conversion operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvtOp {
    /// 0xa7
    I32WrapI64,
    /// 0xa8, 0xa9, 0xaa, 0xab, 0xae, 0xaf, 0xb0, 0xb1
    Trunc(IntTy, FloatTy, SignExtension),
    /// 0xac, 0xad
    I64ExtendI32(SignExtension),
    /// 0xb2, 0xb3, 0xb4, 0xb5, 0xb7, 0xb8, 0xb9, 0xba
    Convert(FloatTy, IntTy, SignExtension),
    /// 0xb6
    F32DemoteF64,
    /// 0xbb
    F64PromoteF32,
    /// 0xbc
    I32ReinterpretF32,
    /// 0xbd
    I64ReinterpretF64,
    /// 0xbe
    F32ReinterpretI32,
    /// 0xbf
    F64ReinterpretI64,
    /// 0xfc 0-7
    TruncSat(IntTy, FloatTy, SignExtension),
}

/// Integer operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumIOp {
    /// Unary operation
    Unary(IUnOp),
    /// Binary operation
    Binary(IBinOp),
    /// Test operation
    Test(ITestOp),
    /// Relative comparision
    Rel(IRelOp),
}

/// Floating point operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumFOp {
    /// Unary operation
    Unary(FUnOp),
    /// Binary operation
    Binary(FBinOp),
    /// Relative comparision
    Rel(FRelOp),
}

/// Numeric operation instruction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    /// Constant
    Constant(Const),
    /// Integer
    Int(IntTy, NumIOp),
    /// Floating point
    Float(FloatTy, NumFOp),
    /// Conversion operations
    Conversion(CvtOp),
}

/// Reference instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ref {
    /// 0xd0
    RefNull(RefTy),
    /// 0xd1
    RefIsNull,
    /// 0xd2
    RefFunc(FuncIndex),
}

/// Parametric instruction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Parametric {
    /// 0x1a
    Drop,
    /// 0x1b and 0x1c
    Select(Option<Vec<ValTy>>),
}

/// Variable instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variable {
    /// 0x20
    LocalGet(LocalIndex),
    /// 0x21
    LocalSet(LocalIndex),
    /// 0x22
    LocalTee(LocalIndex),
    /// 0x23
    GlobalGet(GlobalIndex),
    /// 0x24
    GlobalSet(GlobalIndex),
}

/// Table instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Table {
    /// 0x25
    TableGet(TableIndex),
    /// 0x26
    TableSet(TableIndex),
    /// 0xfc 12
    TableInit {
        elem: ElementIndex,
        table: TableIndex,
    },
    /// 0xfc 13
    ElemDrop(ElementIndex),
    /// 0xfc 14
    TableCopy { x: TableIndex, y: TableIndex },
    /// 0xfc 15
    TableGrow(TableIndex),
    /// 0xfc 16
    TableSize(TableIndex),
    /// 0xfc 17
    TableFill(TableIndex),
}

/// Memory operation arguments
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemArg {
    pub align: u32,
    pub offset: u32,
}

/// Memory instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mem {
    /// 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35
    Load(NumTy, MemArg, Option<(StorageSize, SignExtension)>),
    /// 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e
    Store(NumTy, MemArg, Option<StorageSize>),
    /// 0x3F
    MemorySize,
    /// 0x40
    MemoryGrow,
    /// 0xFC 8
    MemoryInit(DataIndex),
    /// 0xFC 9
    DataDrop(DataIndex),
    /// 0xFC 10
    MemoryCopy,
    /// 0xFC 11
    MemoryFill,
}

/// Block type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockTy {
    /// Empty
    Empty,
    /// Value
    Val(ValTy),
    /// Type Index
    Index(TypeIndex),
}

/// Control instruction
#[derive(Debug, Clone, PartialEq)]
pub enum Control {
    /// 0x00
    Unreachable,
    /// 0x01
    Nop,
    /// 0x02
    Block { bt: BlockTy, instrs: Vec<Instr> },
    /// 0x03
    Loop { bt: BlockTy, instrs: Vec<Instr> },
    /// 0x04
    If {
        bt: BlockTy,
        then: Vec<Instr>,
        el: Vec<Instr>,
    },
    /// 0x0c
    Br(LabelIndex),
    /// 0x0d
    BrIf(LabelIndex),
    /// 0x0e
    BrTable {
        table: Vec<LabelIndex>,
        idx: LabelIndex,
    },
    /// 0x0f
    Return,
    /// 0x10
    Call(FuncIndex),
    /// 0x11
    CallIndirect { y: TypeIndex, x: TableIndex },
}

/// Instruction
#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    /// Control instruction
    Control(Control),
    /// Reference instruction
    Ref(Ref),
    /// Parametric instruction
    Parametric(Parametric),
    /// Variable instruction
    Var(Variable),
    /// Table instruction
    Table(Table),
    /// Memory instruction
    Mem(Mem),
    /// Numeric instructions
    Num(Num),
}

/// Expressions terminated with an explicit 0x0b opcode for end.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub instrs: Vec<Instr>,
}
