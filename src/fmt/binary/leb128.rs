//! LEB128 decoding functions

// If the decode leb128 functions were extracted, one possible signature could be:
// fn $func<F>(next: <F>) -> Result<$num_ty, DecodeError<E>> where F: Fn() -> Result<u8, E>;

use crate::fmt::{binary::DecodeError, Read};

macro_rules! decode_unsigned_leb128 {
    ($func:ident, $num_ty:ty, $bits:literal) => {
        /// Decode an unsigned LEB128 number.
        pub(super) fn $func<R>(reader: &mut R) -> Result<$num_ty, DecodeError<R::Error>>
        where
            R: Read,
        {
            const BITS: u32 = $bits;

            let n = reader.next()?;
            if n & 0x80 == 0 {
                return Ok(<$num_ty>::from(n));
            }

            let mut result = <$num_ty>::from(n & 0x7f);
            let mut shift = 7;
            loop {
                let n = reader.next()?;

                // If unnecessary bits are set (the bits would be dropped when
                // the value is shifted), then return an error.
                //
                // This error may be too strict.
                //
                // There should be at least a simple check to quickly
                // determine that the decoding has failed instead of
                // misinterpreting further data.
                //
                // For a less strict check, the && condition could be:
                //
                // (n & 0x80) != 0
                //
                // Another stricter condition is if the last byte has a 0 value.
                // The encoding is correct but not the minimal number of bytes
                // was used to express the final value.
                if shift == BITS - (BITS % 7) && n >= 1 << (BITS % 7) {
                    return Err(DecodeError::InvalidNum);
                }

                if n & 0x80 == 0 {
                    result |= <$num_ty>::from(n) << shift;
                    return Ok(result);
                }

                result |= <$num_ty>::from(n & 0x7f) << shift;
                shift += 7;
            }
        }
    };
}

decode_unsigned_leb128!(decode_u32, u32, 32);

macro_rules! decode_signed_leb128 {
    ($func:ident, $num_ty:ty, $bits:literal) => {
        /// Decode a signed LEB128 number.
        pub(super) fn $func<R>(reader: &mut R) -> Result<$num_ty, DecodeError<R::Error>>
        where
            R: Read,
        {
            const BITS: u32 = $bits;

            let mut result = 0;
            let mut shift = 0;
            let mut n;

            loop {
                n = reader.next()?;
                let more = n & 0x80 != 0;

                // For the last valid shift, perform some checks to ensure the
                // encoding is valid.
                //
                // Notably, the one bit that MUST NOT be set is the high order bit
                // indicating there are more bytes to decode.
                //
                // For a signed integer, depending on if the value is positive or negative,
                // some bits SHOULD or SHOULD NOT be set.
                //
                // The expectation is that if this is a negative number, then
                // there should have been a sign extension so that all the bits
                // greater than the highest order bit is a 1.
                //
                // 32-bit
                // ------
                //
                // The maximum shift value is 28 meaning a 32-bit number is
                // encoded in a maximum of 5 bytes. If the shift value is 35 or
                // greater, then, the byte's value will be shifted out beyond the
                // 32-bit value.
                //
                // With 28 being the highest valid shift value, the highest
                // order relevant bit in the final byte should be 0x08 or:
                //
                // 0000 1000
                //
                // Any higher bit is "lost" during the bitshift.
                //
                // Due to the encoding rules and two's complement, if the
                // highest order relevant bit is set, then the number is
                // negative and the `1` is extended to the higher bits like:
                //
                // 0111 1000
                //
                // Note that the highest order bit (the first bit from left to right)
                // MUST BE a 0. It is the bit which indicates more bytes should
                // be processed. For the maximum final byte (byte #5 for a
                // 32-bit number)), it MUST be 0. There are no additional bytes
                // to decode.
                //
                // If the highest order relevant bit is not set, then the
                // integer is positive. Any of the lower bits can be set.
                //
                // 0000 0111
                //
                // So the conditions to check are:
                //
                // 1. The highest order bit is not set (so there are no more
                //    bytes to decode). If it is set, the encoding is invalid.
                //    This is the "more" check.
                //
                // 2. Determine if any sign extended negative bit is set.
                //    So is any bit in:
                //
                //    0111 1000
                //
                //    set. If none of the bits are set, then the number is
                //    positive, and the encoding is valid.
                //    This is the "(n & mask != 0)" check.
                // 3. If any sign extended negative bits are set, the number is
                //    negative, and ALL of the bits MUST be set for a valid negative number.
                //    This is the "(n < mask)"" check.
                //    An equivalent check would be that "(n < mask) || (n >= 0x80)"
                //    But the earlier check for "more" removes the need for the additional check.
                //
                //    The check could also be "(n & mask) != mask".
                //
                // Another stricter condition is if the last byte has a 0 value.
                // The encoding is correct but not the minimal number of bytes
                // was used to express the final value.
                if shift == BITS - (BITS % 7) {
                    let mask = ((-1i8 << ((BITS % 7).saturating_sub(1))) & 0x7f) as u8;
                    if more || (n & mask != 0 && n < mask) {
                        return Err(DecodeError::InvalidNum);
                    }
                }

                result |= <$num_ty>::from(n & 0x7f) << shift;
                shift += 7;

                if !more {
                    break;
                }
            }

            if shift < <$num_ty>::BITS && n & 0x40 != 0 {
                result |= -1 << shift;
            }

            Ok(result)
        }
    };
}

decode_signed_leb128!(decode_s32, i32, 32);
decode_signed_leb128!(decode_s33_block_ty, i64, 33);
decode_signed_leb128!(decode_s64, i64, 64);

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    use proptest::prelude::*;

    use crate::fmt::SliceRead;

    decode_unsigned_leb128!(decode_u64, u64, 64);

    macro_rules! encode_unsigned_leb128 {
        ($func:ident, $num_ty:ty) => {
            /// Encodes a number to a signed LEB128 format.
            fn $func(mut value: $num_ty) -> Vec<u8> {
                let mut result = Vec::new();

                loop {
                    let mut b = u8::try_from(value & 0x7f).unwrap();
                    value >>= 7;

                    let done = value == 0;

                    if !done {
                        b |= 0x80;
                    }
                    result.push(b);

                    if done {
                        return result;
                    }
                }
            }
        };
    }

    encode_unsigned_leb128!(encode_u32, u32);
    encode_unsigned_leb128!(encode_u64, u64);

    macro_rules! encode_signed_leb128 {
        ($func:ident, $num_ty:ty) => {
            /// Encodes a number to a signed LEB128 format.
            fn $func(mut value: $num_ty) -> Vec<u8> {
                let mut result = Vec::new();

                loop {
                    let b = u8::try_from(value & 0x7f).unwrap();
                    value >>= 7;

                    if (value == 0 && b & 0x40 == 0) || (value == -1 && (b & 0x40) != 0) {
                        result.push(b);
                        return result;
                    }

                    result.push(b | 0x80);
                }
            }
        };
    }

    encode_signed_leb128!(encode_s32, i32);
    encode_signed_leb128!(encode_s64, i64);

    #[test]
    fn test_decode_u32() {
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0x0f];
        let result = decode_u32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(u32::MAX));

        let bytes = &[0x00];
        let result = decode_u32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(u32::MIN));

        // Valid but in-efficient way to encode 0.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x00];
        let result = decode_u32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(u32::MIN));
    }

    #[test]
    fn test_decode_u32_errors() {
        // Maximum of 5 bytes encoding, the 0x80 bit must not be set.
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0x8f];
        let result = decode_u32(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // Maximum of 5 bytes encoding, the 0x80 bit must not be set.
        // Ensure error is an invalid num instead of an EOF error.
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0x8f, 0x00];
        let result = decode_u32(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // Parts of 0x1f (0x10) will be shifted out of the final value and lost.
        // This may too strict of a check since it could be ok.
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0x1f];
        let result = decode_u32(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));
    }

    #[test]
    fn test_decode_u64() {
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        let result = decode_u64(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(u64::MAX));

        let bytes = &[0x00];
        let result = decode_u64(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(u64::MIN));

        // Valid but in-efficient way to encode 0.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        let result = decode_u64(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(u64::MIN));
    }

    #[test]
    fn test_decode_u64_errors() {
        // Maximum of 10 bytes encoding, the 0x80 bit must not be set in the final byte.
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x81];
        let result = decode_u64(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // Maximum of 10 bytes encoding, the 0x80 bit must not be set.
        // Ensure error is an invalid num instead of an EOF error.
        let bytes = &[
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x81, 0x00,
        ];
        let result = decode_u64(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // 0x02 will be shifted out of the final value and lost.
        // This may too strict of a check since it could be ok.
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02];
        let result = decode_u64(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));
    }

    #[test]
    fn test_decode_s32() {
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0x07];
        let result = decode_s32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(i32::MAX));

        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x78];
        let result = decode_s32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(i32::MIN));

        let bytes = &[0x00];
        let result = decode_s32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(0));

        // Valid but in-efficient way to encode 0.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x00];
        let result = decode_s32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(0));

        let bytes = &[0x40];
        let result = decode_s32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(-64));

        // Valid but in-efficient way to encode -64.
        let bytes = &[0xc0, 0x7f];
        let result = decode_s32(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(-64));
    }

    #[test]
    fn test_decode_s32_errors() {
        // Maximum of 5 bytes encoding, the 0x80 bit must not be set in the final byte.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80];
        let result = decode_s32(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // Maximum of 5 bytes encoding, the 0x80 bit must not be set.
        // Ensure error is an invalid num instead of an EOF error.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        let result = decode_s32(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // If the highest valid bit is set, it should be sign extended. (final byte should be 0x78)
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x08];
        let result = decode_s32(&mut SliceRead::new(bytes.as_slice()));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // If the highest valid bit is set, it should be sign extended. (final byte should be 0x78)
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x38];
        let result = decode_s32(&mut SliceRead::new(bytes.as_slice()));
        assert_eq!(result, Err(DecodeError::InvalidNum));
    }

    #[test]
    fn test_decode_s33() {
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0x0f];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(i64::from(u32::MAX)));

        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x70];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(i64::from(i32::MIN) * 2));

        let bytes = &[0x00];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(0));

        // Valid but in-efficient way to encode 0.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x00];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(0));

        let bytes = &[0x40];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(-64));

        // Valid but in-efficient way to encode -64.
        let bytes = &[0xc0, 0x7f];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(-64));
    }

    #[test]
    fn test_decode_s33_errors() {
        // Maximum of 5 bytes encoding, the 0x80 bit must not be set in the final byte.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // Maximum of 5 bytes encoding, the 0x80 bit must not be set.
        // Ensure error is an invalid num instead of an EOF error.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // If the highest valid bit is set, it should be sign extended. (final byte should be 0x70)
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x10];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes.as_slice()));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // If the highest valid bit is set, it should be sign extended. (final byte should be 0x70)
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x30];
        let result = decode_s33_block_ty(&mut SliceRead::new(bytes.as_slice()));
        assert_eq!(result, Err(DecodeError::InvalidNum));
    }

    #[test]
    fn test_decode_s64() {
        let bytes = &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00];
        let result = decode_s64(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(i64::MAX));

        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f];
        let result = decode_s64(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(i64::MIN));

        let bytes = &[0x00];
        let result = decode_s64(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(0));

        // Valid but in-efficient way to encode 0.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        let result = decode_s64(&mut SliceRead::new(bytes));
        assert_eq!(result, Ok(0));
    }

    #[test]
    fn test_decode_s64_errors() {
        // Maximum of 10 bytes encoding, the 0x80 bit must not be set in the final byte.
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];
        let result = decode_s64(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // Maximum of 10 bytes encoding, the 0x80 bit must not be set.
        // Ensure error is an invalid num instead of an EOF error.
        let bytes = &[
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00,
        ];
        let result = decode_s64(&mut SliceRead::new(bytes));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // If the highest valid bit is set, it should be sign extended. (final byte should be 0x78)
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x08];
        let result = decode_s64(&mut SliceRead::new(bytes.as_slice()));
        assert_eq!(result, Err(DecodeError::InvalidNum));

        // If the highest valid bit is set, it should be sign extended. (final byte should be 0x78)
        let bytes = &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x28];
        let result = decode_s64(&mut SliceRead::new(bytes.as_slice()));
        assert_eq!(result, Err(DecodeError::InvalidNum));
    }

    /// The random byte array could be a valid but inefficient encoding of a
    /// value. To make assertions easier, reduce the encoding to the most
    /// efficient representation.
    fn reduce_uint_encoding<const SIZE: usize>(
        mut bytes_read: usize,
        mut bytes: [u8; SIZE],
    ) -> (usize, [u8; SIZE]) {
        while bytes_read > 1 {
            if bytes[bytes_read - 1] == 0 {
                bytes[bytes_read - 2] &= 0x7f;
                bytes_read -= 1;
            } else {
                break;
            }
        }
        (bytes_read, bytes)
    }

    /// The random byte array could be a valid but inefficient encoding of a
    /// value. To make assertions easier, reduce the encoding to the most
    /// efficient representation.
    fn reduce_sint_encoding<const SIZE: usize>(
        mut bytes_read: usize,
        mut bytes: [u8; SIZE],
    ) -> (usize, [u8; SIZE]) {
        while bytes_read > 1 {
            if bytes[bytes_read - 1] == 0 {
                if bytes[bytes_read - 2] & 0x40 == 0 {
                    bytes[bytes_read - 2] &= 0x7f;
                    bytes_read -= 1;
                } else {
                    break;
                }
            } else if bytes[bytes_read - 1] == 0x7f {
                if bytes[bytes_read - 2] & 0x40 == 0 {
                    break;
                }
                bytes[bytes_read - 2] &= 0x7f;
                bytes_read -= 1;
            } else {
                break;
            }
        }

        (bytes_read, bytes)
    }

    #[cfg(feature = "std")]
    proptest! {
        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_encode_decode_u32_num(n in any::<u32>()) {
            let bytes = encode_u32(n);
            let result = decode_u32(&mut SliceRead::new(&bytes));
            prop_assert_eq!(Ok(n), result);
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_encode_decode_u64_num(n in any::<u64>()) {
            let bytes = encode_u64(n);
            let result = decode_u64(&mut SliceRead::new(&bytes));
            prop_assert_eq!(Ok(n), result);
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_encode_decode_s32_num(n in any::<i32>()) {
            let bytes = encode_s32(n);
            let result = decode_s32(&mut SliceRead::new(&bytes));
            prop_assert_eq!(Ok(n), result);
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_encode_decode_s33_num(n in i64::from(i32::MIN) * 2..i64::from(i32::MAX) * 2) {
            let bytes = encode_s64(n);
            let result = decode_s33_block_ty(&mut SliceRead::new(&bytes));
            prop_assert_eq!(Ok(n), result);
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_encode_decode_s64_num(n in any::<i64>()) {
            let bytes = encode_s64(n);
            let result = decode_s64(&mut SliceRead::new(&bytes));
            prop_assert_eq!(Ok(n), result);
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_decode_encode_u32_bytes(bytes in prop::array::uniform7(any::<u8>())) {
            let reader = &mut SliceRead::new(bytes.as_slice());
            let Ok(value) = decode_u32(reader) else {
                for b in bytes.iter().take(4) {
                    prop_assert!(b & 0x80 != 0);
                }
                prop_assert!(bytes[4] > 0x0f);
                return Ok(());
            };
            let encoded_bytes = encode_u32(value);

            let (bytes_read, reduced_bytes) = reduce_uint_encoding(
                usize::try_from(reader.pos()).unwrap(),
                bytes,
            );

            prop_assert_eq!(
                &reduced_bytes[..bytes_read],
                &encoded_bytes
            );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_decode_encode_u64_bytes(bytes in prop::array::uniform12(any::<u8>())) {
            let reader = &mut SliceRead::new(bytes.as_slice());
            let Ok(value) = decode_u64(reader) else {
                for b in bytes.iter().take(9) {
                    prop_assert!(b & 0x80 != 0);
                }
                prop_assert!(bytes[9] > 0x01);
                return Ok(());
            };
            let encoded_bytes = encode_u64(value);

            let (bytes_read, reduced_bytes) = reduce_uint_encoding(
                usize::try_from(reader.pos()).unwrap(),
                bytes,
            );

            prop_assert_eq!(
                &reduced_bytes[..bytes_read],
                &encoded_bytes
            );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_decode_encode_s32_bytes(bytes in prop::array::uniform7(any::<u8>())) {
            let read_slice = bytes;
            let reader = &mut SliceRead::new(read_slice.as_slice());
            let Ok(value) = decode_s32(reader) else {
                prop_assume!(false);
                return Ok(());
            };
            let encoded_bytes = encode_s32(value);

            let (bytes_read, reduced_bytes) = reduce_sint_encoding(
                usize::try_from(reader.pos()).unwrap(),
                bytes
            );

            prop_assert_eq!(
                &reduced_bytes[..bytes_read],
                &encoded_bytes
             );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_decode_encode_s33_bytes(bytes in prop::array::uniform7(any::<u8>())) {
            let read_slice = bytes;
            let reader = &mut SliceRead::new(read_slice.as_slice());
            let Ok(value) = decode_s33_block_ty(reader) else {
                prop_assume!(false);
                return Ok(());
            };
            let encoded_bytes = encode_s64(value);

            let (bytes_read, reduced_bytes) = reduce_sint_encoding(
                usize::try_from(reader.pos()).unwrap(),
                bytes
            );

            prop_assert_eq!(
                &reduced_bytes[..bytes_read],
                &encoded_bytes
             );
        }

        #[allow(clippy::ignored_unit_patterns)]
        #[test]
        fn test_decode_encode_s64_bytes(bytes in prop::array::uniform7(any::<u8>())) {
            let read_slice = bytes;
            let reader = &mut SliceRead::new(read_slice.as_slice());
            let Ok(value) = decode_s64(reader) else {
                prop_assume!(false);
                return Ok(());
            };
            let encoded_bytes = encode_s64(value);

            let (bytes_read, reduced_bytes) = reduce_sint_encoding(
                usize::try_from(reader.pos()).unwrap(),
                bytes
            );

            prop_assert_eq!(
                &reduced_bytes[..bytes_read],
                &encoded_bytes
             );
        }
    }
}
