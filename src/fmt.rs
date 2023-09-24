//! WebAssembly binary and text formats.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::fmt;
#[cfg(feature = "std")]
use std::{error, fmt, io};

pub mod binary;
pub mod text;

/// Wrapper for read error.
///
/// The primary purpose is to determine if the error is because an end of file
/// condition has been encountered.
#[derive(Debug, PartialEq, Eq)]
struct ReadError<E> {
    inner: E,
    is_eof: bool,
}

impl<E> ReadError<E> {
    /// Wraps an error and a flag which determines if the error is an end of file condition
    #[inline]
    #[must_use]
    fn new(inner: E, is_eof: bool) -> Self {
        Self { inner, is_eof }
    }

    /// If the error is an end of file condition.
    ///
    /// In some cases, EOF is unexpected, but it can also be used to determine there is no additional input.
    #[inline]
    #[must_use]
    fn is_eof(&self) -> bool {
        self.is_eof
    }
}

impl<E> fmt::Display for ReadError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

#[cfg(feature = "std")]
impl<E> error::Error for ReadError<E>
where
    E: error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        Some(&self.inner)
    }
}

/// Trait used to read bytes.
trait Read {
    /// Error type for methods
    type Error;

    /// Consumes and returns the next read byte.
    ///
    /// # Errors
    ///
    /// Returns a [`ReadError`] which wraps an inner error and determines if an EOF condition was reached.
    fn next(&mut self) -> Result<u8, ReadError<Self::Error>>;

    /// Returns the next byte but does not consume.
    ///
    /// Repeated peeks (with no [next()][Read::next] call) should return the same byte.
    ///
    /// # Errors
    ///
    /// Returns a [`ReadError`] which wraps an inner error and determines if an EOF condition was reached.
    fn peek(&mut self) -> Result<u8, ReadError<Self::Error>>;

    /// Returns the position in the stream of bytes.
    fn pos(&self) -> u64;

    /// # Errors
    ///
    /// Returns a [`ReadError`] which wraps an inner error and determines if an EOF condition was reached.
    fn skip(&mut self, mut count: u64) -> Result<(), ReadError<Self::Error>> {
        while count > 0 {
            self.next()?;
            count -= 1;
        }

        Ok(())
    }
}

/// A wrapper to implement this crate's [`Read`] trait for [`std::io::Read`] trait implementations.
#[cfg(feature = "std")]
#[derive(Debug)]
struct IoRead<R>
where
    R: io::Read,
{
    iter: io::Bytes<R>,
    peeked_byte: Option<u8>,
    byte_offset: u64,
}

#[cfg(feature = "std")]
impl<R> IoRead<R>
where
    R: io::Read,
{
    /// Instantiates a new reader.
    fn new(reader: R) -> Self {
        IoRead {
            iter: reader.bytes(),
            peeked_byte: None,
            byte_offset: 0,
        }
    }
}

#[cfg(feature = "std")]
impl<R> Read for IoRead<R>
where
    R: io::Read,
{
    type Error = io::Error;

    #[inline]
    fn next(&mut self) -> Result<u8, ReadError<Self::Error>> {
        match self.peeked_byte.take() {
            Some(b) => {
                self.byte_offset += 1;
                Ok(b)
            }
            None => match self.iter.next() {
                Some(Ok(b)) => {
                    self.byte_offset += 1;
                    Ok(b)
                }
                Some(Err(err)) => {
                    let is_eof = err.kind() == io::ErrorKind::UnexpectedEof;
                    Err(ReadError::new(err, is_eof))
                }
                None => Err(ReadError::new(
                    io::Error::new(io::ErrorKind::UnexpectedEof, "could not get next byte"),
                    true,
                )),
            },
        }
    }

    #[inline]
    fn peek(&mut self) -> Result<u8, ReadError<Self::Error>> {
        match self.peeked_byte {
            Some(b) => Ok(b),
            None => match self.iter.next() {
                Some(Ok(b)) => {
                    self.peeked_byte = Some(b);
                    Ok(b)
                }
                Some(Err(err)) => {
                    let is_eof = err.kind() == io::ErrorKind::UnexpectedEof;
                    Err(ReadError::new(err, is_eof))
                }
                None => Err(ReadError::new(
                    io::Error::new(io::ErrorKind::UnexpectedEof, "could not get next byte"),
                    true,
                )),
            },
        }
    }

    #[inline]
    fn pos(&self) -> u64 {
        self.byte_offset
    }
}

/// The read from a slice is out of bounds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfBoundsError;

impl fmt::Display for OutOfBoundsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("out of bounds")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OutOfBoundsError {}

/// A wrapper to implement this crate's [`Read`] trait for byte slices.
#[derive(Debug)]
struct SliceRead<'a> {
    slice: &'a [u8],
    byte_offset: usize,
}

impl<'a> SliceRead<'a> {
    /// Instantiates a new reader.
    #[must_use]
    fn new(slice: &'a [u8]) -> Self {
        SliceRead {
            slice,
            byte_offset: 0,
        }
    }
}

impl<'a> Read for SliceRead<'a> {
    type Error = OutOfBoundsError;

    #[inline]
    fn next(&mut self) -> Result<u8, ReadError<Self::Error>> {
        if self.byte_offset < self.slice.len() {
            let b = self.slice[self.byte_offset];
            self.byte_offset += 1;
            Ok(b)
        } else {
            Err(ReadError::new(OutOfBoundsError, true))
        }
    }

    #[inline]
    fn peek(&mut self) -> Result<u8, ReadError<Self::Error>> {
        if self.byte_offset < self.slice.len() {
            Ok(self.slice[self.byte_offset])
        } else {
            Err(ReadError::new(OutOfBoundsError, true))
        }
    }

    #[inline]
    fn pos(&self) -> u64 {
        u64::try_from(self.byte_offset).unwrap()
    }
}
