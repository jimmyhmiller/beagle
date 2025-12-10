use std::ffi::c_void;
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo};
use std::sync::OnceLock;

/// Cached page size to avoid repeated sysconf calls
static PAGE_SIZE: OnceLock<usize> = OnceLock::new();

/// Cross-platform function to get the system page size (cached)
#[cfg(target_os = "macos")]
pub fn get_page_size() -> usize {
    *PAGE_SIZE.get_or_init(|| unsafe { libc::vm_page_size })
}

#[cfg(target_os = "linux")]
pub fn get_page_size() -> usize {
    *PAGE_SIZE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn get_page_size() -> usize {
    panic!("Unsupported platform: get_page_size() only supports macOS and Linux")
}

/// A memory-mapped region that can have its protection changed.
/// This is a simple wrapper around libc mmap/mprotect/munmap.
pub struct MappedRegion {
    ptr: *mut u8,
    size: usize,
    is_writable: bool,
}

impl MappedRegion {
    /// Reserve address space without committing memory (PROT_NONE).
    /// The memory cannot be accessed until make_rw() is called.
    pub fn reserve(size: usize) -> Result<Self, std::io::Error> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            size,
            is_writable: false,
        })
    }

    /// Allocate memory that is immediately readable and writable.
    pub fn alloc_rw(size: usize) -> Result<Self, std::io::Error> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            size,
            is_writable: true,
        })
    }

    /// Allocate memory for JIT code.
    /// Starts with PROT_NONE then transitions to RW, matching the mmap-rs behavior.
    pub fn alloc_jit(size: usize) -> Result<Self, std::io::Error> {
        let mut region = Self::reserve(size)?;
        region.make_rw()?;
        Ok(region)
    }

    /// Allocate memory that is initially read-only.
    pub fn alloc_ro(size: usize) -> Result<Self, std::io::Error> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            size,
            is_writable: false,
        })
    }

    /// Make the region readable and writable.
    pub fn make_rw(&mut self) -> Result<(), std::io::Error> {
        self.mprotect(libc::PROT_READ | libc::PROT_WRITE)?;
        self.is_writable = true;
        Ok(())
    }

    /// Make the region readable and executable (for JIT code).
    pub fn make_rx(&mut self) -> Result<(), std::io::Error> {
        self.mprotect(libc::PROT_READ | libc::PROT_EXEC)?;
        self.is_writable = false;
        Ok(())
    }

    /// Make the region read-only.
    pub fn make_ro(&mut self) -> Result<(), std::io::Error> {
        self.mprotect(libc::PROT_READ)?;
        self.is_writable = false;
        Ok(())
    }

    /// Make a portion of the region inaccessible (for guard pages).
    pub fn protect_range(&self, offset: usize, len: usize) -> Result<(), std::io::Error> {
        let result =
            unsafe { libc::mprotect(self.ptr.add(offset) as *mut c_void, len, libc::PROT_NONE) };
        if result != 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn mprotect(&self, prot: i32) -> Result<(), std::io::Error> {
        let result = unsafe { libc::mprotect(self.ptr as *mut c_void, self.size, prot) };
        if result != 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a mutable slice to the entire region.
    /// Safety: caller must ensure the region is writable.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Get a slice to the entire region.
    /// Safety: caller must ensure the region is readable.
    pub unsafe fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }
}

impl Drop for MappedRegion {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut c_void, self.size);
        }
    }
}

unsafe impl Send for MappedRegion {}
unsafe impl Sync for MappedRegion {}

// Index implementations for convenient access
impl Index<usize> for MappedRegion {
    type Output = u8;
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.size, "Index out of bounds");
        unsafe { &*self.ptr.add(index) }
    }
}

impl IndexMut<usize> for MappedRegion {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.size, "Index out of bounds");
        unsafe { &mut *self.ptr.add(index) }
    }
}

impl Index<RangeFull> for MappedRegion {
    type Output = [u8];
    fn index(&self, _: RangeFull) -> &Self::Output {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }
}

impl IndexMut<RangeFull> for MappedRegion {
    fn index_mut(&mut self, _: RangeFull) -> &mut Self::Output {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Index<Range<usize>> for MappedRegion {
    type Output = [u8];
    fn index(&self, range: Range<usize>) -> &Self::Output {
        assert!(range.end <= self.size, "Range end out of bounds");
        unsafe { std::slice::from_raw_parts(self.ptr.add(range.start), range.end - range.start) }
    }
}

impl IndexMut<Range<usize>> for MappedRegion {
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        assert!(range.end <= self.size, "Range end out of bounds");
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.end - range.start)
        }
    }
}

impl Index<RangeFrom<usize>> for MappedRegion {
    type Output = [u8];
    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        assert!(range.start <= self.size, "Range start out of bounds");
        unsafe { std::slice::from_raw_parts(self.ptr.add(range.start), self.size - range.start) }
    }
}

impl IndexMut<RangeFrom<usize>> for MappedRegion {
    fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut Self::Output {
        assert!(range.start <= self.size, "Range start out of bounds");
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.add(range.start), self.size - range.start)
        }
    }
}

impl Index<RangeTo<usize>> for MappedRegion {
    type Output = [u8];
    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        assert!(range.end <= self.size, "Range end out of bounds");
        unsafe { std::slice::from_raw_parts(self.ptr, range.end) }
    }
}

impl IndexMut<RangeTo<usize>> for MappedRegion {
    fn index_mut(&mut self, range: RangeTo<usize>) -> &mut Self::Output {
        assert!(range.end <= self.size, "Range end out of bounds");
        unsafe { std::slice::from_raw_parts_mut(self.ptr, range.end) }
    }
}
