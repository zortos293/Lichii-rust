//! Android parcels as the HOS binder carries them: a 16-byte header, then 4-byte aligned data and
//! an object table.

use bytemuck::Pod;

/// Reads a transaction's input parcel.
pub(super) struct ParcelReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> ParcelReader<'a> {
    /// Starts at the data the header points to, past the interface token.
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self { data, position: 0 };
        reader.position = reader.word_at(4) as usize;
        let _strict_mode: u32 = reader.read();
        let length = reader.read::<u32>() as usize;
        reader.position = (reader.position + (length + 1) * 2).next_multiple_of(4);
        reader
    }

    fn word_at(&self, offset: usize) -> u32 {
        self.data
            .get(offset..offset + 4)
            .map_or(0, bytemuck::pod_read_unaligned)
    }

    /// Reads the next value; bytes past the end read as zero.
    pub fn read<T: Pod>(&mut self) -> T {
        let size = std::mem::size_of::<T>();
        let mut value = T::zeroed();
        let bytes = bytemuck::bytes_of_mut(&mut value);
        let available = self.data.len().saturating_sub(self.position).min(size);
        if available > 0 {
            bytes[..available]
                .copy_from_slice(&self.data[self.position..self.position + available]);
        }
        self.position += size.next_multiple_of(4);
        value
    }

    pub fn read_bool(&mut self) -> bool {
        self.read::<u32>() & 0xff != 0
    }

    /// A flattened value: its size as an i64, then the value.
    pub fn read_flattened<T: Pod>(&mut self) -> T {
        let _size: i64 = self.read();
        self.read()
    }

    /// A nullable flattened object.
    pub fn read_object<T: Pod>(&mut self) -> Option<T> {
        self.read_bool().then(|| self.read_flattened())
    }
}

/// Builds a reply parcel.
#[derive(Default)]
pub(super) struct ParcelWriter {
    data: Vec<u8>,
    objects: Vec<u8>,
}

impl ParcelWriter {
    pub fn write<T: Pod>(&mut self, value: T) {
        self.data.extend_from_slice(bytemuck::bytes_of(&value));
        self.data.resize(self.data.len().next_multiple_of(4), 0);
    }

    /// A nullable flattened object.
    pub fn write_object<T: Pod>(&mut self, value: Option<&T>) {
        let Some(value) = value else {
            self.write(0u32);
            return;
        };
        self.write(1u32);
        self.write(std::mem::size_of::<T>() as i64);
        self.write(*value);
    }

    /// A binder object: its data, and an entry in the object table.
    pub fn write_interface<T: Pod>(&mut self, value: T) {
        self.write(value);
        self.objects.extend_from_slice(&0u32.to_le_bytes());
    }

    pub fn serialize(&self) -> Vec<u8> {
        const HEADER: u32 = 16;
        let data_size = self.data.len() as u32;
        let mut out = Vec::with_capacity(16 + self.data.len() + self.objects.len());
        for word in [
            data_size,
            HEADER,
            self.objects.len() as u32,
            HEADER + data_size,
        ] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&self.data);
        out.extend_from_slice(&self.objects);
        out
    }
}
