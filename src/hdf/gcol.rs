//! Global Heap Collection (GCOL) parser.
//!
//! HDF5 Level 1E - Global Heap for variable-length data.
//! Used to store variable-length strings and object references.

use std::collections::HashMap;

use winnow::binary::{le_u8, le_u16};
use winnow::error::StrContext;
use winnow::stream::Location;
use winnow::token::{literal, take};
use winnow::{ModalResult, Parser};

use super::helpers::varint_size;
use super::parser::Input;

/// ASCII signature: "GCOL"
const GCOL_SIGNATURE: [u8; 4] = [0x47, 0x43, 0x4F, 0x4C];

/// A single object in the global heap.
#[derive(Debug, Clone)]
pub struct GcolObject {
    /// Heap object index (reference ID)
    pub heap_object_index: u16,
    /// Size of the object data
    pub object_size: u64,
    /// The actual value (for small objects <= 8 bytes)
    pub value: u64,
}

/// Global Heap Collection - stores variable-length data.
#[derive(Debug, Clone, Default)]
pub struct GlobalHeap {
    /// Objects indexed by (collection_address, heap_object_index)
    objects: HashMap<(u64, u16), GcolObject>,
}

impl GlobalHeap {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    /// Look up an object by collection address and reference index.
    pub fn get(&self, address: u64, reference: u16) -> Option<&GcolObject> {
        self.objects.get(&(address, reference))
    }

    /// Insert objects from a parsed collection.
    pub fn insert(&mut self, address: u64, objects: Vec<GcolObject>) {
        for obj in objects {
            self.objects.insert((address, obj.heap_object_index), obj);
        }
    }

    /// Check if a collection at the given address has been parsed.
    pub fn has_collection(&self, address: u64) -> bool {
        self.objects.keys().any(|(addr, _)| *addr == address)
    }
}

/// Parse a Global Heap Collection at the current position.
///
/// Returns a list of objects in the collection.
pub(crate) fn gcol_read(input: &mut Input) -> ModalResult<Vec<GcolObject>> {
    let size_of_lengths = input.state.size_of_lengths();

    // Read signature
    literal(GCOL_SIGNATURE)
        .context(StrContext::Label("GCOL signature"))
        .parse_next(input)?;

    // Version must be 1
    le_u8
        .verify(|v| *v == 1)
        .context(StrContext::Label("GCOL version"))
        .context(StrContext::Expected("1".into()))
        .parse_next(input)?;

    // Skip 3 reserved bytes
    take(3usize).parse_next(input)?;

    // Collection size
    let collection_size = varint_size(size_of_lengths)
        .verify(|s| *s <= 0x4_0000_0000) // 16GB limit
        .context(StrContext::Label("GCOL collection size"))
        .parse_next(input)?;

    // Calculate end position (collection_size includes the 8-byte header we already read)
    let start_pos = input.current_token_start();
    let end_pos = start_pos + collection_size as usize - 8;

    let mut objects = Vec::new();

    // Read objects until we reach the end or encounter index 0
    while input.current_token_start() + 8 + size_of_lengths as usize <= end_pos {
        let heap_object_index = le_u16.parse_next(input)?;

        // Index 0 marks end of objects
        if heap_object_index == 0 {
            break;
        }

        // Reference count (unused)
        let _reference_count = le_u16.parse_next(input)?;

        // Skip 4 reserved bytes
        take(4usize).parse_next(input)?;

        // Object size
        let object_size = varint_size(size_of_lengths).parse_next(input)?;

        // For now, only support small objects (value fits in u64)
        if object_size > 8 {
            log::warn!(
                "GCOL object {} has size {} > 8, skipping value read",
                heap_object_index,
                object_size
            );
            // Skip the object data
            take(object_size as usize).parse_next(input)?;
            continue;
        }

        // Read the value
        let value = if object_size > 0 {
            varint_size(object_size as u8).parse_next(input)?
        } else {
            0
        };

        log::info!(
            "GCOL object {} size {} value {:#x}",
            heap_object_index,
            object_size,
            value
        );

        objects.push(GcolObject {
            heap_object_index,
            object_size,
            value,
        });
    }

    Ok(objects)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_heap_lookup() {
        let mut heap = GlobalHeap::new();

        let objects = vec![
            GcolObject {
                heap_object_index: 1,
                object_size: 4,
                value: 0x12345678,
            },
            GcolObject {
                heap_object_index: 2,
                object_size: 8,
                value: 0xDEADBEEF,
            },
        ];

        heap.insert(0x1000, objects);

        assert!(heap.has_collection(0x1000));
        assert!(!heap.has_collection(0x2000));

        let obj1 = heap.get(0x1000, 1).unwrap();
        assert_eq!(obj1.value, 0x12345678);

        let obj2 = heap.get(0x1000, 2).unwrap();
        assert_eq!(obj2.value, 0xDEADBEEF);

        assert!(heap.get(0x1000, 3).is_none());
        assert!(heap.get(0x2000, 1).is_none());
    }
}
