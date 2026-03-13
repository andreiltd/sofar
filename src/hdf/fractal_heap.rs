use winnow::binary::{le_u8, le_u16, le_u32, le_u64};
use winnow::error::{ErrMode, ParserError, StrContext};
use winnow::stream::Stream;
use winnow::token::{literal, take};
use winnow::{ModalResult, Parser};

use super::helpers::varint_size;
use super::parser::Input;

/// Fractal Heap Header signature
pub const FRHP_SIGNATURE: [u8; 4] = [0x46, 0x52, 0x48, 0x50];
/// Fractal Heap Direct Block signature
pub const FHDB_SIGNATURE: [u8; 4] = [0x46, 0x48, 0x44, 0x42];
/// Fractal Heap Indirect Block signature
pub const FHIB_SIGNATURE: [u8; 4] = [0x46, 0x48, 0x49, 0x42];

const MAX_NAME_LENGTH: usize = 0x100;
const MAX_RECURSIVE_DEPTH: u32 = 20;

#[derive(Clone, Debug, Default)]
pub struct FractalHeap {
    pub flags: u8,
    pub heap_id_length: u16,
    pub encoded_length: u16,
    pub table_width: u16,
    pub maximum_heap_size: u16,
    pub starting_row: u16,
    pub current_row: u16,
    pub maximum_size: u32,
    pub filter_mask: u32,
    pub next_huge_object_id: u64,
    pub btree_address_of_huge_objects: u64,
    pub free_space: u64,
    pub address_free_space: u64,
    pub amount_managed_space: u64,
    pub amount_allocated_space: u64,
    pub offset_managed_space: u64,
    pub number_managed_objects: u64,
    pub size_huge_objects: u64,
    pub number_huge_objects: u64,
    pub size_tiny_objects: u64,
    pub number_tiny_objects: u64,
    pub starting_block_size: u64,
    pub maximum_direct_block_size: u64,
    pub address_of_root_block: u64,
    pub size_of_filtered_block: u64,
    pub filter_information: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Attribute {
    pub name: String,
    pub value: Option<String>,
}

#[derive(Clone, Debug)]
pub struct DirectoryEntry {
    pub name: String,
    pub address: u64,
}

#[derive(Clone, Debug)]
pub struct FractalHeapData {
    pub attributes: Vec<Attribute>,
    pub directories: Vec<DirectoryEntry>,
}

pub(crate) fn fractal_heap_read(input: &mut Input) -> ModalResult<(FractalHeap, FractalHeapData)> {
    let size_of_offsets = input.state.size_of_offsets();
    let size_of_lengths = input.state.size_of_lengths();

    let _signature = literal(FRHP_SIGNATURE).parse_next(input)?;

    let _version = le_u8
        .verify(|v| *v == 0)
        .context(StrContext::Label("Fractal heap version"))
        .context(StrContext::Expected("0".into()))
        .parse_next(input)?;

    let heap_id_length = le_u16.parse_next(input)?;
    let encoded_length = le_u16
        .verify(|l| *l <= 0x8000)
        .context(StrContext::Label("Fractal heap encoded length"))
        .context(StrContext::Expected("<= 0x8000".into()))
        .parse_next(input)?;

    let flags = le_u8.parse_next(input)?;
    let maximum_size = le_u32.parse_next(input)?;

    let next_huge_object_id = varint_size(size_of_lengths).parse_next(input)?;
    let btree_address_of_huge_objects = varint_size(size_of_offsets).parse_next(input)?;
    let free_space = varint_size(size_of_lengths).parse_next(input)?;
    let address_free_space = varint_size(size_of_offsets).parse_next(input)?;
    let amount_managed_space = varint_size(size_of_lengths).parse_next(input)?;
    let amount_allocated_space = varint_size(size_of_lengths).parse_next(input)?;
    let offset_managed_space = varint_size(size_of_lengths).parse_next(input)?;
    let number_managed_objects = varint_size(size_of_lengths).parse_next(input)?;
    let size_huge_objects = varint_size(size_of_lengths).parse_next(input)?;
    let number_huge_objects = varint_size(size_of_lengths).parse_next(input)?;
    let size_tiny_objects = varint_size(size_of_lengths).parse_next(input)?;
    let number_tiny_objects = varint_size(size_of_lengths).parse_next(input)?;

    let table_width = le_u16.parse_next(input)?;
    let starting_block_size = varint_size(size_of_lengths).parse_next(input)?;
    let maximum_direct_block_size = varint_size(size_of_lengths).parse_next(input)?;
    let maximum_heap_size = le_u16.parse_next(input)?;
    let starting_row = le_u16.parse_next(input)?;
    let address_of_root_block = varint_size(size_of_offsets).parse_next(input)?;
    let current_row = le_u16.parse_next(input)?;

    let (size_of_filtered_block, filter_mask, filter_information) = if encoded_length > 0 {
        let size_of_filtered_block = varint_size(size_of_lengths).parse_next(input)?;
        let filter_mask = le_u32.parse_next(input)?;
        let filter_information = take(encoded_length).parse_next(input)?.to_vec();
        (size_of_filtered_block, filter_mask, filter_information)
    } else {
        (0, 0, Vec::new())
    };

    // Skip checksum
    let _checksum = take(4usize).parse_next(input)?;

    // Validate constraints from C code
    if number_huge_objects > 0 {
        return Err(ErrMode::assert(input, "Cannot handle huge objects"));
    }

    if number_tiny_objects > 0 {
        return Err(ErrMode::assert(input, "Cannot handle tiny objects"));
    }

    let fractal_heap = FractalHeap {
        flags,
        heap_id_length,
        encoded_length,
        table_width,
        maximum_heap_size,
        starting_row,
        current_row,
        maximum_size,
        filter_mask,
        next_huge_object_id,
        btree_address_of_huge_objects,
        free_space,
        address_free_space,
        amount_managed_space,
        amount_allocated_space,
        offset_managed_space,
        number_managed_objects,
        size_huge_objects,
        number_huge_objects,
        size_tiny_objects,
        number_tiny_objects,
        starting_block_size,
        maximum_direct_block_size,
        address_of_root_block,
        size_of_filtered_block,
        filter_information,
    };

    let mut heap_data = FractalHeapData {
        attributes: Vec::new(),
        directories: Vec::new(),
    };

    // Process root block if valid address
    if input.state.is_address_valid(address_of_root_block) {
        let cp = input.checkpoint();

        // Seek to root block
        input.input.reset_to_start();
        take(address_of_root_block as usize).parse_next(input)?;

        if current_row > 0 {
            // Indirect block
            let block_data = indirect_block_read(input, &fractal_heap, starting_block_size)?;
            heap_data.attributes.extend(block_data.attributes);
            heap_data.directories.extend(block_data.directories);
        } else {
            // Direct block
            let block_data = direct_block_read(input, &fractal_heap)?;
            heap_data.attributes.extend(block_data.attributes);
            heap_data.directories.extend(block_data.directories);
        }

        input.reset(&cp);
    }

    Ok((fractal_heap, heap_data))
}

fn direct_block_read(
    input: &mut Input,
    fractal_heap: &FractalHeap,
) -> ModalResult<FractalHeapData> {
    let size_of_offsets = input.state.size_of_offsets();

    if input.state.recursive_counter() >= MAX_RECURSIVE_DEPTH {
        return Err(ErrMode::assert(input, "Recursive problem in fractal heap"));
    }

    input.state.recursive_counter_inc();

    let _signature = literal(FHDB_SIGNATURE).parse_next(input)?;

    let _version = le_u8
        .verify(|v| *v == 0)
        .context(StrContext::Label("FHDB version"))
        .context(StrContext::Expected("0".into()))
        .parse_next(input)?;

    // Skip heap header address
    let _heap_header_address = varint_size(size_of_offsets).parse_next(input)?;

    let size = fractal_heap.maximum_heap_size.div_ceil(8);
    let _block_offset = varint_size(size as u8).parse_next(input)?;

    if fractal_heap.flags & 2 != 0 {
        let _skip = take(4usize).parse_next(input)?;
    }

    let offset_size = ((fractal_heap.maximum_heap_size as f32).log2() / 8.0).ceil() as u8;
    let length_size = if fractal_heap.maximum_direct_block_size < fractal_heap.maximum_size as u64 {
        ((fractal_heap.maximum_direct_block_size as f32).log2() / 8.0).ceil() as u8
    } else {
        ((fractal_heap.maximum_size as f32).log2() / 8.0).ceil() as u8
    };

    let mut block_data = FractalHeapData {
        attributes: Vec::new(),
        directories: Vec::new(),
    };

    loop {
        let type_and_version = le_u8.parse_next(input)?;
        if type_and_version == 0 {
            break;
        }

        let _offset = varint_size(offset_size).parse_next(input)?;
        let length = varint_size(length_size).parse_next(input)?;

        if length > 0x10000000 {
            return Err(ErrMode::assert(input, "FHDB length too large"));
        }

        match type_and_version {
            3 => {
                // Name-value pair attribute
                let attr = parse_type_3_attribute(input, length as usize)?;
                block_data.attributes.push(attr);
            }
            1 => {
                // Directory entry or complex attribute
                let entry_data = parse_type_1_entry(input, length as usize)?;
                block_data.attributes.extend(entry_data.attributes);
                block_data.directories.extend(entry_data.directories);
            }
            _ => {
                log::warn!("Unknown fractal heap type: {type_and_version}");
                // Skip unknown types gracefully to continue parsing
                let _skip = take(length as usize).parse_next(input)?;
            }
        }
    }

    input.state.recursive_counter_dec();
    Ok(block_data)
}

fn parse_type_3_attribute(input: &mut Input, length: usize) -> ModalResult<Attribute> {
    // Parse the magic number first
    let _magic = varint_size(5usize)
        .verify(|v| *v == 0x0000040008)
        .context(StrContext::Label("FHDB type 3 magic"))
        .context(StrContext::Expected("0x0000040008".into()))
        .parse_next(input)?;

    // Read the name with the specified length (may contain non-UTF8 bytes)
    let name_bytes: &[u8] = take(length).parse_next(input)?;
    let name = String::from_utf8_lossy(name_bytes)
        .trim_matches(|c: char| c.is_whitespace() || c == '\0')
        .to_string();

    // Read the second magic number
    let _magic2 = le_u32
        .verify(|v| *v == 0x00000013)
        .context(StrContext::Label("FHDB type 3 magic2"))
        .context(StrContext::Expected("0x00000013".into()))
        .parse_next(input)?;

    let value_len = le_u16
        .verify(|l| *l <= 0x1000)
        .context(StrContext::Label("FHDB type 3 value length"))
        .context(StrContext::Expected("<= 0x1000".into()))
        .parse_next(input)? as usize;

    let unknown1 = varint_size(6usize).parse_next(input)?;

    let value = match unknown1 {
        0x000000020200 => None,
        0x000000020000 => {
            let val_bytes: &[u8] = take(value_len).parse_next(input)?;
            Some(
                String::from_utf8_lossy(val_bytes)
                    .trim_matches(|c: char| c.is_whitespace() || c == '\0')
                    .to_string(),
            )
        }
        0x20000020000 => Some(String::new()),
        _ => {
            log::warn!("Unsupported FHDB type 3 value format: {unknown1:#x}");
            return Err(ErrMode::assert(input, "Unsupported FHDB type 3 format"));
        }
    };

    Ok(Attribute { name, value })
}

fn parse_type_1_entry(input: &mut Input, _length: usize) -> ModalResult<FractalHeapData> {
    let size_of_offsets = input.state.size_of_offsets();
    let mut entry_data = FractalHeapData {
        attributes: Vec::new(),
        directories: Vec::new(),
    };

    let unknown2 = le_u32.parse_next(input)?;

    match unknown2 {
        0 => {
            // Directory entry case
            let _unknown3 = le_u16
                .verify(|v| *v == 0x0000)
                .context(StrContext::Label("FHDB type 1 unknown3"))
                .context(StrContext::Expected("0x0000".into()))
                .parse_next(input)?;

            let name_len = le_u8
                .verify(|l| (*l as usize) < MAX_NAME_LENGTH)
                .context(StrContext::Label("FHDB type 1 name length"))
                .context(StrContext::Expected("reasonable name length".into()))
                .parse_next(input)? as usize;

            let name = take(name_len).parse_to::<String>().parse_next(input)?;
            let heap_header_address = varint_size(size_of_offsets).parse_next(input)?;

            log::info!("Directory entry: {name} at address {heap_header_address:#x}");

            entry_data.directories.push(DirectoryEntry {
                name,
                address: heap_header_address,
            });
        }
        0x00080008 | 0x00040008 => {
            // Complex attribute cases
            let attr = parse_complex_attribute(input)?;
            entry_data.attributes.push(attr);
        }
        _ => {
            log::warn!("FHDB type 1 unsupported values {unknown2:#08x}");
            return Err(ErrMode::assert(input, "Unsupported FHDB type 1 format"));
        }
    }

    Ok(entry_data)
}

fn parse_complex_attribute(input: &mut Input) -> ModalResult<Attribute> {
    // Both 0x00080008 and 0x00040008 use the same name parsing logic
    // Use stack-allocated buffer to avoid heap allocation for each attribute
    let mut name_bytes = [0u8; MAX_NAME_LENGTH];
    let mut len: Option<usize> = None;

    for (i, name_byte) in name_bytes.iter_mut().enumerate().take(MAX_NAME_LENGTH) {
        let c = le_u8.parse_next(input)?;
        *name_byte = c;

        if len.is_none() && c == 0 {
            len = Some(i);
        }
        if c == 0x13 {
            if len.is_none() {
                // No null terminator found before sentinel; use position up to sentinel
                len = Some(i);
            }
            break;
        }
    }

    let name_end = len.unwrap_or(0);

    // Convert to string up to the null terminator
    let name = String::from_utf8_lossy(&name_bytes[..name_end]).to_string();

    // Read exactly 3 bytes for the reserved field (must be 0x000000 per C spec)
    let _reserved = varint_size(3usize)
        .verify(|v| *v == 0)
        .context(StrContext::Label("Complex attribute reserved bytes"))
        .context(StrContext::Expected("0x000000".into()))
        .parse_next(input)?;

    let value_len = le_u32
        .verify(|l| *l <= 0x1000)
        .context(StrContext::Label("Complex attribute value length"))
        .context(StrContext::Expected("<= 0x1000".into()))
        .parse_next(input)? as usize;

    let unknown4 = le_u64.parse_next(input)?;

    let value = match unknown4 {
        0x00000001 => {
            let val = take(value_len).parse_to::<String>().parse_next(input)?;
            Some(val)
        }
        0x02000002 => None, // No value
        _ => {
            log::warn!("Unknown complex attribute format: {unknown4:#x}");
            return Err(ErrMode::assert(
                input,
                "Unsupported complex attribute format",
            ));
        }
    };

    log::info!("Complex attribute: {name} = {value:?}");
    Ok(Attribute {
        name: name
            .trim_matches(|c: char| c.is_whitespace() || c == '\0')
            .to_string(),
        value,
    })
}

fn indirect_block_read(
    input: &mut Input,
    fractal_heap: &FractalHeap,
    iblock_size: u64,
) -> ModalResult<FractalHeapData> {
    let size_of_offsets = input.state.size_of_offsets();
    let size_of_lengths = input.state.size_of_lengths();

    if input.state.recursive_counter() >= MAX_RECURSIVE_DEPTH {
        return Err(ErrMode::assert(input, "Recursive problem in fractal heap"));
    }

    input.state.recursive_counter_inc();

    let _signature = literal(FHIB_SIGNATURE).parse_next(input)?;

    let _version = le_u8
        .verify(|v| *v == 0)
        .context(StrContext::Label("FHIB version"))
        .context(StrContext::Expected("0".into()))
        .parse_next(input)?;

    // Skip heap header address
    let _heap_header_address = varint_size(size_of_offsets).parse_next(input)?;

    let size = fractal_heap.maximum_heap_size.div_ceil(8);
    let block_offset = varint_size(size as u8).parse_next(input)?;

    if block_offset != 0 {
        return Err(ErrMode::assert(input, "FHIB block offset is not 0"));
    }

    // Calculate nrows and max_dblock_rows using log2
    let nrows = (iblock_size.ilog2() - fractal_heap.starting_block_size.ilog2()) + 1;
    let max_dblock_rows = (fractal_heap.maximum_direct_block_size.ilog2()
        - fractal_heap.starting_block_size.ilog2())
        + 2;

    let k = if nrows < max_dblock_rows {
        nrows * fractal_heap.table_width as u32
    } else {
        max_dblock_rows * fractal_heap.table_width as u32
    };

    let n = if nrows <= max_dblock_rows {
        0
    } else {
        k - (max_dblock_rows * fractal_heap.table_width as u32)
    };

    let mut block_data = FractalHeapData {
        attributes: Vec::new(),
        directories: Vec::new(),
    };

    // Process direct blocks
    for _ in 0..k {
        let child_direct_block = varint_size(size_of_offsets).parse_next(input)?;

        if fractal_heap.encoded_length > 0 {
            let _size_filtered = varint_size(size_of_lengths).parse_next(input)?;
            let _filter_mask = le_u32.parse_next(input)?;
        }

        log::info!("Processing direct block at {child_direct_block:#x}");

        if input.state.is_address_valid(child_direct_block) {
            let cp = input.checkpoint();

            input.input.reset_to_start();
            take(child_direct_block as usize).parse_next(input)?;

            let direct_data = direct_block_read(input, fractal_heap)?;
            block_data.attributes.extend(direct_data.attributes);
            block_data.directories.extend(direct_data.directories);

            input.reset(&cp);
        }
    }

    // Process indirect blocks
    for _ in 0..n {
        let child_indirect_block = varint_size(size_of_offsets).parse_next(input)?;

        log::info!("Processing indirect block at {child_indirect_block:#x}");

        if input.state.is_address_valid(child_indirect_block) {
            let cp = input.checkpoint();

            input.input.reset_to_start();
            take(child_indirect_block as usize).parse_next(input)?;

            let indirect_data = indirect_block_read(input, fractal_heap, iblock_size * 2)?;
            block_data.attributes.extend(indirect_data.attributes);
            block_data.directories.extend(indirect_data.directories);

            input.reset(&cp);
        }
    }

    input.state.recursive_counter_dec();
    Ok(block_data)
}
