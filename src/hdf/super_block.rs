use winnow::binary::{le_u8, le_u16, le_u32};
use winnow::error::StrContext;
use winnow::prelude::*;
use winnow::token::literal;

use super::helpers::varint_size;

/// Signature used to quickly identify a file as being an HDF5 file.
///
/// ASCII C format:               [\211, H,    D,    F,    \r,   \n,  \032,  \n]
const FORMAT_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Version number of Superblock
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SuperBlockVersion(u8);

/// Only present in versions 0 and 1 of the superblock
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SuperBlockVersionExt {
    /// Version number of File’s Free Space Storage
    space_storage: u8,
    /// Version number of Root Group Symbol Table Entry
    root_group: u8,
    /// Version number of Shared Header Message Format
    shared_header: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct SuperBlock {
    pub size_of_offsets: u8,
    pub size_of_lengths: u8,
    pub base_address: u64,
    pub end_of_file_address: u64,
    pub root_group_object_header_address: u64,
    pub superblock_extension_address: Option<u64>,
}

fn version(input: &mut &[u8]) -> ModalResult<SuperBlockVersion> {
    le_u8
        .verify_map(|ver| match ver {
            0..3 => Some(SuperBlockVersion(ver)),
            _ => None,
        })
        .context(StrContext::Label("Super block version"))
        .context(StrContext::Expected("0, 1, 2 or 3".into()))
        .parse_next(input)
}

fn version_ext(input: &mut &[u8]) -> ModalResult<SuperBlockVersionExt> {
    (le_u8, le_u8, le_u8, le_u8)
        .map(
            |(space_storage, root_group, _reserved, shared_header)| SuperBlockVersionExt {
                space_storage,
                root_group,
                shared_header,
            },
        )
        .parse_next(input)
}

fn size_of_offsets(input: &mut &[u8]) -> ModalResult<u8> {
    le_u8
        .verify(|sz| matches!(sz, 2..=8))
        .context(StrContext::Label("Size of Offsets"))
        .context(StrContext::Expected("range 2 to 8".into()))
        .parse_next(input)
}

fn size_of_lengths(input: &mut &[u8]) -> ModalResult<u8> {
    le_u8
        .verify(|sz| matches!(sz, 2..=8))
        .context(StrContext::Label("Size of Lengths"))
        .context(StrContext::Expected("range 2 to 8".into()))
        .parse_next(input)
}

fn indexed_storage(ver: SuperBlockVersion) -> impl FnMut(&mut &[u8]) -> ModalResult<Option<u16>> {
    move |input| match ver.0 {
        1 => {
            let indexed_storage_internal_node_k = le_u16.parse_next(input)?;
            let _reserved = le_u32.parse_next(input)?;

            Ok(Some(indexed_storage_internal_node_k))
        }
        _ => Ok(None),
    }
}

fn super_block_ver_0_or_1(
    ver: SuperBlockVersion,
) -> impl FnMut(&mut &[u8]) -> ModalResult<SuperBlock> {
    move |input: &mut &[u8]| {
        let _version_ext = version_ext.parse_next(input)?;
        let size_of_offsets = size_of_offsets.parse_next(input)?;
        let size_of_lengths = size_of_lengths.parse_next(input)?;
        let _reserved = le_u8.verify(|r| *r == 0).parse_next(input)?;
        let _group_leaf_node_k = le_u16.parse_next(input)?;
        let _group_internal_node_k = le_u16.parse_next(input)?;
        let _file_consistency_flags = le_u32.verify(|f| *f == 0).parse_next(input)?;
        let _indexed_storage = indexed_storage(ver).parse_next(input)?;
        let base_address = varint_size(size_of_offsets)
            .verify(|a| *a == 0)
            .parse_next(input)?;
        let _address_of_file_free_space = varint_size(size_of_offsets).parse_next(input)?;
        let end_of_file_address = varint_size(size_of_offsets).parse_next(input)?;
        let _driver_info_block_address = varint_size(size_of_offsets).parse_next(input)?;
        let _link_name_offset = varint_size(size_of_offsets).parse_next(input)?;
        let root_group_object_header_address = varint_size(size_of_offsets).parse_next(input)?;
        let _cache_type = le_u32.verify(|t| *t <= 2).parse_next(input)?;

        Ok(SuperBlock {
            size_of_offsets,
            size_of_lengths,
            base_address,
            end_of_file_address,
            root_group_object_header_address,
            superblock_extension_address: None,
        })
    }
}

fn super_block_ver_2_or_3(input: &mut &[u8]) -> ModalResult<SuperBlock> {
    let size_of_offsets = size_of_offsets.parse_next(input)?;
    let size_of_lengths = size_of_lengths.parse_next(input)?;
    let _file_consistency_flags = le_u8.parse_next(input)?;
    let base_address = varint_size(size_of_offsets)
        .verify(|a| *a == 0)
        .parse_next(input)?;
    let super_block_extension_address = varint_size(size_of_offsets).parse_next(input)?;
    let end_of_file_address = varint_size(size_of_offsets).parse_next(input)?;
    let root_group_object_header_address = varint_size(size_of_offsets).parse_next(input)?;

    Ok(SuperBlock {
        size_of_offsets,
        size_of_lengths,
        base_address,
        end_of_file_address,
        root_group_object_header_address,
        superblock_extension_address: Some(super_block_extension_address),
    })
}

pub(crate) fn super_block(input: &mut &[u8]) -> ModalResult<SuperBlock> {
    let _signature = literal(FORMAT_SIGNATURE).parse_next(input)?;
    let ver = version.parse_next(input)?;

    match ver.0 {
        0 | 1 => super_block_ver_0_or_1(ver).parse_next(input),
        2 | 3 => super_block_ver_2_or_3.parse_next(input),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_signature_rejected() {
        let mut input: &[u8] = &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(super_block(&mut input).is_err());
    }

    #[test]
    fn test_invalid_version_rejected() {
        // Valid signature but invalid version (4)
        let mut input: &[u8] = &[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a, 0x04];
        assert!(super_block(&mut input).is_err());
    }

    #[test]
    fn test_size_of_offsets_valid_range() {
        // Test that values outside 2-8 are rejected
        let mut input: &[u8] = &[1]; // too small
        assert!(size_of_offsets(&mut input).is_err());

        let mut input: &[u8] = &[9]; // too large
        assert!(size_of_offsets(&mut input).is_err());

        let mut input: &[u8] = &[4]; // valid
        assert!(size_of_offsets(&mut input).is_ok());
    }
}
