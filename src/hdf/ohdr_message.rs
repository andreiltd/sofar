use winnow::binary::{le_u8, le_u16, le_u32, le_u64};
use winnow::combinator::{cond, cut_err, repeat};
use winnow::stream::{Location, Offset, Stream};
use winnow::token::{take, take_till};

use winnow::error::{ErrMode, ParserError, StrContext};
use winnow::{ModalResult, Parser};

use arrayvec::ArrayVec;
use bitflags::bitflags;

use crate::hdf::btree::tree;
use crate::hdf::data_object::DataLayout;
use crate::hdf::fractal_heap::Attribute;
use crate::hdf::gcol::gcol_read;

use super::data_object::{
    AttributeInfo, DataFormat, DataObjectFlags, DataSpace, DataType, GroupInfo, LinkInfo,
};
use super::helpers::varint_size;
use super::parser::Input;

const VALID_HEADER_MESSAGE_FLAGS: u8 = 0b00000101;
const MAX_CONTINUATION_DEPTH: u32 = 25;

bitflags! {
    #[derive(Clone, Copy, Debug)]
    pub struct HeaderMessageFlags: u8 {
        const MESSAGE_DATA_CONST    = 0b00000001;
        const MESSAGE_SHARED        = 0b00000010;
        const MESSAGE_NON_SHAREABLE = 0b00000100;
        const INTERNAL_1            = 0b00001000;
        const INTERNAL_2            = 0b00010000;
        const INTERNAL_3            = 0b00100000;
        const MESSAGE_SHAREABLE     = 0b01000000;
        const INTERNAL_4            = 0b10000000;
    }
}

#[derive(Clone, Debug)]
pub(crate) enum HeaderMessageKind {
    Nil,
    DataSpace(DataSpace),
    LinkInfo(LinkInfo),
    DataType(DataType),
    DataFillOld,
    DataFill,
    DataLayout(Vec<u8>),
    GroupInfo(GroupInfo),
    FilterPipeline,
    Attribute(Option<Attribute>),
    Continue { offset: u64, length: u64 },
    AttributeInfo(AttributeInfo),
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct HeaderMessage {
    pub kind: HeaderMessageKind,
    pub size: u16,
    pub flags: HeaderMessageFlags,
}

pub(crate) fn collect_all_messages(
    input: &mut Input,
    end_of_messages: usize,
    data_object_flags: DataObjectFlags,
) -> ModalResult<Vec<HeaderMessage>> {
    let mut all_messages = Vec::new();

    while input.current_token_start() < end_of_messages - 4 {
        let message = message_entry(data_object_flags).parse_next(input)?;

        match message.kind {
            HeaderMessageKind::Continue { offset, length } => {
                let continuation_messages =
                    message_continue(offset, length, data_object_flags).parse_next(input)?;
                all_messages.extend(continuation_messages);
            }
            _ => {
                all_messages.push(message);
            }
        }
    }

    Ok(all_messages)
}

fn message_entry(
    data_object_flags: DataObjectFlags,
) -> impl FnMut(&mut Input) -> ModalResult<HeaderMessage> {
    move |input| {
        let kind = le_u8.parse_next(input)?;
        let size = le_u16.parse_next(input)? as usize;

        let flags = le_u8
            .verify_map(|flags| {
                let valid_mask = HeaderMessageFlags::from_bits_truncate(VALID_HEADER_MESSAGE_FLAGS);
                let flags = HeaderMessageFlags::from_bits_truncate(flags);

                // Reject if any bit outside the valid mask is set
                flags.difference(valid_mask).is_empty().then_some(flags)
            })
            .context(StrContext::Label("OHDR message"))
            .context(StrContext::Expected("unsupported flags set".into()))
            .parse_next(input)?;

        cond(
            data_object_flags.contains(DataObjectFlags::ATTRIBUTE_CREATION_ORDER_TRACKED),
            le_u16,
        )
        .parse_next(input)?;

        let cp = input.checkpoint();
        let kind = message_kind(kind, size).parse_next(input)?;

        // Note: length mismatch can occur with unsupported data types
        // We log a warning but don't panic - allows partial parsing
        let consumed = input.offset_from(&cp);
        if consumed != size {
            log::warn!(
                "OHDR message length mismatch: expected {}, consumed {}",
                size,
                consumed
            );
            // Skip any remaining bytes to maintain alignment
            if consumed < size {
                let remaining = size - consumed;
                take(remaining).parse_next(input)?;
            }
        }

        Ok(HeaderMessage {
            kind,
            flags,
            size: size as _,
        })
    }
}

fn message_kind(
    kind: u8,
    header_size: usize,
) -> impl FnMut(&mut Input) -> ModalResult<HeaderMessageKind> {
    move |input| {
        Ok(match kind {
            0 => {
                message_nil(header_size).parse_next(input)?;
                HeaderMessageKind::Nil
            }
            1 => {
                let ds = message_data_space.parse_next(input)?;
                input.state.set_data_space(ds.clone());
                HeaderMessageKind::DataSpace(ds)
            }
            2 => {
                let li = message_link_info.parse_next(input)?;
                HeaderMessageKind::LinkInfo(li)
            }
            3 => {
                let dt = message_data_type.parse_next(input)?;
                HeaderMessageKind::DataType(dt)
            }
            4 => {
                message_data_fill_old.parse_next(input)?;
                HeaderMessageKind::DataFillOld
            }
            5 => {
                message_data_fill.parse_next(input)?;
                HeaderMessageKind::DataFill
            }
            8 => {
                let data = message_data_layout.parse_next(input)?;
                HeaderMessageKind::DataLayout(data)
            }
            10 => {
                let gi = message_group_info.parse_next(input)?;
                HeaderMessageKind::GroupInfo(gi)
            }
            11 => {
                message_filter_pipeline.parse_next(input)?;
                HeaderMessageKind::FilterPipeline
            }
            12 => {
                let attr = message_attribute.parse_next(input)?;
                HeaderMessageKind::Attribute(attr)
            }
            16 => {
                // Read continuation offset and length from message body
                let size_of_offsets = input.state.size_of_offsets();
                let size_of_lengths = input.state.size_of_lengths();

                let offset = varint_size(size_of_offsets)
                    .verify(|o| *o < 0x2000000)
                    .parse_next(input)?;
                let length = varint_size(size_of_lengths)
                    .verify(|l| *l < 0x10000000)
                    .parse_next(input)?;

                HeaderMessageKind::Continue { offset, length }
            }
            21 => {
                let ai = message_attribute_info.parse_next(input)?;
                HeaderMessageKind::AttributeInfo(ai)
            }
            _ => {
                // Skip unknown message types - the caller has already read the
                // message size, so we consume the remaining bytes and continue.
                log::warn!("Skipping unknown OHDR header message type {}", kind);
                take(header_size).parse_next(input)?;
                HeaderMessageKind::Nil
            }
        })
    }
}

fn message_nil(skip_len: usize) -> impl FnMut(&mut Input) -> ModalResult<()> {
    move |input| {
        take(skip_len).parse_next(input)?;
        Ok(())
    }
}

fn message_data_space(input: &mut Input) -> ModalResult<DataSpace> {
    let version = le_u8
        .verify(|ver| matches!(ver, 1..=2))
        .context(StrContext::Label("Object OHDR dataspace message"))
        .context(StrContext::Expected("1 or 2".into()))
        .parse_next(input)?;

    let dimensionality = le_u8
        .verify(|d| *d <= 4) // Move this check here
        .context(StrContext::Label("Object OHDR dataspace dimensionality"))
        .context(StrContext::Expected("<= 4".into()))
        .parse_next(input)?;

    let flags = le_u8.parse_next(input)?; // Remove the verify from here

    if version == 1 && flags & 2 != 0 {
        return Err(ErrMode::assert(
            input,
            "Permutation in OHDR is not supported",
        ));
    }

    let kind = match version {
        1 => {
            let _reserved = take(5usize).parse_next(input)?;
            None
        }
        2 => {
            let kind = le_u8.parse_next(input)?;
            Some(kind)
        }
        _ => unreachable!(),
    };

    let mut fill_arr = move |input: &mut Input| -> ModalResult<ArrayVec<u64, 4>> {
        let size_of_lengths = input.state.size_of_lengths();
        let dims = dimensionality as usize;

        let data: Vec<u64> = repeat(
            dims,
            cut_err(varint_size(size_of_lengths).verify(|d| *d <= 1_000_000)),
        )
        .context(StrContext::Label("Dimension Size"))
        .context(StrContext::Expected("dimension size <= 1,000,000".into()))
        .parse_next(input)?;

        let limited: ArrayVec<u64, 4> = data.into_iter().take(std::cmp::min(dims, 4)).collect();

        Ok(limited)
    };

    let dimension_size = fill_arr.parse_next(input)?;
    let dimension_max_size = cond(flags & 1 != 0, fill_arr)
        .parse_next(input)?
        .unwrap_or_default();

    Ok(DataSpace {
        dimension_size,
        dimension_max_size,
        dimensionality,
        flags,
        kind,
    })
}

fn message_link_info(input: &mut Input) -> ModalResult<LinkInfo> {
    let size_of_offsets = input.state.size_of_offsets();

    let _version = le_u8
        .verify(|ver| *ver == 0)
        .context(StrContext::Label("Object OHDR link info message version"))
        .context(StrContext::Expected("0".into()))
        .parse_next(input)?;

    let flags = le_u8.parse_next(input)?;

    let maximum_creation_index = cond(flags & 1 != 0, le_u64).parse_next(input)?;
    let fractal_heap_address = varint_size(size_of_offsets).parse_next(input)?;
    let address_btree_index = varint_size(size_of_offsets).parse_next(input)?;
    let address_btree_order =
        cond(flags & 2 != 0, varint_size(size_of_offsets)).parse_next(input)?;

    Ok(LinkInfo {
        flags,
        maximum_creation_index,
        fractal_heap_address,
        address_btree_index,
        address_btree_order,
    })
}

fn message_data_type(input: &mut Input) -> ModalResult<DataType> {
    let class_and_version = le_u8
        .verify(|cv| *cv & 0xf0 == 0x10 || *cv & 0xf0 == 0x30)
        .context(StrContext::Label(
            "Object OHDR data type message class and version",
        ))
        .context(StrContext::Expected("1".into()))
        .parse_next(input)?;

    // Class bit field is 3 bytes (24 bits), read as le_u24
    let class_bit_field_bytes: &[u8] = take(3usize).parse_next(input)?;
    let class_bit_field = u32::from_le_bytes([
        class_bit_field_bytes[0],
        class_bit_field_bytes[1],
        class_bit_field_bytes[2],
        0,
    ]);

    let size = le_u32.verify(|s| *s < 64).parse_next(input)?;

    let data_fmt = match class_and_version & 0xf {
        // int
        0 => {
            let bit_offset = le_u16.parse_next(input)?;
            let bit_precision = le_u16.parse_next(input)?;
            let data_fmt = DataFormat::Fixed {
                bit_offset,
                bit_precision,
            };

            Some(data_fmt)
        }
        // float
        1 => {
            let bit_offset = le_u16.parse_next(input)?;
            let bit_precision = le_u16.parse_next(input)?;
            let exponent_location = le_u8.parse_next(input)?;
            let exponent_size = le_u8.parse_next(input)?;
            let mantissa_location = le_u8.parse_next(input)?;
            let mantissa_size = le_u8.parse_next(input)?;
            let exponent_bias = le_u32.parse_next(input)?;

            if bit_offset != 0
                || mantissa_location != 0
                || (bit_precision != 32 && bit_precision != 64)
                || (bit_precision == 32
                    && (exponent_location != 23
                        || exponent_size != 8
                        || mantissa_size != 23
                        || exponent_bias != 127))
                || (bit_precision == 64
                    && (exponent_location != 52
                        || exponent_size != 11
                        || mantissa_size != 52
                        || exponent_bias != 1023))
            {
                log::warn!(
                    "Unsupported float format: bit_precision={}, exponent_location={}, exponent_size={}, mantissa_size={}, exponent_bias={}",
                    bit_precision,
                    exponent_location,
                    exponent_size,
                    mantissa_size,
                    exponent_bias
                );
                // Return a cut error instead of panicking - allows caller to handle gracefully
                return Err(ErrMode::Cut(winnow::error::ContextError::new()));
            }

            let data_fmt = DataFormat::Float {
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            };

            Some(data_fmt)
        }
        // string
        3 => None,
        // compound
        6 => {
            match class_and_version >> 4 {
                1 => {
                    for _ in 0..(class_bit_field & 0xffff) {
                        let cp = input.checkpoint();
                        let _name = take_till(0..256, '\0').parse_next(input)?;
                        let _null_byte = le_u8.parse_next(input)?;

                        let skip_bytes = (7 - input.offset_from(&cp)) & 7;
                        let _skip = take(skip_bytes).parse_next(input)?;

                        let _c = le_u32.parse_next(input)?;
                        let _dimension = le_u32
                            .verify(|d| *d == 0)
                            .context(StrContext::Label("Compound v1 dimension"))
                            .context(StrContext::Expected("0".into()))
                            .parse_next(input)?;

                        // ignore the following fields
                        let skip_bytes = (3 + 4 + 4 + 4 * 4) as usize;
                        let _skip = take(skip_bytes).parse_next(input)?;
                        let _dt = message_data_type.parse_next(input)?;
                    }
                }
                3 => {
                    for _ in 0..(class_bit_field & 0xffff) {
                        let _name = take_till(0..0x1000, '\0').parse_next(input)?;
                        let _null_byte = le_u8.parse_next(input)?;

                        let mut j = 0u32;
                        let mut _c = 0u32;

                        while size >> (8 * j) > 0 {
                            _c |= (le_u8.parse_next(input)? as u32) << (8 * j);
                            j += 1;
                        }

                        let _dt = message_data_type.parse_next(input)?;
                    }
                }
                _t => {
                    return Err(ErrMode::assert(
                        input,
                        "object OHDR compound datatype message must have version 1 or 3",
                    ));
                }
            }

            None
        }
        // reference
        7 => None,
        // list
        9 => None,
        _t => {
            return Err(ErrMode::assert(
                input,
                "object OHDR datatype message has unknown variable type",
            ));
        }
    };

    // Handle list
    if class_and_version & 0xf == 9 {
        let dt = message_data_type.parse_next(input)?;
        let other_fmt = dt.data_fmt;

        Ok(DataType {
            class_and_version,
            class_bit_field,
            size,
            data_fmt: data_fmt.or(other_fmt),
            list_size: Some(size),
        })
    } else {
        Ok(DataType {
            class_and_version,
            class_bit_field,
            size,
            data_fmt,
            list_size: None,
        })
    }
}

fn message_data_fill_old(input: &mut Input) -> ModalResult<()> {
    let size = le_u32.parse_next(input)?;
    take(size).parse_next(input)?;

    Ok(())
}

fn message_data_fill(input: &mut Input) -> ModalResult<()> {
    let version = le_u8
        .verify(|lc| matches!(lc, 1..=3))
        .context(StrContext::Label(
            "Object OHDR message data storage fill version",
        ))
        .context(StrContext::Expected("1, 2 or 3".into()))
        .parse_next(input)?;

    match version {
        1 | 2 => {
            let _space_allocation_time = le_u8
                .verify(|t| *t < 128 && *t & 0xFE == 2)
                .parse_next(input)?;
            let _fill_value_write_time = le_u8.verify(|t| *t < 128 && *t == 2).parse_next(input)?;
            let fill_value_defined = le_u8
                .verify(|t| *t < 128 && *t & 0xFE == 0)
                .parse_next(input)?;

            if fill_value_defined > 0 {
                let size = le_u32.parse_next(input)?;
                let _skip = take(size).parse_next(input)?;
            }
        }
        3 => {
            let flags = le_u8.parse_next(input)?;
            if flags & (1 << 5) != 0 {
                let size = le_u32.parse_next(input)?;
                let _skip = take(size).parse_next(input)?;
            }
        }
        _ => {
            unreachable!();
        }
    }

    Ok(())
}

fn message_data_layout(input: &mut Input) -> ModalResult<Vec<u8>> {
    let size_of_offsets = input.state.size_of_offsets();
    let size_of_lengths = input.state.size_of_lengths();

    let _version = le_u8
        .verify(|v| *v == 3)
        .context(StrContext::Label("Object OHDR message data layout version"))
        .context(StrContext::Expected("3".into()))
        .parse_next(input)?;

    let layout_class = le_u8
        .verify(|lc| matches!(lc, 0..=2))
        .context(StrContext::Label("Object OHDR message data layout class"))
        .context(StrContext::Expected("0, 1, or 2".into()))
        .parse_next(input)?;

    let data = match layout_class {
        0 => {
            let data_size = le_u16.parse_next(input)?;
            let _skip = take(data_size).parse_next(input)?;

            log::info!("TODO layout 0, size: {data_size}");
            vec![]
        }
        1 => {
            let data_address = varint_size(size_of_offsets).parse_next(input)?;
            let data_size = varint_size(size_of_lengths)
                .verify(|sz| *sz < 0x1000_0000)
                .context(StrContext::Label(
                    "Object OHDR message data layout, data size",
                ))
                .context(StrContext::Expected("< 0x10000000".into()))
                .parse_next(input)?;

            log::info!("CHUNK Contiguous SIZE: {data_size}");

            if input.state.is_address_valid(data_address) {
                // Use absolute seek to avoid underflow when data_address < cur_pos
                let cp = input.checkpoint();
                input.input.reset_to_start();
                let _skip = take(data_address as usize).parse_next(input)?;
                let data = take(data_size as usize).parse_next(input)?;

                input.reset(&cp);
                data.to_vec()
            } else {
                vec![]
            }
        }
        2 => {
            let dimensionality = le_u8
                .verify(|d| matches!(d, 1..=5))
                .context(StrContext::Label(
                    "Object OHDR message data layout 2 dimensionality",
                ))
                .context(StrContext::Expected("1..=5".into()))
                .parse_next(input)?;

            let data_address = varint_size(size_of_offsets).parse_next(input)?;
            log::info!("Dimensionality: {dimensionality}");
            log::info!("CHUNK at address: {data_address:#X}");

            let mut data_layout_chunk = DataLayout::new();

            for _ in 0..dimensionality {
                let item = le_u32.parse_next(input)?;
                data_layout_chunk.push(item);
            }

            let data_size = data_layout_chunk.last().copied().unwrap() as u64;
            let Some(data_space) = input.state.data_space() else {
                return Err(ErrMode::assert(input, "Data space is not available"));
            };

            // SAFETY, we check if dimensionality is non zero, so here we can
            // safely assume we have at least one element.
            let data_size = data_space
                .dimension_size
                .iter()
                .fold(data_size, |acc, s| u64::saturating_mul(acc, *s))
                as usize;

            if data_size > 0x1000_0000 {
                return Err(ErrMode::assert(
                    input,
                    "Object OHDR message data layout, data size too large",
                ));
            }

            // Note: The B-tree reader only supports data_space.dimensionality <= 3
            // The layout dimensionality can be 4 (includes element size) even for 3D data
            if input.state.is_address_valid(data_address) && data_space.dimensionality <= 3 {
                // Use absolute seek to avoid underflow when data_address < cur_pos
                let cp = input.checkpoint();
                input.input.reset_to_start();
                let _skip = take(data_address as usize).parse_next(input)?;
                let data = tree(data_size, data_space, data_layout_chunk).parse_next(input)?;

                input.reset(&cp);
                data
            } else {
                vec![]
            }
        }
        _ => {
            unreachable!();
        }
    };

    Ok(data)
}

fn message_group_info(input: &mut Input) -> ModalResult<GroupInfo> {
    let _version = le_u8
        .verify(|v| *v == 0)
        .context(StrContext::Label("Object OHDR group info version"))
        .context(StrContext::Expected("0".into()))
        .parse_next(input)?;

    let flags = le_u8.parse_next(input)?;
    let values = cond(flags & 1 != 0, (le_u16, le_u16)).parse_next(input)?;
    let entries = cond(flags & 2 != 0, (le_u16, le_u16)).parse_next(input)?;

    Ok(GroupInfo {
        flags,
        maximum_compact_value: values.map(|v| v.0),
        minimum_dense_value: values.map(|v| v.1),
        number_of_entries: entries.map(|v| v.0),
        length_of_entries: entries.map(|v| v.1),
    })
}

fn message_filter_pipeline(input: &mut Input) -> ModalResult<()> {
    let version = le_u8
        .verify(|v| matches!(v, 1..=2))
        .context(StrContext::Label("Filter pipeline version"))
        .context(StrContext::Expected("1 or 2".into()))
        .parse_next(input)?;

    let filters = le_u8
        .verify(|f| *f < 32)
        .context(StrContext::Label("Filters number"))
        .context(StrContext::Expected("< 32".into()))
        .parse_next(input)?;

    let mut filter_id = |input: &mut Input| -> ModalResult<u16> {
        le_u16
            .context(StrContext::Label("Filter identification value"))
            .parse_next(input)
    };

    match version {
        1 => {
            let _reserved = take(6usize)
                .verify(|s: &[u8]| !s.iter().any(|x| *x != 0))
                .context(StrContext::Label("Filters pipeline v1, reserved value"))
                .context(StrContext::Expected("0".into()))
                .parse_next(input)?;

            for _ in 0..filters {
                let filter_id_value = filter_id.parse_next(input)?;
                let name_len = le_u16.parse_next(input)?;
                let _flags = le_u16.parse_next(input)?;
                let values = le_u16.verify(|v| *v <= 0x1000).parse_next(input)?;

                log::info!("  filter {filter_id_value} namelen {name_len} values {values}");

                let skip = ((name_len - 1) & !7) + 8;
                take(skip).parse_next(input)?;
                repeat(values as usize, le_u32)
                    .map(|()| ())
                    .parse_next(input)?;
                cond((values & 1) == 1, le_u32).parse_next(input)?;
            }
        }
        2 => {
            for _ in 0..filters {
                let filter_id_value = filter_id.parse_next(input)?;
                let _flags = le_u16.parse_next(input)?;
                let values = le_u16.verify(|v| *v <= 0x1000).parse_next(input)?;

                log::info!("  filter {filter_id_value}");
                repeat(values as usize, le_u32)
                    .map(|()| ())
                    .parse_next(input)?;
            }
        }
        _ => {
            unreachable!();
        }
    }

    Ok(())
}

/// Parse OHDR Attribute message (type 0x0C).
///
/// Parses the attribute header and extracts string values from the data.
/// For non-string datatypes, returns None for the value.
fn message_attribute(input: &mut Input) -> ModalResult<Option<Attribute>> {
    let version = le_u8
        .verify(|v| *v == 1 || *v == 3)
        .context(StrContext::Label("Object OHDR attribute version"))
        .context(StrContext::Expected("1 or 3".into()))
        .parse_next(input)?;

    let _flags = le_u8.parse_next(input)?;
    let name_size = le_u16.verify(|s| *s <= 0x1000).parse_next(input)?;
    let _datatype_size = le_u16.parse_next(input)?;
    let _dataspace_size = le_u16.parse_next(input)?;
    let _encoding = cond(version == 3, le_u8).parse_next(input)?;

    // Read attribute name
    let name_bytes: &[u8] = take(name_size).parse_next(input)?;
    let name = String::from_utf8_lossy(name_bytes)
        .trim_end_matches('\0')
        .to_string();

    // Version 1 pads name to 8-byte boundary
    if version == 1 {
        let padding = (8usize.saturating_sub(name_size as usize)) & 7;
        if padding > 0 {
            take(padding).parse_next(input)?;
        }
    }

    // Parse the datatype to determine the data class and size.
    // We save a checkpoint in case parsing datatype/dataspace/data fails,
    // so we can fall back to skipping the remaining bytes.
    let dt_cp = input.checkpoint();

    let result: Option<String> = (|| -> Result<Option<String>, _> {
        let dt = message_data_type.parse_next(input)?;

        // Version 1 pads datatype to 8-byte boundary
        if version == 1 {
            let dt_consumed = input.offset_from(&dt_cp);
            let dt_padding = ((8usize.wrapping_sub(dt_consumed)) & 7) % 8;
            if dt_padding > 0 {
                take(dt_padding).parse_next(input)?;
            }
        }

        let ds_cp = input.checkpoint();
        let ds = message_data_space.parse_next(input)?;

        // Version 1 pads dataspace to 8-byte boundary
        if version == 1 {
            let ds_consumed = input.offset_from(&ds_cp);
            let ds_padding = ((8usize.wrapping_sub(ds_consumed)) & 7) % 8;
            if ds_padding > 0 {
                take(ds_padding).parse_next(input)?;
            }
        }

        let num_elements: u64 = if ds.dimension_size.is_empty() {
            1 // scalar
        } else {
            ds.dimension_size.iter().product()
        };

        let data_class = dt.class_and_version & 0xf;
        match data_class {
            // class 3: fixed-length string
            3 if dt.size > 0 && dt.size <= 0x1000 => {
                let total_size = dt.size as usize * num_elements as usize;
                let val_bytes: &[u8] = take(total_size).parse_next(input)?;
                Ok(Some(
                    String::from_utf8_lossy(val_bytes)
                        .trim_end_matches('\0')
                        .to_string(),
                ))
            }
            // class 9: variable-length. The data contains a GCOL address +
            // reference that points to the actual string. Read the GCOL
            // collection and resolve the value.
            9 if dt.list_size.is_some() => {
                let list_size = dt.list_size.unwrap() as usize;

                // Read GCOL address from the list prefix
                let gcol_prefix_size = list_size - dt.size as usize;
                let gcol_address = if gcol_prefix_size == 8 {
                    let _unknown = le_u32.parse_next(input)?;
                    varint_size(4u8).parse_next(input)?
                } else if gcol_prefix_size > 0 {
                    varint_size(gcol_prefix_size as u8).parse_next(input)?
                } else {
                    0
                };

                // Read the reference data: 4 bytes unknown + reference ID
                let reference_size = dt.size as usize;
                if reference_size >= 4 {
                    let _unknown_ref = le_u32.parse_next(input)?;
                    let reference = if reference_size > 4 {
                        varint_size((reference_size - 4) as u8).parse_next(input)? as u16
                    } else {
                        0
                    };

                    // Seek to GCOL address and read the collection
                    if input.state.is_address_valid(gcol_address) {
                        let gcol_cp = input.checkpoint();
                        input.input.reset_to_start();
                        take(gcol_address as usize).parse_next(input)?;

                        if let Ok(objects) = gcol_read(input) {
                            input.reset(&gcol_cp);

                            // Find the object with matching reference
                            if let Some(obj) =
                                objects.iter().find(|o| o.heap_object_index == reference)
                            {
                                // The value is a data object address. Seek there
                                // and read the object name as the string value.
                                let obj_addr = obj.value;
                                if input.state.is_address_valid(obj_addr) {
                                    let obj_cp = input.checkpoint();
                                    input.input.reset_to_start();
                                    take(obj_addr as usize).parse_next(input)?;
                                    if let Ok(data_obj) =
                                        super::data_object::data_object("ref").parse_next(input)
                                    {
                                        input.reset(&obj_cp);
                                        return Ok(Some(data_obj.name));
                                    }
                                    input.reset(&obj_cp);
                                }
                            }
                        } else {
                            input.reset(&gcol_cp);
                        }
                    }
                } else {
                    take(reference_size).parse_next(input)?;
                }
                Ok(None)
            }
            _ => {
                // For non-string types, read and discard the data
                let element_size = dt.list_size.unwrap_or(dt.size) as usize;
                let total_size = element_size * num_elements as usize;
                if total_size > 0 && total_size <= 0x100000 {
                    take(total_size).parse_next(input)?;
                }
                Ok(None)
            }
        }
    })()
    .unwrap_or_else(|_: winnow::error::ErrMode<winnow::error::ContextError>| {
        // If parsing failed, reset and let the message_entry skip handler deal with it
        input.reset(&dt_cp);
        None
    });

    Ok(Some(Attribute {
        name,
        value: result,
    }))
}

fn message_attribute_info(input: &mut Input) -> ModalResult<AttributeInfo> {
    let size_of_offsets = input.state.size_of_offsets();

    let _version = le_u8
        .verify(|v| *v == 0)
        .context(StrContext::Label(
            "Object OHDR attribute info message version",
        ))
        .context(StrContext::Expected("0".into()))
        .parse_next(input)?;

    let flags = le_u8.parse_next(input)?;

    let maximum_creation_index = cond(flags & 1 != 0, le_u16)
        .parse_next(input)?
        .map(|v| v as u64)
        .unwrap_or(0);

    let fractal_heap_address = varint_size(size_of_offsets).parse_next(input)?;
    let attribute_name_btree = varint_size(size_of_offsets).parse_next(input)?;

    let attribute_creation_order_btree = cond(flags & 2 != 0, varint_size(size_of_offsets))
        .parse_next(input)?
        .unwrap_or(0);

    Ok(AttributeInfo {
        flags,
        maximum_creation_index,
        fractal_heap_address,
        attribute_name_btree,
        attribute_creation_order_btree,
    })
}

fn message_continue(
    offset: u64,
    length: u64,
    data_object_flags: DataObjectFlags,
) -> impl FnMut(&mut Input) -> ModalResult<Vec<HeaderMessage>> {
    move |input| {
        log::info!(" continue {offset:#x} {length:#x}");

        if input.state.recursive_counter() >= MAX_CONTINUATION_DEPTH {
            return Err(ErrMode::assert(input, "Recursive problem"));
        }

        let cp = input.checkpoint();
        input.state.recursive_counter_inc();

        // Seek to continuation chunk
        input.input.reset_to_start();
        take(offset as usize).parse_next(input)?;

        let _ochk_signature = "OCHK".parse_next(input)?;

        let end_of_continuation = input.current_token_start() + length as usize - 4;
        let mut continuation_messages = Vec::new();

        while input.current_token_start() < end_of_continuation - 4 {
            let message = message_entry(data_object_flags).parse_next(input)?;

            match message.kind {
                HeaderMessageKind::Continue {
                    offset: nested_offset,
                    length: nested_length,
                } => {
                    let nested_messages =
                        message_continue(nested_offset, nested_length, data_object_flags)
                            .parse_next(input)?;
                    continuation_messages.extend(nested_messages);
                }
                _ => {
                    continuation_messages.push(message);
                }
            }
        }

        take(4usize).parse_next(input)?;

        // Restore position and decrement counter
        input.reset(&cp);
        input.state.recursive_counter_dec();

        log::info!(" continue back");
        Ok(continuation_messages)
    }
}
