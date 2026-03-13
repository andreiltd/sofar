pub(crate) use super::{
    data_object::{DataLayout, DataSpace},
    helpers::varint_size,
    parser::Input,
};

use miniz_oxide::inflate::decompress_to_vec_zlib_with_limit;
use winnow::ModalResult;
use winnow::Parser;
use winnow::binary::{le_u8, le_u16, le_u32, le_u64};
use winnow::combinator::repeat;
use winnow::error::{ErrMode, ParserError, StrContext};
use winnow::stream::Stream;
use winnow::token::{literal, take};

use log::info;

/// ASCII C format:                 [   T,    R,    E,    E]
pub const TREE_SIGNATURE: [u8; 4] = [0x54, 0x52, 0x45, 0x45];

pub(crate) fn tree(
    data_len: usize,
    data_space: DataSpace,
    data_layout: DataLayout,
) -> impl FnMut(&mut Input) -> ModalResult<Vec<u8>> {
    move |input| {
        // Validate data_len to prevent excessive allocation from malformed files
        const MAX_DATA_LEN: usize = 0x1000_0000; // 256MB limit, consistent with other size checks
        if data_len > MAX_DATA_LEN {
            return Err(ErrMode::assert(
                input,
                "Tree data_len exceeds maximum allowed size",
            ));
        }

        let dimensionality = data_space.dimensionality as usize;
        let size_of_offsets = input.state.size_of_offsets();
        let size_of_lengths = input.state.size_of_lengths();

        if dimensionality > 3 {
            return Err(ErrMode::assert(input, "Tree dimension is greater than 3"));
        }

        let _signature = literal(TREE_SIGNATURE).parse_next(input)?;
        let node_type = le_u8.parse_next(input)?;
        let _node_level = le_u8.parse_next(input)?;
        let entries_used = le_u16
            .verify(|e| *e <= 0x1000)
            .context(StrContext::Label("Tree entries used"))
            .context(StrContext::Expected("<= 0x1000".into()))
            .parse_next(input)?;

        let _address_of_left_sibling = varint_size(size_of_offsets).parse_next(input)?;
        let _address_of_right_sibling = varint_size(size_of_offsets).parse_next(input)?;

        let elements = data_layout
            .iter()
            .take(dimensionality)
            .fold(1, |acc, x| u32::saturating_mul(acc, *x));

        let size = data_layout
            .get(dimensionality)
            .copied()
            .ok_or_else(|| ErrMode::assert(input, "Data layout size index out of bounds"))?;

        info!("Tree elements: {elements}, size: {size}");

        if elements == 0 || size == 0 || elements >= 0x130000 || size > 0x10 {
            return Err(ErrMode::assert(input, "Invalid tree elements or size"));
        }

        let mut data = vec![0; data_len];

        for _ in 0..entries_used * 2 {
            if node_type == 0 {
                let _key = varint_size(size_of_lengths).parse_next(input)?;
            } else {
                let size_of_chunk = le_u32.parse_next(input)?;
                let _filter_mask = le_u32
                    .verify(|m| *m == 0)
                    .context(StrContext::Label("TREE filter mask"))
                    .context(StrContext::Expected(
                        "all filters must be enabled (0)".into(),
                    ))
                    .parse_next(input)?;

                let start: Vec<u64> = repeat(dimensionality, le_u64).parse_next(input)?;
                info!("start {start:#?}");

                let next = le_u64.parse_next(input)?;
                if next != 0 {
                    break;
                }

                let child_pointer = varint_size(size_of_offsets).parse_next(input)?;
                info!(" data at {child_pointer:#x} len {size_of_chunk}");

                if !input.state.is_address_valid(child_pointer) {
                    return Err(ErrMode::assert(input, "Invalid child pointer address"));
                }

                let cp = input.checkpoint();
                input.input.reset_to_start();

                let _skip = take(child_pointer as usize).parse_next(input)?;
                let chunk = take(size_of_chunk).parse_next(input)?;
                let olen = (elements * size) as usize;

                let output = decompress_to_vec_zlib_with_limit(chunk, olen)
                    .map_err(|_err| ErrMode::assert(input, "Failed to inflate btree data"))?;

                if output.len() != olen {
                    return Err(ErrMode::assert(input, "Invalid tree chunk length"));
                }

                // Safe array access with defaults
                let dy = data_layout.get(1).copied().unwrap_or(1) as u64;
                let dz = data_layout.get(2).copied().unwrap_or(1) as u64;
                let sx = data_space.dimension_size.first().copied().unwrap_or(1);
                let sy = data_space.dimension_size.get(1).copied().unwrap_or(1);
                let sz = data_space.dimension_size.get(2).copied().unwrap_or(1);
                let dzy = dz * dy;
                let szy = sz * sy;

                let data_len = data_len as u64;
                let olen = olen as u64;
                let size = size as u64;
                let elements = elements as u64;

                match dimensionality {
                    1 => {
                        for i in 0..olen {
                            let b = i / elements;
                            let x = i % elements + start[0];

                            if x < sx {
                                let j = x * size + b;
                                if j < data_len {
                                    data[j as usize] = output[i as usize];
                                }
                            }
                        }
                    }
                    2 => {
                        for i in 0..olen {
                            let b = i / elements;
                            let mut x = i % elements;
                            let y = x % dy + start[1];
                            x = x / dy + start[0];

                            if y < sy && x < sx {
                                let j = ((x * sy + y) * size) + b;
                                if j < data_len {
                                    data[j as usize] = output[i as usize];
                                }
                            }
                        }
                    }
                    3 => {
                        for i in 0..olen {
                            let b = i / elements;
                            let mut x = i % elements;
                            let z = x % dz + start[2];
                            let y = (x / dz) % dy + start[1];
                            x = (x / dzy) + start[0];

                            if z < sz && y < sy && x < sx {
                                let j = (x * szy + y * sz + z) * size + b;
                                if j < data_len {
                                    data[j as usize] = output[i as usize];
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(ErrMode::assert(input, "Invalid dimensionality"));
                    }
                }

                input.reset(&cp);
            }
        }

        let _checksum = take(4usize).parse_next(input)?;
        Ok(data)
    }
}
