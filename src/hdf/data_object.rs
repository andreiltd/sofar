use winnow::binary::le_u8;
use winnow::stream::Location;
use winnow::stream::Stream;
use winnow::token::{literal, take};

use winnow::ModalResult;
use winnow::Parser;
use winnow::error::StrContext;

use arrayvec::ArrayVec;
use bitflags::bitflags;

use super::fractal_heap::{Attribute, DirectoryEntry, FractalHeap, fractal_heap_read};
use super::ohdr_message::{HeaderMessage, HeaderMessageKind, collect_all_messages};

use super::helpers::varint_size;
use super::parser::Input;

/// ASCII C format:             [   O,    H,    D,    R]
pub const OHDR_SIGNATURE: [u8; 4] = [0x4F, 0x48, 0x44, 0x52];
pub const DATAOBJECT_MAX_DIMENSIONALITY: usize = 5;

/// Data Layout Chunk alias
pub(crate) type DataLayout = ArrayVec<u32, DATAOBJECT_MAX_DIMENSIONALITY>;

#[derive(Clone, Copy, Debug)]
pub enum DataFormat {
    Fixed {
        bit_offset: u16,
        bit_precision: u16,
    },
    Float {
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum Record {
    Type5 { hash_of_name: u32, heap_id: u64 },
}

impl Default for Record {
    fn default() -> Self {
        Record::Type5 {
            hash_of_name: 0,
            heap_id: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DataSpace {
    pub dimension_size: ArrayVec<u64, 4>,
    pub dimension_max_size: ArrayVec<u64, 4>,
    pub dimensionality: u8,
    pub flags: u8,
    pub kind: Option<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinkInfo {
    pub flags: u8,
    pub maximum_creation_index: Option<u64>,
    pub fractal_heap_address: u64,
    pub address_btree_index: u64,
    pub address_btree_order: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GroupInfo {
    pub flags: u8,
    pub maximum_compact_value: Option<u16>,
    pub minimum_dense_value: Option<u16>,
    pub number_of_entries: Option<u16>,
    pub length_of_entries: Option<u16>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AttributeInfo {
    pub flags: u8,
    pub maximum_creation_index: u64,
    pub fractal_heap_address: u64,
    pub attribute_name_btree: u64,
    pub attribute_creation_order_btree: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DataType {
    pub class_and_version: u8,
    pub class_bit_field: u32,
    pub size: u32,
    pub list_size: Option<u32>,
    pub data_fmt: Option<DataFormat>,
}

#[derive(Clone, Debug, Default)]
pub struct BinaryTree {
    pub kind: u8,
    pub split_percent: u8,
    pub merge_percent: u8,
    pub record_size: u16,
    pub depth: u16,
    pub number_of_records: u16,
    pub node_size: u32,
    pub root_node_address: u64,
    pub total_number: u64,
    pub records: Vec<Record>,
}

#[derive(Clone, Debug)]
pub struct DataObject {
    pub name: String,
    pub address: u64,
    pub flags: DataObjectFlags,
    pub dt: DataType,
    pub ds: DataSpace,
    pub li: LinkInfo,
    pub ai: AttributeInfo,
    pub gi: GroupInfo,

    pub objects_btree: BinaryTree,
    pub objects_heap: FractalHeap,
    pub attributes_btree: BinaryTree,
    pub attributes_heap: FractalHeap,
    pub data_layout_chunk: DataLayout,

    pub data: Vec<u8>,
    pub parsed_attributes: Vec<Attribute>,
    pub child_directories: Vec<DirectoryEntry>,
}

bitflags! {
    #[derive(Clone, Copy, Debug)]
    pub struct DataObjectFlags: u8 {
        const SIZE_OF_CHUNK                    = 0b00000011;
        const ATTRIBUTE_CREATION_ORDER_TRACKED = 0b00000100;
        const ATTRIBUTE_CREATION_ORDER_INDEXED = 0b00001000;
        const NON_DEFAULT_ATTRIBUTES_STORED    = 0b00010000;
        const TIMESTAMPS_STORED                = 0b00100000;
    }
}

/// Version number of Superblock
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OhdrVersion(u8);

fn build_data_object_from_messages(
    name: String,
    address: u64,
    flags: DataObjectFlags,
    messages: Vec<HeaderMessage>,
) -> DataObject {
    let mut data_object = DataObject {
        name,
        address,
        flags,
        dt: DataType::default(),
        ds: DataSpace::default(),
        li: LinkInfo::default(),
        ai: AttributeInfo::default(),
        gi: GroupInfo::default(),
        objects_btree: BinaryTree::default(),
        objects_heap: FractalHeap::default(),
        attributes_btree: BinaryTree::default(),
        attributes_heap: FractalHeap::default(),
        data_layout_chunk: DataLayout::new(),
        data: Vec::new(),
        parsed_attributes: Vec::new(),
        child_directories: Vec::new(),
    };

    for message in messages {
        match message.kind {
            HeaderMessageKind::DataSpace(ds) => data_object.ds = ds,
            HeaderMessageKind::LinkInfo(li) => data_object.li = li,
            HeaderMessageKind::DataType(dt) => data_object.dt = dt,
            HeaderMessageKind::AttributeInfo(ai) => data_object.ai = ai,
            HeaderMessageKind::DataLayout(data) => data_object.data = data,
            HeaderMessageKind::GroupInfo(gi) => data_object.gi = gi,
            HeaderMessageKind::Attribute(Some(attr)) => {
                data_object.parsed_attributes.push(attr);
            }
            _ => {}
        }
    }

    data_object
}

pub(crate) fn data_object(
    name: impl ToString,
) -> impl FnMut(&mut Input) -> ModalResult<DataObject> {
    move |input| {
        let address = input.current_token_start() as u64;
        log::debug!("data_object: parsing at address {:#x}", address);

        let _signature = literal(OHDR_SIGNATURE)
            .context(StrContext::Label("data_object signature"))
            .parse_next(input)?;
        log::debug!("data_object: OHDR signature OK");

        let _version = ohdr_version
            .context(StrContext::Label("data_object version"))
            .parse_next(input)?;
        log::debug!("data_object: version OK");

        let flags = data_object_flags
            .context(StrContext::Label("data_object flags"))
            .parse_next(input)?;
        log::debug!("data_object: flags={:?}", flags);

        // skip timestamps if stored
        if flags.contains(DataObjectFlags::TIMESTAMPS_STORED) {
            take(16usize).parse_next(input)?;
            log::debug!("data_object: skipped timestamps");
        }

        let size_of_chunk_size = 1usize << (flags & DataObjectFlags::SIZE_OF_CHUNK).bits();
        let size_of_chunk = varint_size(size_of_chunk_size)
            .verify(|sz| *sz <= 0x0100_0000)
            .context(StrContext::Label("data_object chunk_size"))
            .parse_next(input)?;
        log::debug!(
            "data_object: chunk_size={}, end_pos={:#x}",
            size_of_chunk,
            input.current_token_start() + size_of_chunk as usize
        );

        let end_of_messages = input.current_token_start() + size_of_chunk as usize;
        log::debug!("data_object: parsing messages...");
        let messages = match collect_all_messages(input, end_of_messages, flags) {
            Ok(m) => {
                log::debug!("data_object: collected {} messages", m.len());
                m
            }
            Err(e) => {
                log::error!("data_object: collect_all_messages failed: {:?}", e);
                return Err(e);
            }
        };

        // Skip final checksum
        take(4usize).parse_next(input)?;
        log::debug!("data_object: building from messages");
        let mut data_object =
            build_data_object_from_messages(name.to_string(), address, flags, messages);
        log::debug!(
            "data_object: built, ai.fractal_heap={:#x}, li.fractal_heap={:#x}",
            data_object.ai.fractal_heap_address,
            data_object.li.fractal_heap_address
        );

        // Process attributes fractal heap
        if input
            .state
            .is_address_valid(data_object.ai.fractal_heap_address)
        {
            let cp = input.checkpoint();
            input.input.reset_to_start();
            take(data_object.ai.fractal_heap_address as usize).parse_next(input)?;

            let (heap, heap_data) = fractal_heap_read.parse_next(input)?;
            data_object.attributes_heap = heap;
            data_object.parsed_attributes.extend(heap_data.attributes);

            input.reset(&cp);
        }

        // Process objects fractal heap
        if input
            .state
            .is_address_valid(data_object.li.fractal_heap_address)
        {
            let cp = input.checkpoint();
            input.input.reset_to_start();
            take(data_object.li.fractal_heap_address as usize).parse_next(input)?;

            let (heap, heap_data) = fractal_heap_read.parse_next(input)?;
            data_object.objects_heap = heap;
            data_object.child_directories.extend(heap_data.directories);

            input.reset(&cp);
        }

        log::debug!("data_object: SUCCESS name={}", data_object.name);
        Ok(data_object)
    }
}

fn ohdr_version(input: &mut Input) -> ModalResult<OhdrVersion> {
    le_u8
        .verify_map(|ver| match ver {
            2 => Some(OhdrVersion(ver)),
            _ => None,
        })
        .context(StrContext::Label("OHDR version"))
        .context(StrContext::Expected("2".into()))
        .parse_next(input)
}

fn data_object_flags(input: &mut Input) -> ModalResult<DataObjectFlags> {
    le_u8
        .verify_map(|flags| {
            let flags = DataObjectFlags::from_bits_truncate(flags);

            (!flags.contains(DataObjectFlags::NON_DEFAULT_ATTRIBUTES_STORED)).then_some(flags)
        })
        .context(StrContext::Label("OHDR flags"))
        .context(StrContext::Expected(
            "unsupported flags bit 4 (Non-default attributes) set".into(),
        ))
        .parse_next(input)
}
