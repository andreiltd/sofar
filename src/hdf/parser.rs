use winnow::error::ParserError;
use winnow::stream::{LocatingSlice, Location, Stateful, Stream};
use winnow::token::take;
use winnow::{error::ErrMode, prelude::*};

use super::data_object::{DataObject, DataSpace, data_object};
use super::super_block::{SuperBlock, super_block};

pub(crate) type Input<'a> = Stateful<LocatingSlice<&'a [u8]>, State>;

/// Context state that is available after parsing Super Block
#[derive(Debug, Clone)]
pub(crate) struct State {
    size_of_lengths: u8,
    size_of_offsets: u8,
    end_of_file_address: u64,
    recursive_counter: u32,
    data_space: Option<DataSpace>,
}

impl State {
    pub fn new(block: &SuperBlock) -> Self {
        Self {
            size_of_lengths: block.size_of_lengths,
            size_of_offsets: block.size_of_offsets,
            end_of_file_address: block.end_of_file_address,
            recursive_counter: 0,
            data_space: None,
        }
    }

    pub fn size_of_lengths(&self) -> u8 {
        self.size_of_lengths
    }

    pub fn size_of_offsets(&self) -> u8 {
        self.size_of_offsets
    }

    #[allow(dead_code)]
    pub fn end_of_file_address(&self) -> u64 {
        self.end_of_file_address
    }

    pub fn is_address_valid(&self, address: u64) -> bool {
        address > 0 && address < self.end_of_file_address
    }

    pub(crate) fn data_space(&self) -> Option<DataSpace> {
        self.data_space.clone()
    }

    pub(crate) fn set_data_space(&mut self, data_space: DataSpace) {
        self.data_space = Some(data_space);
    }

    pub(crate) fn recursive_counter(&self) -> u32 {
        self.recursive_counter
    }

    pub(crate) fn recursive_counter_inc(&mut self) {
        self.recursive_counter = self.recursive_counter.saturating_add(1);
    }

    pub(crate) fn recursive_counter_dec(&mut self) {
        self.recursive_counter = self.recursive_counter.saturating_sub(1);
    }
}

/// Parsed HDF5 file with ability to navigate to child objects.
pub struct ParsedHdf<'a> {
    data: &'a [u8],
    state: State,
    /// The root data object
    pub root: DataObject,
}

impl<'a> ParsedHdf<'a> {
    /// Parse a child data object by address.
    ///
    /// Use addresses from `root.child_directories` to navigate the tree.
    pub fn parse_child(&self, name: &str, address: u64) -> ModalResult<DataObject> {
        if !self.state.is_address_valid(address) {
            return Err(ErrMode::assert(&self.data, "Invalid child object address"));
        }

        let input = LocatingSlice::new(self.data);
        let mut stream = Input {
            input,
            state: self.state.clone(),
        };

        let _skip = take(address as usize).parse_next(&mut stream)?;
        data_object(name).parse_next(&mut stream)
    }

    /// Find a child by name and parse it.
    pub fn get_child(&self, name: &str) -> Option<ModalResult<DataObject>> {
        self.root
            .child_directories
            .iter()
            .find(|d| d.name == name)
            .map(|d| self.parse_child(&d.name, d.address))
    }
}

/// Parse an HDF5/SOFA file and return a navigable structure.
pub fn parse_with_children(input: &[u8]) -> ModalResult<ParsedHdf<'_>> {
    let mut slice = input;
    let cp = slice.checkpoint();
    let super_block = super_block.parse_next(&mut slice)?;

    slice.reset(&cp);

    if super_block.end_of_file_address as usize != slice.eof_offset() {
        return Err(ErrMode::assert(&slice, "File size mismatch"));
    }

    let state = State::new(&super_block);
    let locating = LocatingSlice::new(slice);

    let mut stream = Input {
        input: locating,
        state: state.clone(),
    };

    let _skip =
        take(super_block.root_group_object_header_address as usize).parse_next(&mut stream)?;
    let root = data_object("root").parse_next(&mut stream)?;

    Ok(ParsedHdf {
        data: input,
        state,
        root,
    })
}

pub fn parse(mut input: &[u8]) -> ModalResult<DataObject> {
    let cp = input.checkpoint();
    let super_block = super_block.parse_next(&mut input)?;

    input.reset(&cp);

    if super_block.end_of_file_address as usize != input.eof_offset() {
        log::error!(
            "File size mismatch: header says {}, actual {}",
            super_block.end_of_file_address,
            input.eof_offset()
        );
        return Err(ErrMode::assert(&input, "File size mismatch"));
    }

    log::debug!(
        "SuperBlock parsed: offsets={}, lengths={}, root_addr={:#x}",
        super_block.size_of_offsets,
        super_block.size_of_lengths,
        super_block.root_group_object_header_address
    );

    let state = State::new(&super_block);
    let input = LocatingSlice::new(input);

    let mut stream = Input { input, state };

    // jump to the first object
    let _skip =
        take(super_block.root_group_object_header_address as usize).parse_next(&mut stream)?;

    log::debug!(
        "About to parse root data_object at position {:#x}",
        stream.input.current_token_start()
    );

    match data_object("root").parse_next(&mut stream) {
        Ok(obj) => {
            log::debug!("Root object parsed successfully: {}", obj.name);
            Ok(obj)
        }
        Err(e) => {
            log::error!("Failed to parse root object: {:?}", e);
            Err(e)
        }
    }
}
