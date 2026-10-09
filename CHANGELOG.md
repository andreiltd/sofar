# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0](https://github.com/andreiltd/sofar/compare/sofar-v0.3.0...sofar-v0.4.0) - 2026-10-06

### Added

- [**breaking**] redesign renderer for real-time filter updates ([#48](https://github.com/andreiltd/sofar/pull/48))

### Fixed

- *(hdf)* parse object headers written by recent netCDF4/HDF5 ([#29](https://github.com/andreiltd/sofar/pull/29))
- make dependapbot to use conventional commits

### Other

- *(deps)* bump dtolnay/rust-toolchain ([#46](https://github.com/andreiltd/sofar/pull/46))
- *(deps)* bump taiki-e/install-action from 2.83.4 to 2.87.23 ([#45](https://github.com/andreiltd/sofar/pull/45))
- *(deps)* bump Swatinem/rust-cache from 2.9.1 to 2.9.2 ([#40](https://github.com/andreiltd/sofar/pull/40))
- *(deps)* bump mozilla-actions/sccache-action from 0.0.10 to 0.0.11 ([#36](https://github.com/andreiltd/sofar/pull/36))
- *(deps)* bump actions/checkout from 7.0.0 to 7.0.1 ([#34](https://github.com/andreiltd/sofar/pull/34))
- *(deps)* bump release-plz/action from 0.5.130 to 0.5.131 ([#32](https://github.com/andreiltd/sofar/pull/32))
- *(deps)* bump taiki-e/install-action from 2.81.7 to 2.83.4 ([#31](https://github.com/andreiltd/sofar/pull/31))
- *(deps)* bump actions/checkout from 6.0.3 to 7.0.0 ([#24](https://github.com/andreiltd/sofar/pull/24))
- *(deps)* bump EmbarkStudios/cargo-deny-action from 2.0.20 to 2.1.1 ([#30](https://github.com/andreiltd/sofar/pull/30))
- *(deps)* bump release-plz/action from 0.5.129 to 0.5.130 ([#20](https://github.com/andreiltd/sofar/pull/20))
- *(deps)* bump actions/checkout from 6.0.2 to 6.0.3 ([#17](https://github.com/andreiltd/sofar/pull/17))
- *(deps)* bump taiki-e/install-action from 2.81.1 to 2.81.7 ([#19](https://github.com/andreiltd/sofar/pull/19))
- *(deps)* bump deps ([#16](https://github.com/andreiltd/sofar/pull/16))
- *(deps)* update miniz_oxide requirement from 0.8.9 to 0.9.1 ([#14](https://github.com/andreiltd/sofar/pull/14))
- *(deps)* update rand requirement from 0.9 to 0.10 ([#10](https://github.com/andreiltd/sofar/pull/10))
- *(deps)* bump mozilla-actions/sccache-action ([#15](https://github.com/andreiltd/sofar/pull/15))
- Update ringbuf requirement from 0.4 to 0.5 ([#9](https://github.com/andreiltd/sofar/pull/9))
- enable sccache
- Update criterion requirement from 0.5 to 0.8 ([#13](https://github.com/andreiltd/sofar/pull/13))
- lint ci actions

## [0.3.0](https://github.com/andreiltd/sofar/compare/sofar-v0.2.1...sofar-v0.3.0) - 2026-03-14

### Added

- [**breaking**] rewrite libmysofa in Rust ([#5](https://github.com/andreiltd/sofar/pull/5))

### Other

- add sofa compatibility integration tests ([#7](https://github.com/andreiltd/sofar/pull/7))
