# rustyman


[![Version](https://img.shields.io/crates/v/rustyman)](https://crates.io/crates/rustyman)
[![Downloads](https://img.shields.io/crates/d/rustyman)](https://crates.io/crates/rustyman)
[![License](https://img.shields.io/crates/l/rustyman)](https://crates.io/crates/rustyman)
![Rust](https://github.com/edg-l/rustyman/workflows/Rust/badge.svg)
[![Docs](https://docs.rs/rustyman/badge.svg)](https://docs.rs/rustyman)

Huffman compression and decompression implemented in rust

### Example

```rust
use rustyman::Huffman;

let payload = b"hello from the other side of the river";

let huffman = Huffman::new_from_data(payload);
let compressed = huffman.compress(payload);
let decompressed = huffman.decompress(&compressed);

assert!(compressed.len() < payload.len());
assert_eq!(&payload[..], decompressed);
```

License: MIT OR Apache-2.0
