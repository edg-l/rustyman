# rustyman
Huffman compression and decompression implemented in rust

```rust
use rustyman::Huffman;

let payload = b"hello from the other side of the river";
    
let huffman = Huffman::new_from_data(payload);
let compressed = huffman.compress(payload);
let decompressed = huffman.decompress(&compressed);

assert!(compressed.len() < payload.len());
assert_eq!(&payload[..], decompressed)
```
