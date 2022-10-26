//!
//! [![Version](https://img.shields.io/crates/v/rustyman)](https://crates.io/crates/rustyman)
//! [![Downloads](https://img.shields.io/crates/d/rustyman)](https://crates.io/crates/rustyman)
//! [![License](https://img.shields.io/crates/l/rustyman)](https://crates.io/crates/rustyman)
//! ![Rust](https://github.com/edg-l/rustyman/workflows/Rust/badge.svg)
//! [![Docs](https://docs.rs/rustyman/badge.svg)](https://docs.rs/rustyman)
//!
//! Huffman compression and decompression implemented in rust
//!
//! ## Example
//!
//! ```rust
//! use rustyman::Huffman;
//!
//! let payload = b"hello from the other side of the river";
//!     
//! let huffman = Huffman::new_from_data(payload);
//! let compressed = huffman.compress(payload);
//! let decompressed = huffman.decompress(&compressed);
//!
//! assert!(compressed.len() < payload.len());
//! assert_eq!(&payload[..], decompressed);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::missing_doc_code_examples)]

use std::collections::BinaryHeap;

use bit_vec::BitVec;

// Created with help from:
// - https://en.wikipedia.org/wiki/Huffman_coding
// - https://aquarchitect.github.io/swift-algorithm-club/Huffman%20Coding/

/// Max symbols in the frequency table. Covers all possible u8 values.
pub const MAX_SYMBOLS: usize = u8::MAX as usize + 1;

const TREE_SIZE: usize = MAX_SYMBOLS * 2 - 1;

#[derive(Debug, Clone, Copy)]
struct Node {
    pub index: usize,
    pub count: usize,
    pub parent: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl Node {
    fn new(index: usize, count: usize) -> Self {
        Self {
            index,
            count,
            parent: None,
            left: None,
            right: None,
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        other.count.eq(&self.count)
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.count.cmp(&self.count))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.count.cmp(&self.count)
    }
}

/// Holds the data needed to (de)compress.
///
/// - Compress with [Self::compress]
/// - Decompress with [Self::decompress]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Huffman {
    tree: [Node; TREE_SIZE],
    root_index: usize,
}

impl Huffman {
    /// Initializes the huffman interface using the provided frequency table.
    pub fn new(frequency_table: &[usize; MAX_SYMBOLS]) -> Self {
        let mut tree = std::array::from_fn(|i| Node::new(i, 0));
        let root_index = Self::build_tree(&mut tree, frequency_table);

        Self { tree, root_index }
    }

    /// Creates the Huffman frequency table from the provided data and initializes from it.
    pub fn new_from_data(data: &[u8]) -> Self {
        let mut table = [0; MAX_SYMBOLS];
        Self::calculate_freq_table(&mut table, data);
        Self::new(&table)
    }

    /// Calculates the frequency table from the provided data.
    pub fn calculate_freq_table(table: &mut [usize; MAX_SYMBOLS], data: &[u8]) {
        table.fill(0);
        for i in data {
            table[*i as usize] += 1;
        }
    }

    /// Builds a binary tree, the root is the last node, the leafs are at the start.
    ///
    /// Returns the root index.
    fn build_tree(tree: &mut [Node; TREE_SIZE], table: &[usize; MAX_SYMBOLS]) -> usize {
        let mut priority_queue: BinaryHeap<Node> = table
            .iter()
            .enumerate()
            .map(|(index, v)| Node::new(index, *v))
            .collect();

        let mut tree_index = 256;

        while priority_queue.len() > 1 {
            let node1 = priority_queue.pop().unwrap();
            let node2 = priority_queue.pop().unwrap();

            tree[node1.index] = node1;
            tree[node2.index] = node2;
            tree[node1.index].parent = Some(tree_index);
            tree[node2.index].parent = Some(tree_index);

            let parent = Node {
                count: node1.count + node2.count,
                left: Some(node1.index),
                right: Some(node2.index),
                parent: None,
                index: tree_index,
            };
            tree[tree_index] = parent;
            tree_index += 1;

            priority_queue.push(parent);
        }

        priority_queue.pop().unwrap().index
    }

    // Recursively walk to the root and back to calculate the bits.
    fn traverse(&self, bits: &mut BitVec, index: usize, child_index: Option<usize>) {
        // First walk up to the root
        if let Some(parent) = self.tree[index].parent {
            self.traverse(bits, parent, Some(index));
        }

        // Then walk down back while pushing the bits.
        if let Some(child_index) = child_index {
            if Some(child_index) == self.tree[index].left {
                bits.push(true);
            } else if Some(child_index) == self.tree[index].right {
                bits.push(false);
            }
        }
    }

    /// Compresses the provided data.
    pub fn compress(&self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut bits = BitVec::new();

        for b in data.iter() {
            self.traverse(&mut bits, *b as usize, None)
        }

        bits.to_bytes()
    }

    /// Decompresses the provided data.
    pub fn decompress(&self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let bits = BitVec::from_bytes(data);
        let mut decompressed = Vec::with_capacity(bits.len() * 2);
        let byte_count = self.tree[self.root_index].count;

        let mut bits_iter = bits.iter();

        for _ in 0..byte_count {
            let mut index = self.root_index;

            while self.tree[index].left.is_some() || self.tree[index].right.is_some() {
                let bit = bits_iter.next().expect("missing data");
                if bit {
                    index = self.tree[index].left.expect("should have left index");
                } else {
                    index = self.tree[index].right.expect("should have right index")
                }
            }

            decompressed.push(index as u8);
        }

        decompressed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn compress_decompress() {
        let payload = b"so much words wow many compression";

        let huffman = Huffman::new_from_data(payload);
        let compressed = huffman.compress(payload);
        let decompressed = huffman.decompress(&compressed);

        assert!(compressed.len() < payload.len());
        assert_eq!(&payload[..], decompressed)
    }

    #[test]
    fn compress_decompress_lorem_ipsum() {
        let payload = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

        let huffman = Huffman::new_from_data(payload);
        let compressed = huffman.compress(payload);
        let decompressed = huffman.decompress(&compressed);

        assert!(compressed.len() < payload.len());
        assert_eq!(&payload[..], decompressed)
    }

    #[test]
    fn create_freq_table() {
        let mut table = [0; MAX_SYMBOLS];
        Huffman::calculate_freq_table(&mut table, &[0u8]);

        assert_eq!(table[0], 1);
    }

    #[test]
    fn payload_size_0() {
        let payload = &[];

        let huffman = Huffman::new_from_data(payload);
        let compressed = huffman.compress(payload);
        let decompressed = huffman.decompress(&compressed);

        assert_eq!(&payload[..], decompressed)
    }

    #[test]
    fn payload_size_1() {
        let payload = &[0u8];

        let huffman = Huffman::new_from_data(payload);
        let compressed = huffman.compress(payload);
        let decompressed = huffman.decompress(&compressed);

        assert_eq!(&payload[..], decompressed)
    }

    proptest! {
        #[test]
        fn proptest_compress_decompress(data: Vec<u8>) {
            let huffman = Huffman::new_from_data(&data);
            let compressed = huffman.compress(&data);
            let decompressed = huffman.decompress(&compressed);

            prop_assert!(compressed.len() <= data.len());
            prop_assert_eq!(data, decompressed);
        }

        #[test]
        fn proptest_freq_table(data: Vec<u8>) {
            let mut table = [0; MAX_SYMBOLS];
            Huffman::calculate_freq_table(&mut table, &data);

            for b in data {
                prop_assert!(table.get(b as usize).is_some());
                prop_assert!(table[b as usize] > 0);
            }
        }
    }
}
