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

use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap},
    rc::Rc,
};

use bit_vec::BitVec;

// Created with help from:
// - https://en.wikipedia.org/wiki/Huffman_coding
// - https://aquarchitect.github.io/swift-algorithm-club/Huffman%20Coding/

#[derive(Debug, Clone, Copy)]
struct Node {
    pub data: Option<u8>,
    pub count: usize,
    pub index: Option<usize>,
    pub parent: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl Node {
    fn new(data: u8, count: usize) -> Self {
        Self {
            data: Some(data),
            count,
            index: None,
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
    tree: Vec<Node>,
    // index lookup table for the leaf nodes.
    indexes: HashMap<u8, usize>,
}

impl Huffman {
    /// Initializes the huffman interface using the provided frequency table.
    pub fn new(frequency_table: &HashMap<u8, usize>) -> Self {
        let tree = Self::build_tree(frequency_table);
        let indexes = tree
            .iter()
            .filter(|x| x.data.is_some())
            .map(|n| (n.data.unwrap(), n.index.expect("should have index")))
            .collect();

        Self { tree, indexes }
    }

    /// Creates the Huffman frequency table from the provided data and initializes from it.
    pub fn new_from_data(data: &[u8]) -> Self {
        Self::new(&Self::calculate_freq_table(data))
    }

    /// Calculates the frequency table from the provided data.
    pub fn calculate_freq_table(data: &[u8]) -> HashMap<u8, usize> {
        let mut table: HashMap<u8, usize> = HashMap::with_capacity(256.min(data.len() / 2));

        for i in data {
            if let Some(c) = table.get_mut(i) {
                *c += 1;
            } else {
                table.insert(*i, 1);
            }
        }

        table
    }

    /// Builds a binary tree, the root is the last node, the leafs are at the start.
    fn build_tree(table: &HashMap<u8, usize>) -> Vec<Node> {
        let mut priority_queue: BinaryHeap<Rc<RefCell<Node>>> = table
            .iter()
            .map(|(c, v)| Rc::new(RefCell::new(Node::new(*c, *v))))
            .collect();

        let mut tree: Vec<Rc<RefCell<Node>>> = Vec::with_capacity(priority_queue.len() * 2);

        // Handle case where the frequency table has only 1 value.
        if priority_queue.len() == 1 {
            let shared_node = priority_queue.pop().unwrap();
            let mut node = shared_node.borrow_mut();
            node.index = Some(tree.len());

            tree.push(shared_node.clone());

            let parent = Node {
                data: None,
                count: node.count,
                left: node.index,
                right: None,
                parent: None,
                index: Some(tree.len()),
            };

            node.parent = parent.index;

            let parent = Rc::new(RefCell::new(parent));
            tree.push(parent);
        }

        while priority_queue.len() > 1 {
            let shared_node1 = priority_queue.pop().unwrap();
            let shared_node2 = priority_queue.pop().unwrap();

            let mut node1 = shared_node1.borrow_mut();
            if node1.index.is_none() {
                node1.index = Some(tree.len());
                tree.push(shared_node1.clone());
            }

            let mut node2 = shared_node2.borrow_mut();
            if node2.index.is_none() {
                node2.index = Some(tree.len());
                tree.push(shared_node2.clone());
            }

            let parent_index = tree.len();

            node1.parent = Some(parent_index);
            node2.parent = Some(parent_index);

            let parent = Node {
                data: None,
                count: node1.count + node2.count,
                left: node1.index,
                right: node2.index,
                parent: None,
                index: Some(parent_index),
            };

            let parent = Rc::new(RefCell::new(parent));
            tree.push(parent.clone());
            priority_queue.push(parent);
        }

        tree.into_iter().map(|x| *x.borrow()).collect()
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
            self.traverse(
                &mut bits,
                *self.indexes.get(b).unwrap_or_else(|| {
                    panic!("frequency table did not contain this byte: {:?}", b)
                }),
                None,
            )
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
        let root_index = self.tree.len() - 1;
        let byte_count = self.tree[root_index].count;

        let mut bits_iter = bits.iter();

        for _ in 0..byte_count {
            let mut index = root_index;

            while self.tree[index].left.is_some() || self.tree[index].right.is_some() {
                let bit = bits_iter.next().expect("missing data");
                if bit {
                    index = self.tree[index].left.expect("should have left index");
                } else {
                    index = self.tree[index].right.expect("should have right index")
                }
            }

            decompressed.push(self.tree[index].data.expect("should have data"));
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
        let table = Huffman::calculate_freq_table(&[0]);

        assert_eq!(table.len(), 1);
        assert_eq!(*table.get(&0).unwrap(), 1);

        let table = Huffman::calculate_freq_table(&[0, 1, 2, 2, 3, 3, 3]);

        assert_eq!(table.len(), 4);
        assert_eq!(*table.get(&0).unwrap(), 1);
        assert_eq!(*table.get(&1).unwrap(), 1);
        assert_eq!(*table.get(&2).unwrap(), 2);
        assert_eq!(*table.get(&3).unwrap(), 3);
    }

    #[test]
    fn test_payload_size_1() {
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
            let table = Huffman::calculate_freq_table(&data);

            for b in &data {
                prop_assert!(table.get(b).is_some());
            }
        }
    }
}
