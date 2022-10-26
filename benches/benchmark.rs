use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustyman::Huffman;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("compress", |b| {
        let huffman = Huffman::new_from_data(include_bytes!("../assets/bench_input.txt"));
        let data = include_bytes!("../assets/bench_input.txt");
        b.iter_with_large_drop(|| huffman.compress(black_box(data)));
    });

    c.bench_function("decompress", |b| {
        let huffman = Huffman::new_from_data(include_bytes!("../assets/bench_input.txt"));
        let data = include_bytes!("../assets/bench_input.txt");
        let compressed = huffman.compress(data);

        b.iter_with_large_drop(|| huffman.decompress(black_box(&compressed)));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
