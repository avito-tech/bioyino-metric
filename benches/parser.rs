#![allow(unused)]
use std::collections::hash_map::Entry;
use std::collections::HashMap;

use bioyino_metric::metric::Metric;
use bioyino_metric::name::MetricName;
use bioyino_metric::new_parser::MetricParser as NewMetricParser;
use bioyino_metric::parser::{DummyParseErrorHandler, MetricParser, MetricParsingError, ParseErrorHandler};

use bytes::BytesMut;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::random;
use std::time::Instant;

struct PanicErrorHandler;
impl ParseErrorHandler for PanicErrorHandler {
    fn handle(&self, buf: &[u8], pos: usize, e: MetricParsingError) {
        panic!("parse error {:?}", e);
    }
}

fn create_samples(size: usize) -> BytesMut {
    let mut types = Vec::new();
    types.push(Vec::from(&b"g"[..]));
    types.push(Vec::from(&b"c"[..]));
    types.push(Vec::from(&b"ms"[..]));
    let mut names = Vec::new();
    names.push(Vec::from(&b"lorem"[..]));
    names.push(Vec::from(&b"ipsum"[..]));
    names.push(Vec::from(&b"dolor"[..]));
    names.push(Vec::from(&b"sit"[..]));
    names.push(Vec::from(&b"amet"[..]));
    names.push(Vec::from(&b"consectetur"[..]));
    names.push(Vec::from(&b"adipiscing"[..]));
    names.push(Vec::from(&b"elit"[..]));
    names.push(Vec::from(&b"aenean"[..]));
    names.push(Vec::from(&b"laoreet"[..]));
    names.push(Vec::from(&b"porta"[..]));
    names.push(Vec::from(&b"mauris"[..]));
    names.push(Vec::from(&b"vestibulum"[..]));
    names.push(Vec::from(&b"mollis"[..]));
    names.push(Vec::from(&b"est"[..]));
    names.push(Vec::from(&b"rhoncus"[..]));
    names.push(Vec::from(&b"at"[..]));
    let name_max = 7usize;
    let mut data = BytesMut::new();
    for _ in 0..size {
        let name_size = random::<usize>() % name_max;
        let mut i = 0;
        let mut type_pos = 10000;
        loop {
            let word_pos = random::<usize>() % names.len();
            if i == 0 {
                // always take type same as first word, so same name metrics have same type
                type_pos = word_pos % types.len();
            }
            data.extend_from_slice(&names[word_pos]);
            if i >= name_size {
                break;
            }
            i += 1;
            data.extend_from_slice(&b"."[..]);
        }
        data.extend_from_slice(&b":"[..]);
        let value = format!("{}", random::<f64>());
        data.extend_from_slice(value.as_bytes());
        data.extend_from_slice(&b"|"[..]);
        data.extend_from_slice(&types[type_pos]);
        data.extend_from_slice(&b"\n"[..]);
    }
    data
}

pub type Float = f64;
pub type Cache = HashMap<MetricName, Metric<Float>>;

fn update_metric(cache: &mut Cache, name: MetricName, metric: Metric<Float>) {
    let ename = name.clone();
    match cache.entry(name) {
        Entry::Occupied(ref mut entry) => {
            entry.get_mut().accumulate(metric).unwrap_or_else(|_| {
                panic!("WTF");
            });
        }
        Entry::Vacant(entry) => {
            entry.insert(metric);
        }
    };
}

fn parse_input(input: &mut BytesMut) {
    let mut parser = MetricParser::<f64, PanicErrorHandler>::new(input, 1000, 1000, PanicErrorHandler);

    let mut cache: Cache = HashMap::with_capacity(1000);
    //let names_arena = BytesMut::with_capacity(1000);

    for (name, metric) in parser {
        // TODO: when tagged metrics come
        //if name.has_tags() {
        //names_arena.extend_from_slice(name.name_without_tags());
        //let untagged = MetricName::new_untagged(names_arena.split());
        //update_metric(&mut self.short, untagged, metric.clone());
        //}
        update_metric(&mut cache, name, metric);
    }
    if false {
        println!("{}", cache.len())
    }
}

fn parse_input_new(input: &mut BytesMut) {
    let mut parser = NewMetricParser::<f64, PanicErrorHandler>::new(input, 1000, 1000, PanicErrorHandler);
    let mut cache: Cache = HashMap::with_capacity(1000);
    //let names_arena = BytesMut::with_capacity(1000);

    for (name, metric) in parser {
        // TODO: when tagged metrics come
        //if name.has_tags() {
        //names_arena.extend_from_slice(name.name_without_tags());
        //let untagged = MetricName::new_untagged(names_arena.split());
        //update_metric(&mut self.short, untagged, metric.clone());
        //}
        update_metric(&mut cache, name, metric);
    }
    if false {
        println!("{}", cache.len())
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    //let mut samples1k = create_samples(1000);

    for i in 5..10 {
        let size = i * 1000;
        let mut samples = create_samples(size);

        dbg!(String::from_utf8_lossy(&samples[..99]));
        dbg!(samples.len());

        let mut group = c.benchmark_group(format!("{}", size));
        group.bench_function("parse_valid_untagged_new", |b| {
            b.iter_batched(
                || BytesMut::from(&samples[..]),
                |mut samples| {
                    parse_input_new(&mut samples);
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("parse_valid_untagged", |b| {
            b.iter_batched(
                || BytesMut::from(&samples[..]),
                |mut samples| {
                    parse_input(&mut samples);
                },
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
