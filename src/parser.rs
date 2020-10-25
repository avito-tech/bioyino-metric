use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::from_utf8;
use std::str::FromStr;

use combine::error::{ParseError, StreamError};
use combine::parser::byte::{byte, bytes as parse_bytes, digit, newline};
use combine::parser::range::{recognize, take, take_until_range, take_while1};
use combine::stream::easy;
use combine::stream::{decode, PointerOffset, RangeStream, StreamErrorFor};
use combine::{choice, position};
use combine::{eof, skip_many};
use combine::{optional, skip_many1, Parser};

use bytes::{Buf, BytesMut};
use lexical_core::{parse as parse_number, FromLexical};
use num_traits::{AsPrimitive, Float};

use crate::metric::{FromF64, Metric, MetricType};
use crate::name::{sort_tags, MetricName, TagFormat};

#[derive(Debug)]
pub enum ParsedPart<F>
where
    F: Float + FromStr + Debug + AsPrimitive<f64>,
{
    Metric((PointerOffset<[u8]>, PointerOffset<[u8]>), Option<PointerOffset<[u8]>>, Metric<F>),
    Trash(PointerOffset<[u8]>),
    TotalTrash(PointerOffset<[u8]>),
}

// The current goal is to be fast, use less allocs and to not depend on error type. that's why
// the signature may seem to be cryptic.
/// Parse stream of multiple metrics in statsd format. Usage of MetricParser is recommended instead.
pub fn metric_stream_parser<'a, I, F>(max_unparsed: usize, max_tags_len: usize) -> impl Parser<I, Output = ParsedPart<F>, PartialState = impl Default + 'a>
where
    I: 'a + combine::StreamOnce<Token = u8, Range = &'a [u8], Position = PointerOffset<[u8]>> + std::fmt::Debug + RangeStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    F: 'a + Float + Debug + FromStr + AsPrimitive<f64> + FromF64 + FromLexical + Sync,
    <F as FromStr>::Err: std::error::Error + Sync + Send + 'static,
{
    // empty comments help rustfmt with formatting

    // to avoid allocations, we only find a position of a name in buffer
    let name_with_tags = (
        //
        position(),
        take_while1::<I, _>(|c: u8| c != b';' && c != b':' && c != b'\n'),
        optional((byte(b';'), position(), take_while1::<I, _>(|c: u8| c != b':' && c != b'\n'))),
        position(),
    )
        .skip(byte(b':'))
        .and_then(move |(start, name, maybe_tag, stop)| {
            //TODO: introduce metric name limits, use is.alphabetical for each unicode char
            from_utf8(name).map_err(|_e| StreamErrorFor::<I>::unexpected_static_message("name part is not valid utf8"))?;
            let tag_pos = if let Some((_, tag_pos, tag)) = maybe_tag {
                if tag.len() > max_tags_len {
                    return Err(StreamErrorFor::<I>::unexpected_static_message("tag part is too long"));
                }
                from_utf8(tag).map_err(|_e| StreamErrorFor::<I>::unexpected_static_message("tag part is not valid utf8"))?;
                Some(tag_pos)
            } else {
                None
            };
            Ok::<_, StreamErrorFor<I>>((start, tag_pos, stop))
        });

    let sign = byte(b'+').map(|_| 1i8).or(byte(b'-').map(|_| -1i8));

    // This should parse metric value and separator
    let val = take_while1(|c: u8| c != b'|' && c != b'\n')
        //
        .skip(byte(b'|'))
        .and_then(|value| parse_number::<F>(value).map_err(|_e| StreamErrorFor::<I>::unexpected_static_message("value is not a valid number")));

    // This parses metric type
    let mtype = parse_bytes(b"ms")
        //
        .map(|_| MetricType::Timer(Vec::<F>::new()))
        .or(byte(b'g').map(|_| MetricType::Gauge(None)))
        .or(byte(b'C').map(|_| MetricType::DiffCounter(F::zero())))
        .or(byte(b'c').map(|_| MetricType::Counter))
        .or(byte(b's').map(|_| MetricType::Set(HashSet::new())));

    let unsigned_float = skip_many1(digit())
        //
        .and(optional((byte(b'.'), skip_many1(digit()))))
        .and(optional(
            //
            (byte(b'e'), optional(byte(b'+').or(byte(b'-'))), skip_many1(digit())),
        ));

    let sampling = (parse_bytes(b"|@"), recognize(unsigned_float))
        .and_then(|(_, val)| parse_number::<F>(val).map_err(|_e| StreamErrorFor::<I>::unexpected_static_message("sampling value is not a valid number")));

    let metric = (
        optional(sign),
        val,
        mtype,
        choice((
            sampling.map(Some),
            skip_many(newline()).map(|_| None),
            eof().map(|_| None),
            //skip_many(newline()).map(|_| None),
        )),
    )
        .map(|(sign, mut val, mtype, sampling)| {
            let mtype = if let MetricType::Gauge(_) = mtype {
                MetricType::Gauge(sign)
            } else {
                if sign == Some(-1) {
                    // get negative values back
                    val = -val
                }
                mtype
            };

            Metric::new(val, mtype, None, sampling).unwrap()
        });

    // here's what we are trying to parse
    choice((
        // valid metric with (probably) tags
        (skip_many(newline()), name_with_tags, metric, skip_many(newline())).map(|(_, (start, tag, stop), m, _)| ParsedPart::Metric((start, stop), tag, m)),
        (take_until_range(&b"\n"[..]), skip_many(newline()), position()).map(|(_, _, pos)| ParsedPart::Trash(pos)),
        // trash not ending with \n, but too long to be metric
        (take(max_unparsed), skip_many(newline()), position()).map(|(_, _, pos)| ParsedPart::TotalTrash(pos)),
    ))
}

pub type MetricParsingError<'a> = easy::Errors<u8, &'a [u8], PointerOffset<[u8]>>;

#[allow(unused_variables)]
/// Used to handle parsing errors
pub trait ParseErrorHandler {
    fn handle(&self, buf: &[u8], pos: usize, e: MetricParsingError) {}
}

/// Does nothing about error, can be used for ignoring all errors
pub struct DummyParseErrorHandler;
impl ParseErrorHandler for DummyParseErrorHandler {}

/// A high level parser to parse metric and split names from BytesMut.
/// Follows an iterator pattern, which fires metrics until it is possible,
/// modifying the buffer on the fly
pub struct MetricParser<'a, F, E: ParseErrorHandler> {
    input: &'a mut BytesMut,
    skip: usize,
    max_unparsed: usize,
    max_tags_len: usize,
    handler: E,
    sort_buf: Vec<u8>,
    _pd: PhantomData<F>,
}

impl<'a, F, E> MetricParser<'a, F, E>
where
    E: ParseErrorHandler,
{
    pub fn new(input: &'a mut BytesMut, max_unparsed: usize, max_tags_len: usize, handler: E) -> Self {
        let sort_buf = vec![0; max_unparsed];
        Self {
            input,
            skip: 0,
            max_unparsed,
            max_tags_len,
            handler,
            sort_buf,
            _pd: PhantomData,
        }
    }
}

impl<'a, F, E> Iterator for MetricParser<'a, F, E>
where
    E: ParseErrorHandler,
    F: Float + FromStr + AsPrimitive<f64> + FromF64 + Debug + FromLexical + Sync,
    <F as FromStr>::Err: std::error::Error + Sync + Send + 'static,
{
    type Item = (MetricName, Metric<F>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.skip >= self.input.len() {
                return None;
            }

            let res = {
                let input = &self.input[self.skip..];

                let parser = metric_stream_parser(self.max_unparsed, self.max_tags_len);
                //            let res = decode(
                //parser,
                //combine::stream::PartialStream(input),
                //&mut Default::default(),
                //            );
                decode(parser, &mut combine::easy::Stream(input), &mut Default::default())
            };

            match res {
                Ok((None, _)) => {
                    // there was not enough data for parser to finish
                    // end the iteration to get the buffer filled
                    return None;
                }
                Ok((Some(ParsedPart::Metric(name_pos, tag_pos, metric)), consumed)) => {
                    // at this point our input buffer looks like this
                    // [bad_data][useless_data][name][metric]
                    // we consider a metric name WITH tags as a "name" here
                    // self.skip point to end of bad_data
                    // `consumed` contains length ot all valid data (from useless to metric end)
                    // name_pos.0 points at the start of name
                    // name_pos.1 points at the end of name

                    // translate_position requires the pointers on original input to match
                    // so before changing anything, we want to count name position
                    let input = &self.input[self.skip..];
                    let start = name_pos.0.translate_position(input);
                    let stop = name_pos.1.translate_position(input);

                    // tag_pos is counted relative to input buffer
                    // but we need it to be relative to name
                    // to be related correctly, we have to shift it to `start` bytes right
                    let tag_pos = tag_pos.map(|pos| pos.translate_position(input) - start - 1);

                    // before touching the buffer calculate position to advance after name
                    let metriclen = consumed - stop;

                    // there can be errors found before correct parsing
                    // so we cut it off
                    if self.skip > 0 {
                        self.handler.handle(self.input, self.skip, easy::Errors::empty(name_pos.0));
                        self.input.advance(self.skip);
                    }

                    // then we cut everything until name start considering it a useless crap
                    // (usually newlines)
                    self.input.advance(start);

                    // now we can cut the name itself
                    let mut name = self.input.split_to(stop - start);

                    self.input.advance(metriclen);

                    self.skip = 0;

                    if let Some(pos) = tag_pos {
                        // with tag_pos found we need to try to sort tags
                        //
                        // since the buffer is created by ourselves, we are responsible for it's size, so
                        // it's WAY better to panic here if buffer size is incorrect
                        sort_tags(&mut name[..], TagFormat::Graphite, &mut self.sort_buf, pos).unwrap();
                    }

                    return Some((MetricName::from_raw_parts(name.freeze(), tag_pos), metric));
                }
                Ok((Some(ParsedPart::Trash(pos)), consumed)) => {
                    // trash matched
                    // skip it and continue, because
                    // it can still be followed by metric
                    if consumed == 0 {
                        // the very first byte or byte sequence was error, try shifting
                        // buffer to one char so we can move forward
                        self.skip += 1;
                    } else {
                        self.skip += consumed;
                    }
                    // TODO error information
                    self.handler.handle(self.input, self.skip, easy::Errors::empty(pos));
                }
                Ok((Some(ParsedPart::TotalTrash(pos)), consumed)) => {
                    // buffer is at max allowed length, but still no metrics there
                    // break cutting buffer to length specified
                    // this is the same action as trash, but a separate error message
                    // could be useful to signal a DDoS attempt or to distinguish
                    // some badly parsed metric(because of \n) from totally ununderstood
                    // bytes

                    self.skip = 0;

                    // buffer can be more than max_unparsed, so we cut and continue
                    // TODO error information
                    self.handler.handle(self.input, consumed, easy::Errors::empty(pos));
                    self.input.advance(consumed);
                }
                Err(e) => {
                    // error happens when no parsers match yet, i.e. some trash starts to come,
                    // yet didn't reach the max_unparsed, but we already know it
                    // cannot be metric
                    // in that case we can try to skip all the bytes until those where error
                    // happened

                    let input = &self.input[self.skip..];
                    let skip = e.position.translate_position(input);
                    self.handler.handle(self.input, self.skip, e);
                    if skip == 0 {
                        // error found at the very first byte
                        // skip 1 byte and try again
                        //self.skip += 1;

                        // try to find \r to cun until there
                        match input.iter().position(|c| *c == 10u8) {
                            Some(pos) => self.input.advance(pos),
                            None => self.input.clear(),
                        }
                    //cut all the input otherwise
                    } else {
                        // cut buffer to position where error was found
                        self.skip = 0;
                        self.input.advance(skip);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bytes::Bytes;
    // TODO: Questioned cases:
    //  * negative counters
    //  * diff counters

    struct TestParseErrorHandler;
    impl ParseErrorHandler for TestParseErrorHandler {
        fn handle(&self, input: &[u8], _: usize, e: MetricParsingError) {
            println!(
                "parse error at {:?} in {:?}: {:?}",
                e.position.translate_position(input),
                String::from_utf8_lossy(input),
                e
            );
        }
    }

    fn make_parser(input: &mut BytesMut) -> MetricParser<f64, TestParseErrorHandler> {
        MetricParser::<f64, TestParseErrorHandler>::new(input, 100, 50, TestParseErrorHandler)
    }

    #[test]
    fn parse_metric_good_counter() {
        let mut data = BytesMut::from(&b"gorets:1|c|@1"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(1f64, MetricType::Counter, None, Some(1f64)).unwrap());

        assert_eq!(parser.next(), None);
    }

    #[test]
    fn parse_metric_good_counter_float() {
        let mut data = BytesMut::from(&b"gorets:12.65|c|@0.001"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(12.65f64, MetricType::Counter, None, Some(1e-3f64)).unwrap());

        assert_eq!(parser.next(), None);
    }

    #[test]
    fn parse_metric_with_newline() {
        let mut data = BytesMut::from(&b"complex.bioyino.test1:-1e10|g\n\ncomplex.bioyino.test10:-1e10|g\n\n\n"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"complex.bioyino.test1"[..]);
        assert_eq!(metric, Metric::<f64>::new(1e10f64, MetricType::Gauge(Some(-1)), None, None).unwrap());

        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"complex.bioyino.test10"[..]);
        assert_eq!(metric, Metric::<f64>::new(1e10f64, MetricType::Gauge(Some(-1)), None, None).unwrap());

        assert_eq!(parser.next(), None);
    }

    #[test]
    fn parse_metric_without_newline() {
        let mut data = BytesMut::from(&b"complex.bioyino.test1:-1e10|gcomplex.bioyino.test10:-1e10|g"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"complex.bioyino.test1"[..]);
        assert_eq!(metric, Metric::<f64>::new(1e10f64, MetricType::Gauge(Some(-1)), None, None).unwrap());

        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"complex.bioyino.test10"[..]);
        assert_eq!(metric, Metric::<f64>::new(1e10f64, MetricType::Gauge(Some(-1)), None, None).unwrap());

        assert_eq!(parser.next(), None);
    }

    #[test]
    fn parse_metric_without_newline_sampling() {
        let mut data = BytesMut::from(&b"gorets:+1000|g|@0.4e-3gorets:-1000|g|@0.5"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(1)), None, Some(0.0004)).unwrap());
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(-1)), None, Some(0.5)).unwrap());
        assert_eq!(parser.next(), None);
    }

    #[test]
    fn parse_metric_short() {
        let mut data = BytesMut::from(&b"gorets:1|c"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(1f64, MetricType::Counter, None, None).unwrap());
        assert_eq!(parser.next(), None);
    }

    #[test]
    fn parse_metric_many() {
        let mut data = BytesMut::from(&b"gorets:+1000|g\ngorets:-1000|g|@0.5"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(1)), None, None).unwrap());
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(-1)), None, Some(0.5)).unwrap());
        assert_eq!(parser.next(), None);
    }

    #[test]
    fn parse_metric_bad_utf8() {
        use bytes::BufMut;
        let mut data = BytesMut::from(&b"borets1"[..]);
        data.reserve(1000);
        data.put_u8(193u8);
        data.put_u8(129u8);
        data.put(&b":+1000|g\ngorets:-1000|g|@0.5"[..]);
        let mut parser = make_parser(&mut data);
        let r = parser.next();
        let (name, metric) = r.unwrap();

        // Only one metric should be parsed
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(-1)), None, Some(0.5)).unwrap());
    }

    #[test]
    fn parse_metric_with_tags() {
        let mut data = BytesMut::from(&b"gorets;a=b.j.k.l;c=d:+1000|g\ngorets:-1000|g|@0.5"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        // name is still full string, including tags
        assert_eq!(&name.name[..], &b"gorets;a=b.j.k.l;c=d"[..]);
        assert_eq!(name.tag_pos, Some(6usize));
        assert_eq!(&name.name[name.tag_pos.unwrap()..], &b";a=b.j.k.l;c=d"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(1)), None, None).unwrap());
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(-1)), None, Some(0.5)).unwrap());
    }

    #[test]
    fn parse_metric_with_long_tags() {
        let mut data = BytesMut::from(&b"gorets;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b:+1000|g"[..]);
        let mut parser = make_parser(&mut data);
        assert!(parser.next().is_none());
    }

    #[test]
    fn parse_many_metrics_with_long_tags() {
        // metric with long tags followed by another metric
        let mut data = BytesMut::from(&b"gorets;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b;a=b:+1000|g\n\nbobets;c=d:1000|g\n\n"[..]);
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"bobets;c=d"[..]);
        assert_eq!(name.tag_pos, Some(6usize));
        assert_eq!(&name.name[name.tag_pos.unwrap()..], &b";c=d"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(None), None, None).unwrap());
    }

    #[test]
    fn parse_trashed_metric_with_tags() {
        let mut data = BytesMut::new();
        data.extend_from_slice(b"trash\ngorets1:+1e3|g\nTRASH\n\n\ngorets2;tag3=sh.t;t2=fuck:-1e+3|g|@5e-1\nMORE;tra=sh;|TrasH\nFUUU");
        let mut parser = make_parser(&mut data);
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets1"[..]);
        assert_eq!(name.tag_pos, None);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(1)), None, None).unwrap());

        // parser must sort tags
        let (name, metric) = parser.next().unwrap();
        assert_eq!(&name.name[..], &b"gorets2;t2=fuck;tag3=sh.t"[..]);
        assert_eq!(name.tag_pos, Some(7usize));
        assert_eq!(&name.name[name.tag_pos.unwrap()..], &b";t2=fuck;tag3=sh.t"[..]);
        assert_eq!(metric, Metric::<f64>::new(1000f64, MetricType::Gauge(Some(-1)), None, Some(0.5)).unwrap());
    }

    #[test]
    fn parse_split_metric_buf() {
        let mut data = BytesMut::new();
        data.extend_from_slice(b"gorets1:+1001|g\nT\x01RAi:|\x01SH\nnuggets2:-1002|s|@0.5\nMORETrasH\nFUUU\n\ngorets3:+1003|ggorets4:+1004|ms:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ggorets5:1005|ms");

        let correct = vec![
            (
                Bytes::from("gorets1"),
                Metric::<f64>::new(1001f64, MetricType::Gauge(Some(1)), None, None).unwrap(),
            ),
            (
                Bytes::from("nuggets2"),
                Metric::<f64>::new(-1002f64, MetricType::Set(HashSet::new()), None, Some(0.5)).unwrap(),
            ),
            (
                Bytes::from("gorets3"),
                Metric::<f64>::new(1003f64, MetricType::Gauge(Some(1)), None, None).unwrap(),
            ),
            (
                Bytes::from("gorets4"),
                Metric::<f64>::new(1004f64, MetricType::Timer(Vec::new()), None, None).unwrap(),
            ),
            (
                Bytes::from("gorets5"),
                Metric::<f64>::new(1005f64, MetricType::Timer(Vec::new()), None, None).unwrap(),
            ),
        ];
        for i in 1..(data.len() + 1) {
            // this is out test case - partially received data
            let mut testinput = BytesMut::from(&data[0..i]);
            println!("TEST[{}] {:?}", i, String::from_utf8(Vec::from(&testinput[..])).unwrap());

            let mut res = Vec::new();
            // we use 20 as max_unparsed essentially to test total trash path
            let parser = MetricParser::<f64, TestParseErrorHandler>::new(&mut testinput, 20, 20, TestParseErrorHandler);
            for (name, metric) in parser {
                res.push((name, metric));
            }

            println!("RES: {:?}", res);
            // until 15th gorets1 is not there, no metrics should be parsed
            if i < 15 {
                assert!(res.len() == 0)
            }

            // 15 and later gorets1 should always be parsed
            if i >= 15 {
                assert!(res.len() > 0)
            }
            // between 15 and 43 ONLY gorets1 should be parsed
            if i >= 15 && i < 43 {
                assert!(res.len() == 1)
            }

            // on 43 wild nuggets2 appears without sampling spec...
            if i == 43 {
                assert!(res.len() == 2)
            }

            // .. and disappears because parser understands there are more chars to be parsed
            if i >= 44 && i < 46 {
                assert!(res.len() == 1)
            }

            // nuggets2:-1000|g|@0 is ok on 46
            if i == 46 {
                assert!(res.len() == 2)
            }

            // nuggets2:-1000|g|@0. is not ok on 47
            if i == 47 {
                assert!(res.len() == 1)
            }
            // 48 and forth both metrics must exist
            if i > 47 && i < 80 {
                assert!(res.len() == 2)
            }

            // after 79 there must be 3 metrics
            if i >= 80 && i < 96 {
                assert!(res.len() == 3)
            }

            // after 97 there must be 4 metrics
            if i >= 96 && i < 171 {
                assert!(res.len() == 4)
            }
            if i >= 171 {
                assert!(res.len() == 5)
            }

            for (n, (cname, cmet)) in correct.iter().enumerate() {
                if let Some((name, met)) = res.get(n) {
                    assert_eq!(cname, &name.name);
                    if n == 1 && i == 43 || i == 46 {
                        // nuggets is intentionally parsed in another way on some steps
                        continue;
                    }
                    assert_eq!(cmet, met);
                }
            }
        }
    }
}
