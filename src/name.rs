use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use bytes::BytesMut;

pub enum TagMode {
    Graphite,
}

/// Contains buffer containing the full metric name including tags
/// and some data to split tags from name useful for appending aggregates
#[derive(Debug, PartialEq, Eq)]
pub struct MetricName {
    pub name: BytesMut,
    pub tag_pos: Option<usize>,
    //pub tags: BTreeMap<BytesMut, BytesMut>, // we need btreemap to have tags sorted
}

impl MetricName {
    pub fn new(name: BytesMut, tag_pos: Option<usize>) -> Self {
        Self {
            name,
            tag_pos, /*tags: BTreeMap::new()*/
        }
    }

    pub fn find_tag_pos(&mut self, force: bool) -> bool {
        if force || self.tag_pos.is_none() {
            self.tag_pos = self.name.iter().position(|c| *c == b';')
        }
        self.tag_pos.is_some()
    }

    /// sort tags in place using intermediate buffer, buffer length MUST be at least
    /// `name.len() - tag_pos` bytes
    /// sorting is made lexicographically
    pub fn sort_tags<B: AsRef<[u8]>>(
        &mut self,
        mode: TagMode,
        intermediate: &mut B,
    ) -> Result<(), ()> {
        // TODO: think error type
        Err(())
    }

    //  allocate tags structure and shorten name fetching tags into BTreeMap
    //pub fn fetch_tags<B: BufMut>(&mut self, mode: TagMode) {
    //let tag_pos = match self.tag_pos {
    //None => return,
    //Some(t) => t,
    //};

    //// TODO: handle sorting
    //// TODO: handle repeating same tags i.e. gorets;a=b;e=b;a=b:...
    //match mode {
    //TagMode::Graphite => unimplemented!(),
    //}
    //}
}

impl Hash for MetricName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // metric with tag position found and tag position not found should be the same
        self.name.hash(state);
        //if self.tags.len() > 0 {
        //self.tags.hash(state);
        //}
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    use bytes::Bytes;

    #[test]
    fn metric_name_tag_position() {
        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez;a=b;c=d"[..]), Some(1)); //provide incorrect pos
        name.find_tag_pos(true);
        assert_eq!(name.tag_pos, Some(12));
    }

    #[test]
    fn metric_name_sort_tags_graphite() {
        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez;a=b;c=d"[..]), None);
        name.find_tag_pos(false);
        let _ = name.tag_pos.unwrap();

        //     let mut parser = make_parser(&mut data);
        //let (name, metric) = parser.next().unwrap();
        //// name is still full string, including tags
        //assert_eq!(&name.name[..], &b"gorets;a=b;c=d"[..]);
        //assert_eq!(name.tag_pos, Some(7usize));
        //assert_eq!(&name.name[name.tag_pos.unwrap()..], &b"a=b;c=d"[..]);
        //assert_eq!(
        //metric,
        //Metric::<f64>::new(1000f64, MetricType::Gauge(Some(1)), None, None).unwrap()
        //);
        //let (name, metric) = parser.next().unwrap();
        //assert_eq!(&name.name[..], &b"gorets"[..]);
        //assert_eq!(
        //metric,
        //Metric::<f64>::new(1000f64, MetricType::Gauge(Some(-1)), None, Some(0.5)).unwrap()
        //     );
    }
}
