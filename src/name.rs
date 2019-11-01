use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use bytes::{BufMut, BytesMut};
use num_traits::{AsPrimitive, Float};
use serde_derive::{Deserialize, Serialize};

use crate::aggregate::Aggregate;
use crate::metric::FromF64;

// TODO: Think error type. There is single possible error atm, so sort_tags returns () instead
// TODO: Think if we need sorted tags in btreemap instead of string
// TODO: Handle repeating same tags i.e. gorets;a=b;e=b;a=b:...
// TODO: Split MetricName type to two: RawMetricName and MetricName, where the former is readonly
// and guarantees the tag position was already searched for, so we can remove those "expects tag position is
// found" everywhere

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TagFormat {
    Graphite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationDestination {
    /// place depending on tags existence, if they do - always place in tag
    Smart,
    /// always place aggregate as postfix in name separated with dot
    Name,
    /// always place aggregate into tag, adding ones if they are not there yet
    Tag,
    /// always place aggregate in both tag and name postfix
    Both,
}

/// Contains buffer containing the full metric name including tags
/// and some data to split tags from name useful for appending aggregates
#[derive(Debug, Eq, Clone)]
pub struct MetricName {
    pub name: BytesMut,
    pub tag_pos: Option<usize>,
    //pub tags: BTreeMap<BytesMut, BytesMut>, // we need btreemap to have tags sorted
}

impl MetricName {
    pub fn new(name: BytesMut, tag_pos: Option<usize>) -> Self {
        Self { name, tag_pos /*tags: BTreeMap::new()*/ }
    }

    /// find position where tags start, forcing re-search when it's already found
    pub fn find_tag_pos(&mut self, force: bool) -> bool {
        if force || self.tag_pos.is_none() {
            self.tag_pos = self.name.iter().position(|c| *c == b';')
        }
        self.tag_pos.is_some()
    }

    /// returns only name, without tags, considers tag position was already found before
    pub fn name_without_tags(&self) -> &[u8] {
        if let Some(pos) = self.tag_pos {
            &self.name[..pos]
        } else {
            &self.name[..]
        }
    }

    /// returns slice with full name, including tags
    pub fn name_with_tags(&self) -> &[u8] {
        &self.name[..]
    }

    /// returns slice with only tags, includes leading semicolon, expects tag position was already
    /// found
    pub fn tags_without_name(&self) -> &[u8] {
        if let Some(pos) = self.tag_pos {
            &self.name[pos..]
        } else {
            &[]
        }
    }

    /// returns length of tags field, including leading semicolon
    /// considers tag position was already found before
    pub fn tags_len(&self) -> usize {
        if let Some(pos) = self.tag_pos {
            self.name.len() - pos
        } else {
            0
        }
    }

    /// put name into buffer with suffix added with dot after name
    fn put_with_suffix(&self, buf: &mut BytesMut, suffix: &[u8]) {
        let suflen = suffix.len();
        let namelen = self.name.len();

        buf.reserve(namelen + suflen + 1);
        match self.tag_pos {
            None => {
                buf.put_slice(&self.name);
                buf.put(b'.');
                buf.put_slice(suffix);
            }
            Some(pos) => {
                buf.put_slice(&self.name[..pos]);
                buf.put(b'.');
                buf.put_slice(suffix);
                buf.put_slice(&self.name[pos..]);
            }
        }
    }

    fn put_with_fixed_tag(&self, buf: &mut BytesMut, tag_name: &[u8], tag: &[u8], only_tag: bool) {
        let namelen = self.name.len();
        let semicolon = self.name[namelen - 1] == b';';
        let mut addlen = tag_name.len() + tag.len() + 1; // 1 is for `=`
        if !only_tag {
            addlen += namelen
        }
        if !semicolon {
            addlen += 1 // add one more for `;`
        };
        buf.reserve(addlen);
        if !only_tag {
            buf.put_slice(&self.name);
        }
        if !semicolon {
            buf.put(b';');
        }
        buf.put_slice(tag_name);
        buf.put(b'=');
        buf.put_slice(tag);
    }

    //fn put_with_value_tag(&self, buf: &mut BytesMut, tag: &[u8], value: f64) {
    //    let namelen = self.name.len();

    //FIXME
    /*
    let mut addlen = namelen + agglen + 1 + 32; // 1 is for `=`, 32 is for the float value itself
    if !semicolon {
        addlen += 1 // add one more for `;`
    };
    buf.reserve(addlen);
    buf.put_slice(&self.name);
    if !semicolon {
        buf.put(b';');
    }
    buf.put_slice(agg_name);
    buf.put(b'=');
    buf.put_slice(&self.name[pos..]);
    */
    //}

    /// puts a name with an aggregate to provided buffer depending on dest
    /// expects tag_pos to be already found
    pub fn put_with_aggregate<F>(&self, buf: &mut BytesMut, dest: AggregationDestination, agg: Aggregate<F>, replacements: &HashMap<Aggregate<F>, String>) -> Result<(), ()>
    where
        F: Float + Debug + FromF64 + AsPrimitive<usize>,
    {
        let agg_name = replacements.get(&agg).ok_or(())?.as_bytes();
        let agg_tag = replacements.get(&Aggregate::AggregateTag).ok_or(())?.as_bytes();

        match dest {
            AggregationDestination::Smart if self.tag_pos.is_none() => {
                // metric is untagged, add aggregate to name
                Ok(self.put_with_suffix(buf, agg_name))
            }
            AggregationDestination::Smart => {
                // metric is tagged, add aggregate as tag
                Ok(self.put_with_fixed_tag(buf, agg_tag, agg_name, false))
            }
            AggregationDestination::Name => Ok(self.put_with_suffix(buf, agg_name)),
            AggregationDestination::Tag => Ok(self.put_with_fixed_tag(buf, agg_tag, agg_name, false)),
            AggregationDestination::Both => {
                self.put_with_suffix(buf, agg_name);
                self.put_with_fixed_tag(buf, agg_tag, agg_name, true);
                Ok(())
            }
        }
    }

    /// sort tags in place using intermediate buffer, buffer length MUST be at least
    /// `name.len() - tag_pos` bytes
    /// sorting is made lexicographically
    pub fn sort_tags<B: AsMut<[u8]>>(&mut self, mode: TagFormat, intermediate: &mut B) -> Result<(), ()> {
        if self.tag_pos.is_none() && !self.find_tag_pos(true) {
            return Err(());
        }

        let intermediate: &mut [u8] = intermediate.as_mut();
        if intermediate.len() < self.name.len() - self.tag_pos.unwrap() {
            return Err(());
        }
        use lazysort::Sorted;
        match mode {
            TagFormat::Graphite => {
                let mut offset = 0;
                for part in self.name.split(|c| *c == b';').skip(1).sorted() {
                    let end = offset + part.len();
                    intermediate[offset..end].copy_from_slice(part);
                    if end < intermediate.len() - 1 {
                        intermediate[end] = b';';
                    }
                    offset = end + 1;
                }
                offset -= 1;
                let tp = self.tag_pos.unwrap() + 1;
                self.name[tp..].copy_from_slice(&intermediate[..offset]);
            }
        }
        Ok(())
    }

    //  allocate tags structure and shorten name fetching tags into BTreeMap
    //pub fn fetch_tags<B: BufMut>(&mut self, mode: TagFormat) {
    //let tag_pos = match self.tag_pos {
    //None => return,
    //Some(t) => t,
    //};

    //match mode {
    //TagFormat::Graphite => unimplemented!(),
    //}
    //}
}

impl PartialEq for MetricName {
    fn eq(&self, other: &Self) -> bool {
        // metric with tag position found and tag position not found should be the same
        self.name == other.name
    }
}

impl Hash for MetricName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bytes::BufMut;

    #[test]
    fn metric_name_tag_position() {
        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez;a=b;c=d"[..]), Some(1));
        name.find_tag_pos(true);
        assert_eq!(name.tag_pos, Some(12));
    }

    #[test]
    fn metric_name_tags_len() {
        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez;a=b;c=d"[..]), None);
        name.find_tag_pos(true);
        assert_eq!(name.tags_len(), 8);

        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobezzz"[..]), None);
        name.find_tag_pos(true);
        assert_eq!(name.tags_len(), 0);
    }

    #[test]
    fn metric_name_splits() {
        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez"[..]), None);
        name.find_tag_pos(true);
        assert_eq!(name.name_without_tags(), &b"gorets.bobez"[..]);
        assert_eq!(name.tags_without_name().len(), 0);

        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez;a=b;c=d"[..]), None);
        name.find_tag_pos(true);
        assert_eq!(name.name_without_tags(), &b"gorets.bobez"[..]);
        assert_eq!(name.tags_without_name(), &b";a=b;c=d"[..]);
    }

    #[test]
    fn metric_aggregate_modes() {
        // create some replacements
        let mut replacements: HashMap<Aggregate<f64>, String> = HashMap::new();
        replacements.insert(Aggregate::Count, "count".to_string());
        replacements.insert(Aggregate::AggregateTag, "aggregate".to_string());

        let mut without_tags = MetricName::new(BytesMut::from(&b"gorets.bobez"[..]), None);
        let mut with_tags = MetricName::new(BytesMut::from(&b"gorets.bobez;tag=value"[..]), None);
        let mut with_semicolon = MetricName::new(BytesMut::from(&b"gorets.bobez;"[..]), None);
        without_tags.find_tag_pos(true);
        with_tags.find_tag_pos(true);
        with_semicolon.find_tag_pos(true);

        // create 0-size buffer to make sure allocation counts work as intended
        let mut buf = BytesMut::new();

        assert!(without_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, Aggregate::Max, &replacements).is_err());

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, Aggregate::Count, &replacements).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count"[..]);

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Name, Aggregate::Count, &replacements).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count"[..]);

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Tag, Aggregate::Count, &replacements).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;aggregate=count"[..]);

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Both, Aggregate::Count, &replacements).unwrap();
        dbg!(&buf);
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count;aggregate=count"[..]);

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, Aggregate::Count, &replacements).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;tag=value;aggregate=count"[..]);

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Name, Aggregate::Count, &replacements).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count;tag=value"[..]);

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Tag, Aggregate::Count, &replacements).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;tag=value;aggregate=count"[..]);

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Both, Aggregate::Count, &replacements).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count;tag=value;aggregate=count"[..]);

        // ensure trailing semicolon is not duplicated
        with_semicolon.put_with_aggregate(&mut buf, AggregationDestination::Smart, Aggregate::Count, &replacements).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;aggregate=count"[..]);
    }

    #[test]
    fn metric_name_sort_tags_graphite() {
        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez;t=y;a=b;c=e;u=v;c=d;c=b;aaa=z"[..]), None);
        name.find_tag_pos(false);
        let tag_pos = name.tag_pos.unwrap();

        let tag_len = name.name.len() - tag_pos;
        assert_eq!(tag_len, 30);

        //let mut intermediate = Vec::with_capacity(tag_len);
        let mut intermediate = BytesMut::new();
        intermediate.resize(tag_len - 1, 0u8); // intentionally resize to less bytes than required
        assert!(name.sort_tags(TagFormat::Graphite, &mut intermediate).is_err());

        intermediate.put(0u8);
        assert!(name.sort_tags(TagFormat::Graphite, &mut intermediate).is_ok());
        assert_eq!(&name.name[..], &b"gorets.bobez;a=b;aaa=z;c=b;c=d;c=e;t=y;u=v"[..]);
    }
}
