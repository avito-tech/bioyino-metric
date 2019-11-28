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
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
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
///
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

    // TODO example
    /// find position where tags start, optionally forcing re-search when it's already found
    /// Note, that found position points to the first semicolon, not the tags part itself
    pub fn find_tag_pos(&mut self, force: bool) -> bool {
        if force || self.tag_pos.is_none() {
            self.tag_pos = self.name.iter().position(|c| *c == b';');
        }
        self.tag_pos.is_some()
    }

    /// true if metric is tagged
    pub fn has_tags(&mut self) -> bool {
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

    /// puts a name with an aggregate to provided buffer depending on dest
    /// expects tag_pos to be already found
    pub fn put_with_aggregate<F>(
        // rustfmt
        &self,
        buf: &mut BytesMut,
        dest: AggregationDestination,
        agg: &Aggregate<F>,
        postfix_replacements: &HashMap<Aggregate<F>, String>,
        tag_replacements: &HashMap<Aggregate<F>, String>,
    ) -> Result<(), ()>
    where
        F: Float + Debug + FromF64 + AsPrimitive<usize>,
    {
        // for value aggregate ignore replacements and other shit
        if agg == &Aggregate::Value {
            let namelen = self.name.len();
            buf.reserve(namelen);
            buf.put_slice(&self.name);
            return Ok(());
        }

        // we should not use let agg_postfix before the match, because with the tag case we don't
        // need it
        // the same applies to tag name and value: we only need them when aggregating to tags

        match dest {
            AggregationDestination::Smart if self.tag_pos.is_none() => {
                let agg_postfix = postfix_replacements.get(agg).ok_or(())?.as_bytes();
                // metric is untagged, add aggregate to name
                Ok(self.put_with_suffix(buf, agg_postfix))
            }
            AggregationDestination::Smart => {
                let agg_tag_name = tag_replacements.get(&Aggregate::AggregateTag).ok_or(())?.as_bytes();
                let agg_tag_value = tag_replacements.get(agg).ok_or(())?.as_bytes();

                // metric is tagged, add aggregate as tag
                Ok(self.put_with_fixed_tag(buf, agg_tag_name, agg_tag_value, false))
            }
            AggregationDestination::Name => {
                let agg_postfix = postfix_replacements.get(agg).ok_or(())?.as_bytes();
                Ok(self.put_with_suffix(buf, agg_postfix))
            }
            AggregationDestination::Tag => {
                let agg_tag_name = tag_replacements.get(&Aggregate::AggregateTag).ok_or(())?.as_bytes();
                let agg_tag_value = tag_replacements.get(agg).ok_or(())?.as_bytes();

                Ok(self.put_with_fixed_tag(buf, agg_tag_name, agg_tag_value, false))
            }
            AggregationDestination::Both => {
                let agg_postfix = postfix_replacements.get(agg).ok_or(())?.as_bytes();
                let agg_tag_name = tag_replacements.get(&Aggregate::AggregateTag).ok_or(())?.as_bytes();
                let agg_tag_value = tag_replacements.get(agg).ok_or(())?.as_bytes();

                self.put_with_suffix(buf, agg_postfix);
                self.put_with_fixed_tag(buf, agg_tag_name, agg_tag_value, true);
                Ok(())
            }
        }
    }

    /// sort tags in place using intermediate buffer, buffer length MUST be at least
    /// `name.len() - tag_pos` bytes
    /// sorting is made lexicographically
    pub fn sort_tags<B: AsMut<[u8]>>(&mut self, mode: TagFormat, intermediate: &mut B) -> Result<(), ()> {
        if self.tag_pos.is_none() && !self.find_tag_pos(true) {
            // tag position was not found, so no tags
            // but it is ok since we have nothing to sort
            return Ok(());
        }

        let intermediate: &mut [u8] = intermediate.as_mut();
        if intermediate.len() < (self.name.len() - self.tag_pos.unwrap()) {
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
        let mut p_reps: HashMap<Aggregate<f64>, String> = HashMap::new();
        p_reps.insert(Aggregate::Count, "count".to_string());
        p_reps.insert(Aggregate::AggregateTag, "MUST NOT BE USED".to_string());
        p_reps.insert(Aggregate::Percentile(0.8f64), "percentile80".to_string());

        let mut t_reps: HashMap<Aggregate<f64>, String> = HashMap::new();
        t_reps.insert(Aggregate::Count, "cnt".to_string());
        t_reps.insert(Aggregate::AggregateTag, "agg".to_string());
        // intentionally skip adding percentile80

        let mut without_tags = MetricName::new(BytesMut::from(&b"gorets.bobez"[..]), None);
        let mut with_tags = MetricName::new(BytesMut::from(&b"gorets.bobez;tag=value"[..]), None);
        let mut with_semicolon = MetricName::new(BytesMut::from(&b"gorets.bobez;"[..]), None);
        without_tags.find_tag_pos(true);
        with_tags.find_tag_pos(true);
        with_semicolon.find_tag_pos(true);

        // create 0-size buffer to make sure allocation counts work as intended
        let mut buf = BytesMut::new();

        // --------- without_tags

        // max is not in replacements
        assert!(without_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Max, &p_reps, &t_reps).is_err(), "non existing replacement gave no error");

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Value, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez"[..]);

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count"[..]);

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Percentile(0.8f64), &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.percentile80"[..], "existing postfix replacement was not put");

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Name, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count"[..]);

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Name, &Aggregate::Percentile(0.8f64), &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.percentile80"[..], "existing postfix replacement was not put");

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Tag, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;agg=cnt"[..]);

        let err = without_tags.put_with_aggregate(&mut buf, AggregationDestination::Tag, &Aggregate::Percentile(0.8f64), &p_reps, &t_reps);
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        without_tags.put_with_aggregate(&mut buf, AggregationDestination::Both, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count;agg=cnt"[..]);

        let err = without_tags.put_with_aggregate(&mut buf, AggregationDestination::Both, &Aggregate::Percentile(0.8f64), &p_reps, &t_reps);
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        // --------- with_tags
        assert!(with_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Max, &p_reps, &t_reps).is_err());

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Value, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;tag=value"[..]);

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;tag=value;agg=cnt"[..]);

        let err = with_tags.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Percentile(0.8f64), &p_reps, &t_reps);
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Name, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count;tag=value"[..]);

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Tag, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;tag=value;agg=cnt"[..]);

        let err = with_tags.put_with_aggregate(&mut buf, AggregationDestination::Tag, &Aggregate::Percentile(0.8f64), &p_reps, &t_reps);
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        with_tags.put_with_aggregate(&mut buf, AggregationDestination::Both, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.count;tag=value;agg=cnt"[..]);

        let err = with_tags.put_with_aggregate(&mut buf, AggregationDestination::Both, &Aggregate::Percentile(0.8f64), &p_reps, &t_reps);
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        // ensure trailing semicolon is not duplicated
        with_semicolon.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Value, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;"[..]);

        with_semicolon.put_with_aggregate(&mut buf, AggregationDestination::Smart, &Aggregate::Count, &p_reps, &t_reps).unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;agg=cnt"[..]);
    }

    #[test]
    fn metric_name_sort_tags_graphite() {
        let mut intermediate: Vec<u8> = Vec::new();
        let mut name = MetricName::new(BytesMut::from(&b"gorets2;tag3=shit;t2=fuck"[..]), None);
        name.find_tag_pos(false);
        if intermediate.len() < name.name.len() {
            intermediate.resize(name.name.len(), 0u8);
        }
        assert!(name.sort_tags(TagFormat::Graphite, &mut intermediate).is_ok());

        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez;t=y;a=b;c=e;u=v;c=d;c=b;aaa=z"[..]), None);
        name.find_tag_pos(false);
        let tag_pos = name.tag_pos.unwrap();

        let tag_len = name.name.len() - tag_pos;
        assert_eq!(tag_len, 30);

        //let mut intermediate = Vec::with_capacity(tag_len);

        let mut intermediate = BytesMut::new();
        intermediate.resize(tag_len - 1, 0u8); // intentionally resize to less bytes than required
        assert!(name.sort_tags(TagFormat::Graphite, &mut intermediate).is_err());

        intermediate.put(0u8); // resize to good length now
        assert!(name.sort_tags(TagFormat::Graphite, &mut intermediate).is_ok());
        assert_eq!(&name.name[..], &b"gorets.bobez;a=b;aaa=z;c=b;c=d;c=e;t=y;u=v"[..]);
    }
}
