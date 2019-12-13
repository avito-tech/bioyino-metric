use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use bytes::{BufMut, Bytes, BytesMut};
use num_traits::{AsPrimitive, Float};
use serde_derive::{Deserialize, Serialize};

use crate::aggregate::Aggregate;
use crate::metric::FromF64;

// TODO: Think error type. There is single possible error atm, so sort_tags returns () instead
// TODO: Think if we need sorted tags in btreemap instead of string (at the moment of writing this we don't, because of allocation)
// TODO: Handle repeating same tags i.e. gorets;a=b;e=b;a=b:...

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

pub fn find_tag_pos(name: &[u8], mode: TagFormat) -> Option<usize> {
    match mode {
        TagFormat::Graphite => name.iter().position(|c| *c == b';'),
    }
}

/// Sorts tags inside name using intermediate buffer
pub(crate) fn sort_tags(name: &mut [u8], mode: TagFormat, intermediate: &mut [u8], tag_pos: usize) -> Result<usize, ()> {
    if intermediate.len() < (name.len() - tag_pos) {
        return Err(());
    }

    use lazysort::Sorted;
    match mode {
        TagFormat::Graphite => {
            // There is one special case: the metric without tags, but received as tagged
            // (i.e. with trailing semicolon)
            // we want to save this semicolon for metric to stay tagged and tag position to remain
            // correct
            if tag_pos == name.len() - 1 {
                return Ok(name.len());
            }

            let mut offset = 0; // meaningful length of data in intermediate buffer
            let mut cutlen = 0; // length to cut from name because of removed empty tags
            for part in name.split(|c| *c == b';').skip(1).sorted() {
                if part.is_empty() {
                    //   offset += 1;
                    cutlen += 1;
                } else {
                    let end = offset + part.len();
                    intermediate[offset..end].copy_from_slice(part);
                    if end < intermediate.len() - 1 {
                        intermediate[end] = b';';
                    }
                    offset = end + 1;
                }
            }

            let newlen = if intermediate[offset] == b';' {
                offset -= 1;
                name.len() - cutlen - 1
            } else {
                name.len() - cutlen
            };

            if offset > 0 {
                offset -= 1;
                name[tag_pos + 1..newlen].copy_from_slice(&intermediate[..offset]);
            }

            Ok(newlen)
        }
    }
}

/// Contains buffer containing the full metric name including tags
/// and some data to work with tags
#[derive(Debug, Eq, Clone)]
pub struct MetricName {
    pub name: Bytes,
    pub(crate) tag_pos: Option<usize>,
    //pub(crate) tag_format: TagFormat,  // TODO we mayn need this in future
    //pub tags: BTreeMap<BytesMut, BytesMut>, // we need btreemap to have tags sorted
}

impl MetricName {
    pub fn new<B: AsMut<[u8]>>(mut name: BytesMut, mode: TagFormat, intermediate: &mut B) -> Result<Self, ()> {
        let tag_pos = find_tag_pos(&name[..], mode);
        // sort tags in place using intermediate buffer, buffer length MUST be at least
        // `name.len() - tag_pos` bytes
        // sorting is made lexicographically
        match tag_pos {
            // tag position was not found, so no tags
            // but it is ok since we have nothing to sort
            None => return Ok(Self { name: name.freeze(), tag_pos }),
            Some(pos) => {
                let intermediate: &mut [u8] = intermediate.as_mut();
                let newlen = sort_tags(&mut name[..], mode, intermediate, pos)?;
                name.truncate(newlen);
            }
        };

        Ok(Self {
            name: name.freeze(),
            tag_pos, /*tag_format*/
        })
    }

    /// Convenience method to create metric that is for sure has no tags in any format
    pub fn new_untagged(name: BytesMut) -> Self {
        Self {
            name: name.freeze(),
            tag_pos: None,
        }
    }

    /// Assemble name from internal parts *without checks*. Tags must be sorted, tag position must
    /// be found according to mode
    pub fn from_raw_parts(name: Bytes, tag_pos: Option<usize>) -> Self {
        Self { name, tag_pos }
    }

    // TODO example
    // find position where tags start, optionally forcing re-search when it's already found
    // Note, that found position points to the first semicolon, not the tags part itself
    // fn find_tag_pos(&mut self) -> bool {
    //if force || self.tag_pos.is_none() {
    //self.tag_pos = self.name.iter().position(|c| *c == b';');
    //}
    //self.tag_pos.is_some()
    // }

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
    fn put_with_suffix(&self, buf: &mut BytesMut, suffix: &[u8], with_tags: bool) {
        let suflen = suffix.len();
        let namelen = self.name.len();

        buf.reserve(namelen + suflen + 1);
        match self.tag_pos {
            None => {
                buf.put_slice(&self.name);
                if suflen > 0 {
                    buf.put(b'.');
                    buf.put_slice(suffix);
                }
            }
            Some(pos) => {
                buf.put_slice(&self.name[..pos]);
                if suflen > 0 {
                    buf.put(b'.');
                    buf.put_slice(suffix);
                }
                if with_tags {
                    buf.put_slice(&self.name[pos..]);
                }
            }
        }
    }

    fn put_with_fixed_tag(&self, buf: &mut BytesMut, tag_name: &[u8], tag: &[u8], only_tag: bool) {
        // we must put tag in sorted order
        // only_tag = true means name part has been placed WITHOUT trailing semicolon

        // always add *at least* this amount of bytes
        let mut addlen = tag_name.len() + tag.len() + 1; // 1 is for `=`
        let namelen = self.name.len();

        if !only_tag {
            addlen += namelen
        }

        buf.reserve(addlen + 1); // 1 is for `;`
        match self.tag_pos {
            None => {
                if !only_tag {
                    buf.put_slice(&self.name);
                }
                // easy case: no tags
                if self.name[namelen - 1] != b';' {
                    buf.put(b';');
                }
                buf.put_slice(tag_name);
                buf.put(b'=');
                buf.put_slice(tag);
            }
            Some(pos) => {
                // put the name itself anyways
                if !only_tag {
                    buf.extend_from_slice(&self.name[..pos]); // add name without the semicolon
                }

                // knowing tags are already sorted
                // find position to place tag
                let mut offset = pos + 1; // no point to compare first semicolon

                for part in self.name[pos + 1..].split(|c| *c == b';') {
                    if part < tag_name {
                        offset += part.len() + 1; // always shift considering semicolon at the end
                    } else {
                        break;
                    }
                }

                // at this point offset is the position of name split
                // name is always in buffer, without trailing semicolon
                if offset >= namelen {
                    // new tag is put after all tags

                    // put the whole name with all old tags including first semicolon
                    buf.extend_from_slice(&self.name[pos..]);

                    // prepend new tag with semicolon if required
                    if self.name[namelen - 1] != b';' {
                        buf.put(b';');
                    }

                    // put new tag
                    buf.put_slice(tag_name);
                    buf.put(b'=');
                    buf.put_slice(tag);
                } else if offset == pos + 1 {
                    // new tag is put before all tags

                    // put the new tag with semicolon
                    buf.put(b';');
                    buf.put_slice(tag_name);
                    buf.put(b'=');
                    buf.put_slice(tag);

                    // put other tags with leading semicolon
                    buf.extend_from_slice(&self.name[pos..]);
                } else {
                    dbg!("MID", String::from_utf8_lossy(&buf[..]));
                    dbg!("MID1??", &self.name);
                    dbg!("MID1", String::from_utf8_lossy(&self.name[..]));
                    dbg!("MID2", String::from_utf8_lossy(&self.name[pos..offset]));
                    // put tags before offset including first semicolon
                    buf.extend_from_slice(&self.name[pos..offset]);

                    // put the new tag
                    buf.put_slice(tag_name);
                    buf.put(b'=');
                    buf.put_slice(tag);

                    if self.name[offset] != b';' {
                        buf.put(b';');
                    }

                    buf.extend_from_slice(&self.name[offset..]);
                }
                //
            }
        };
    }

    /// puts a name with an aggregate to provided buffer depending on dest
    /// to avoid putting different aggregates into same names
    /// requires all replacements to exist, giving error otherwise
    /// does no checks on overriding though
    #[allow(clippy::unit_arg)]
    pub fn put_with_aggregate<F>(
        // rustfmt
        &self,
        buf: &mut BytesMut,
        dest: AggregationDestination,
        agg: &Aggregate<F>,
        postfix_replacements: &HashMap<Aggregate<F>, String>,
        prefix_replacements: &HashMap<Aggregate<F>, String>,
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

        // find and put prefix first
        let prefix = prefix_replacements.get(agg).ok_or(())?;
        if !prefix.is_empty() {
            buf.reserve(prefix.len() + 1);
            buf.put(prefix);
            buf.put(b'.');
        }

        // we should not use let agg_postfix before the match, because with the tag case we don't
        // need it
        // the same applies to tag name and value: we only need them when aggregating to tags

        match dest {
            AggregationDestination::Smart if self.tag_pos.is_none() => {
                let agg_postfix = postfix_replacements.get(agg).ok_or(())?.as_bytes();
                // metric is untagged, add aggregate to name
                Ok(self.put_with_suffix(buf, agg_postfix, true))
            }
            AggregationDestination::Smart => {
                let agg_tag_name = tag_replacements.get(&Aggregate::AggregateTag).ok_or(())?.as_bytes();
                let agg_tag_value = tag_replacements.get(agg).ok_or(())?.as_bytes();

                // metric is tagged, add aggregate as tag
                Ok(self.put_with_fixed_tag(buf, agg_tag_name, agg_tag_value, false))
            }
            AggregationDestination::Name => {
                let agg_postfix = postfix_replacements.get(agg).ok_or(())?.as_bytes();
                Ok(self.put_with_suffix(buf, agg_postfix, true))
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

                self.put_with_suffix(buf, agg_postfix, false);
                self.put_with_fixed_tag(buf, agg_tag_name, agg_tag_value, true);
                Ok(())
            }
        }
    }
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

    fn new_name_graphite(n: &[u8]) -> MetricName {
        let mut buf = Vec::with_capacity(9000);
        buf.resize(9000, 0u8);
        MetricName::new(BytesMut::from(n), TagFormat::Graphite, &mut buf).unwrap()
    }

    fn assert_buf(buf: &mut BytesMut, match_: &[u8], error: &'static str) {
        let res = &buf.take()[..];
        assert_eq!(
            res,
            match_,
            "\n{}:\n{}\n{}",
            error,
            String::from_utf8_lossy(match_),
            String::from_utf8_lossy(res)
        );
    }

    #[test]
    fn metric_name_tag_position() {
        let name = Bytes::from(&b"gorets.bobez;a=b;c=d"[..]);
        assert_eq!(find_tag_pos(&name[..], TagFormat::Graphite), Some(12));
    }

    #[test]
    fn metric_name_tags_len() {
        let name = new_name_graphite(b"gorets.bobez;a=b;c=d");
        assert_eq!(name.tags_len(), 8);

        let name = new_name_graphite(&b"gorets.bobezzz"[..]);
        assert_eq!(name.tags_len(), 0);
    }

    #[test]
    fn metric_name_splits() {
        let name = new_name_graphite(&b"gorets.bobez"[..]);
        assert_eq!(name.name_without_tags(), &b"gorets.bobez"[..]);
        assert_eq!(name.tags_without_name().len(), 0);

        let name = new_name_graphite(&b"gorets.bobez;a=b;c=d"[..]);
        assert_eq!(name.name_without_tags(), &b"gorets.bobez"[..]);
        assert_eq!(name.tags_without_name(), &b";a=b;c=d"[..]);
    }

    #[test]
    fn metric_name_put_tag() {
        let with_tags = new_name_graphite(&b"gorets.bobez;aa=v;aa=z;bb=z;dd=h;"[..]);

        // create some replacements
        let mut po_reps: HashMap<Aggregate<f64>, String> = HashMap::new();
        po_reps.insert(Aggregate::Count, "count".to_string());

        let mut pr_reps: HashMap<Aggregate<f64>, String> = HashMap::new();
        pr_reps.insert(Aggregate::Count, String::new());

        let mut t_reps: HashMap<Aggregate<f64>, String> = HashMap::new();
        t_reps.insert(Aggregate::Count, "count".to_string());

        let mut buf = BytesMut::new();

        t_reps.insert(Aggregate::AggregateTag, "aa".to_string());

        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Tag, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();

        assert_buf(
            &mut buf,
            &b"gorets.bobez;aa=count;aa=v;aa=z;bb=z;dd=h"[..],
            "aggregate properly added in front of tags",
        );

        dbg!("WTF??", &with_tags.name);
        t_reps.insert(Aggregate::AggregateTag, "cc".to_string());
        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Tag, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();

        assert_buf(
            &mut buf,
            &b"gorets.bobez;aa=v;aa=z;bb=z;cc=count;dd=h"[..],
            "aggregate properly added in middle of tags",
        );

        t_reps.insert(Aggregate::AggregateTag, "ff".to_string());
        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Tag, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();

        assert_buf(
            &mut buf,
            &b"gorets.bobez;aa=v;aa=z;bb=z;dd=h;ff=count"[..],
            "aggregate properly added in end of tags",
        );
    }

    #[test]
    fn metric_aggregate_modes() {
        // create some replacements
        let mut po_reps: HashMap<Aggregate<f64>, String> = HashMap::new();
        po_reps.insert(Aggregate::Count, "count".to_string());
        po_reps.insert(Aggregate::AggregateTag, "MUST NOT BE USED".to_string());
        po_reps.insert(Aggregate::Percentile(0.8f64), "percentile80".to_string());
        po_reps.insert(Aggregate::UpdateCount, "".to_string());

        let mut pr_reps: HashMap<Aggregate<f64>, String> = HashMap::new();
        pr_reps.insert(Aggregate::Count, "counts".to_string());
        pr_reps.insert(Aggregate::UpdateCount, "updates".to_string());
        pr_reps.insert(Aggregate::Percentile(0.8f64), "".to_string());

        let mut t_reps: HashMap<Aggregate<f64>, String> = HashMap::new();
        t_reps.insert(Aggregate::Count, "cnt".to_string());
        t_reps.insert(Aggregate::UpdateCount, "updates".to_string());
        t_reps.insert(Aggregate::AggregateTag, "agg".to_string());
        // intentionally skip adding percentile80

        let without_tags = new_name_graphite(&b"gorets.bobez"[..]);
        let with_tags = new_name_graphite(&b"gorets.bobez;tag=value"[..]);
        let with_semicolon = new_name_graphite(&b"gorets.bobez;"[..]);

        // create 0-size buffer to make sure allocation counts work as intended
        let mut buf = BytesMut::new();

        // --------- without_tags

        // max is not in replacements
        assert!(
            without_tags
                .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::Max, &po_reps, &pr_reps, &t_reps)
                .is_err(),
            "non existing replacement gave no error"
        );

        // value is aggregated withtout replacements
        without_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::Value, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez"[..]);

        // update count is aggregated only with prefix
        without_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::UpdateCount, &po_reps, &pr_reps, &t_reps)
            .unwrap();

        assert_eq!(&buf.take()[..], &b"updates.gorets.bobez"[..]);

        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::UpdateCount, &po_reps, &pr_reps, &t_reps)
            .unwrap();

        assert_buf(
            &mut buf,
            &b"updates.gorets.bobez;agg=updates;tag=value"[..],
            "add aggregate to tagged metric in smart mode",
        );

        // different aggregation modes work as intended
        without_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"counts.gorets.bobez.count"[..]);

        without_tags
            .put_with_aggregate(
                &mut buf,
                &AggregationDestination::Smart,
                &Aggregate::Percentile(0.8f64),
                &po_reps,
                &pr_reps,
                &t_reps,
            )
            .unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.percentile80"[..], "existing postfix replacement was not put");

        without_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Name, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"counts.gorets.bobez.count"[..]);

        without_tags
            .put_with_aggregate(
                &mut buf,
                &AggregationDestination::Name,
                &Aggregate::Percentile(0.8f64),
                &po_reps,
                &pr_reps,
                &t_reps,
            )
            .unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez.percentile80"[..], "existing postfix replacement was not put");

        without_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Tag, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"counts.gorets.bobez;agg=cnt"[..]);

        let err = without_tags.put_with_aggregate(
            &mut buf,
            &AggregationDestination::Tag,
            &Aggregate::Percentile(0.8f64),
            &po_reps,
            &pr_reps,
            &t_reps,
        );
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        without_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Both, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"counts.gorets.bobez.count;agg=cnt"[..]);

        let err = without_tags.put_with_aggregate(
            &mut buf,
            &AggregationDestination::Both,
            &Aggregate::Percentile(0.8f64),
            &po_reps,
            &pr_reps,
            &t_reps,
        );
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        // --------- with_tags
        assert!(with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::Max, &po_reps, &pr_reps, &t_reps)
            .is_err());

        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::Value, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;tag=value"[..]);

        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"counts.gorets.bobez;agg=cnt;tag=value"[..]);

        let err = with_tags.put_with_aggregate(
            &mut buf,
            &AggregationDestination::Smart,
            &Aggregate::Percentile(0.8f64),
            &po_reps,
            &pr_reps,
            &t_reps,
        );
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Name, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_buf(&mut buf, &b"counts.gorets.bobez.count;tag=value"[..], "put tagged metric in name mode");

        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Tag, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_buf(
            &mut buf,
            &b"counts.gorets.bobez;agg=cnt;tag=value"[..],
            "add aggregate to tagged metric in tag mode",
        );

        let err = with_tags.put_with_aggregate(
            &mut buf,
            &AggregationDestination::Tag,
            &Aggregate::Percentile(0.8f64),
            &po_reps,
            &pr_reps,
            &t_reps,
        );
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        with_tags
            .put_with_aggregate(&mut buf, &AggregationDestination::Both, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_buf(
            &mut buf,
            &b"counts.gorets.bobez.count;agg=cnt;tag=value"[..],
            "add aggregate to tagged metric in both mode",
        );

        let err = with_tags.put_with_aggregate(
            &mut buf,
            &AggregationDestination::Both,
            &Aggregate::Percentile(0.8f64),
            &po_reps,
            &pr_reps,
            &t_reps,
        );
        assert_eq!(err, Err(()), "p80 aggregated into tags whilt it should not");

        // ensure trailing semicolon is not duplicated
        with_semicolon
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::Value, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"gorets.bobez;"[..]);

        with_semicolon
            .put_with_aggregate(&mut buf, &AggregationDestination::Smart, &Aggregate::Count, &po_reps, &pr_reps, &t_reps)
            .unwrap();
        assert_eq!(&buf.take()[..], &b"counts.gorets.bobez;agg=cnt"[..]);
    }

    #[test]
    fn metric_name_sort_tags_graphite() {
        let mut name = BytesMut::from(&b"gorets2;tag3=shit;t2=fuck"[..]);
        let mode = TagFormat::Graphite;

        let mut intermediate: Vec<u8> = Vec::new();
        intermediate.resize(name.len(), 0u8);

        let tag_pos = find_tag_pos(&name, mode).unwrap();

        assert!(sort_tags(&mut name[..], mode, &mut intermediate, tag_pos).is_ok());

        let mut name = BytesMut::from(&b"gorets.bobez;t=y;a=b;;;c=e;u=v;c=d;c=b;aaa=z"[..]);
        let tag_pos = find_tag_pos(&name, mode).unwrap();
        let tag_len = name.len() - tag_pos;
        assert_eq!(tag_len, 32);
        assert_eq!(&name[..tag_pos], &b"gorets.bobez"[..]);
        assert_eq!(&name[tag_pos..], &b";t=y;a=b;;;c=e;u=v;c=d;c=b;aaa=z"[..]);

        let mut intermediate = BytesMut::new();
        intermediate.resize(tag_len - 1, 0u8); // intentionally resize to less bytes than required
        assert!(sort_tags(&mut name[..], mode, &mut intermediate, tag_pos).is_err());

        intermediate.extend_from_slice(&[0u8]); // resize to good length now
        let newlen = sort_tags(&mut name[..], mode, &mut intermediate, tag_pos).unwrap();
        assert_eq!(newlen, name.len() - 2); // two semicolons should be removed
        assert_eq!(
            &name[..newlen],
            &b"gorets.bobez;a=b;aaa=z;c=b;c=d;c=e;t=y;u=v"[..],
            "{} {}",
            String::from_utf8_lossy(&name[..]),
            String::from_utf8_lossy(&name[..newlen])
        );

        let mut name = BytesMut::from(&b"gorets.bobez;bb=z;aa=z;aa=v;dd=h;"[..]);
        let newlen = sort_tags(&mut name[..], mode, &mut intermediate, tag_pos).unwrap();
        assert_eq!(&name[..newlen], &b"gorets.bobez;aa=v;aa=z;bb=z;dd=h"[..]);
    }
}
