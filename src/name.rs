use std::collections::HashMap;
use std::fmt::Debug;

use bytes::{BufMut, Bytes, BytesMut};
use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::aggregate::Aggregate;
use crate::metric::{FromF64, MetricTypeName};

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

            if intermediate.len() < (name.len() - tag_pos - 1) {
                return Err(());
            }

            let mut offset = 0; // meaningful length of data in intermediate buffer
            for part in name.split(|c| *c == b';').skip(1).sorted() {
                if part.is_empty() {
                    continue;
                }

                let end = offset + part.len();
                intermediate[offset..end].copy_from_slice(part);
                // add a trailing semicolon, if it is not an end of the buffer
                if end < intermediate.len() {
                    intermediate[end] = b';';
                }
                // anyways, set the offset to the next position
                //
                // in case of last part and precise length of the intermediate
                // this will be next byte after intermediate boundary
                // this byte will never be read and will be decreased right away
                offset = end + 1;
            }

            // remove trailing semicolon, if any was added
            if offset > 0 {
                offset -= 1;
            }

            let newlen = tag_pos + 1 + offset;
            name[tag_pos + 1..newlen].copy_from_slice(&intermediate[..offset]);
            Ok(newlen)
        }
    }
}

/// Represents a metric name as a buffer containing the full metric name including tags.
/// Also provides methods to work with tags.
#[derive(Debug, Eq, Clone, PartialEq, Hash)]
pub struct MetricName {
    pub name: Bytes,
    pub(crate) tag_pos: Option<usize>,
    //pub(crate) tag_format: TagFormat,  // TODO we may need this in future
    //pub tags: BTreeMap<BytesMut, BytesMut>, // we may need btreemap to have tags sorted
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
    /// be found according to required format
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
                    buf.put_u8(b'.');
                    buf.put_slice(suffix);
                }
            }
            Some(pos) => {
                buf.put_slice(&self.name[..pos]);
                if suflen > 0 {
                    buf.put_u8(b'.');
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
                // easy case: no tags
                if !only_tag {
                    buf.put_slice(&self.name);
                }

                // special case: tag name and value is empty: do nothing
                if !tag_name.is_empty() || !tag.is_empty() {
                    if self.name[namelen - 1] != b';' {
                        buf.put_u8(b';');
                    }

                    if !tag_name.is_empty() {
                        buf.put_slice(tag_name);
                        buf.put_u8(b'=');
                        buf.put_slice(tag);
                    }
                }
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
                        buf.put_u8(b';');
                    }

                    // put new tag
                    if !tag_name.is_empty() {
                        buf.put_slice(tag_name);
                        buf.put_u8(b'=');
                        buf.put_slice(tag);
                    }
                } else if offset == pos + 1 {
                    // new tag is put before all tags

                    // put the new tag with semicolon
                    //
                    if !tag_name.is_empty() {
                        buf.put_u8(b';');
                        buf.put_slice(tag_name);
                        buf.put_u8(b'=');
                        buf.put_slice(tag);
                    }

                    // put other tags with leading semicolon
                    buf.extend_from_slice(&self.name[pos..]);
                } else {
                    // put tags before offset including first semicolon
                    buf.extend_from_slice(&self.name[pos..offset]);

                    // put the new tag
                    if !tag_name.is_empty() {
                        buf.put_slice(tag_name);
                        buf.put_u8(b'=');
                        buf.put_slice(tag);

                        if self.name[offset] != b';' {
                            buf.put_u8(b';');
                        }
                    }
                    buf.extend_from_slice(&self.name[offset..]);
                }
                //
            }
        };
    }

    /// Puts a name with an aggregate to provided buffer depending on dest.
    /// To avoid putting different aggregates into same names requires all replacements to exist, giving error otherwise.
    /// Does no do the check if such overriding is done.
    pub fn put_full(&self, buf: &mut BytesMut, dest: AggregationDestination, postfix: &[u8], prefix: &[u8], tag: &[u8], tag_value: &[u8]) {
        if !prefix.is_empty() {
            buf.reserve(prefix.len() + 1);
            buf.put_slice(prefix);
            buf.put_u8(b'.');
        }

        // we should not use let agg_postfix before the match, because with the tag case we don't
        // need it
        // the same applies to tag name and value: we only need them when aggregating to tags

        match dest {
            AggregationDestination::Smart if self.tag_pos.is_none() => {
                // metric is untagged, add aggregate to name
                self.put_with_suffix(buf, postfix, true)
            }
            AggregationDestination::Smart => {
                // metric is tagged, add aggregate as tag
                self.put_with_fixed_tag(buf, tag, tag_value, false)
            }
            AggregationDestination::Name => self.put_with_suffix(buf, postfix, true),
            AggregationDestination::Tag => self.put_with_fixed_tag(buf, tag, tag_value, false),
            AggregationDestination::Both => {
                self.put_with_suffix(buf, postfix, false);
                self.put_with_fixed_tag(buf, tag, tag_value, true)
            }
        }
    }

    /// Puts a name with an aggregate to provided buffer depending on dest.
    /// To avoid putting different aggregates into same names requires all replacements to exist, giving error otherwise.
    /// Does no do the check if such overriding is done.
    /// Corner cases:
    /// * does not put tag if tag name is empty, puts it if value is empty though
    /// * does not consider (probably expected) defaults, like not putting postfix or tag_value for Aggregate::Value;
    /// this beaviour must be explicitly specified in options with prefix = "" and/or tag_name = "".
    #[allow(clippy::unit_arg)]
    pub fn put_with_options<F>(
        &self,
        buf: &mut BytesMut,
        name: MetricTypeName,
        agg: Aggregate<F>,
        options: &HashMap<(MetricTypeName, Aggregate<F>), NamingOptions>,
    ) -> Result<(), ()>
    where
        F: Float + Debug + FromF64 + AsPrimitive<usize>,
    {
        let naming = options.get(&(name, agg)).ok_or(())?;

        self.put_full(buf, naming.destination, &naming.postfix, &naming.prefix, &naming.tag, &naming.tag_value);

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct NamingOptions {
    /// global default prefix
    pub prefix: Bytes,

    /// the default tag name(i.e. key) to be used for aggregation
    pub tag: Bytes,

    /// replacements for aggregate tag values, naming is <tag>=<tag_value>
    pub tag_value: Bytes,

    /// names for aggregate postfixes
    pub postfix: Bytes,

    /// Where to put aggregate postfix
    pub destination: AggregationDestination,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::possible_aggregates;

    fn new_name_graphite(n: &[u8]) -> MetricName {
        let mut buf = Vec::with_capacity(9000);
        buf.resize(9000, 0u8);
        MetricName::new(BytesMut::from(n), TagFormat::Graphite, &mut buf).unwrap()
    }

    pub fn default_options(s: &[u8]) -> NamingOptions {
        NamingOptions {
            prefix: Bytes::new(),
            tag: Bytes::copy_from_slice(b"aggregate"),
            tag_value: Bytes::copy_from_slice(s),
            postfix: Bytes::copy_from_slice(s),
            destination: AggregationDestination::Smart,
        }
    }

    fn assert_buf(buf: &mut BytesMut, match_: &[u8], error: &'static str) {
        let res = &buf.split()[..];
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

        let typename = MetricTypeName::Timer;

        // create some replacements
        let mut opts: HashMap<(MetricTypeName, Aggregate<f64>), NamingOptions> = HashMap::new();
        let nopts = default_options(b"count");
        let key = (MetricTypeName::Timer, Aggregate::Count);
        opts.insert(key, nopts);

        let mut buf = BytesMut::new();

        opts.get_mut(&key).unwrap().destination = AggregationDestination::Tag;
        opts.get_mut(&key).unwrap().tag = Bytes::from_static(b"aa");

        with_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts).unwrap();

        assert_buf(
            &mut buf,
            &b"gorets.bobez;aa=count;aa=v;aa=z;bb=z;dd=h"[..],
            "aggregate properly added in front of tags",
        );

        opts.get_mut(&key).unwrap().tag = Bytes::from_static(b"cc");
        with_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts).unwrap();

        assert_buf(
            &mut buf,
            &b"gorets.bobez;aa=v;aa=z;bb=z;cc=count;dd=h"[..],
            "aggregate properly added in middle of tags",
        );

        opts.get_mut(&key).unwrap().tag = Bytes::from_static(b"ff");
        with_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts).unwrap();

        assert_buf(
            &mut buf,
            &b"gorets.bobez;aa=v;aa=z;bb=z;dd=h;ff=count"[..],
            "aggregate properly added in end of tags",
        );
    }

    #[test]
    fn metric_naming_aggregates() {
        let all_aggs = possible_aggregates(Some(1.5f64), Some(10));
        let mut intermediate: Vec<u8> = Vec::new();
        intermediate.resize(256, 0u8);

        let name = MetricName::new(BytesMut::from("gorets.bobez;tag2=val2;tag1=value1"), TagFormat::Graphite, &mut intermediate).unwrap();

        // create naming options for all of the aggs
        let mut opts: HashMap<(MetricTypeName, Aggregate<f64>), NamingOptions> = HashMap::new();
        for (ty, aggs) in &all_aggs {
            for agg in aggs {
                opts.insert(
                    (ty.clone(), agg.clone()),
                    // for testing purposes we:
                    // * make the min aggregate have no tag and no postfix emulating value aggregate
                    // * make the max aggregate have no tag but still have postfix
                    // * use the default value aggregate setting where the postfix is empty but tag exists having empty tag_value
                    NamingOptions {
                        prefix: Bytes::copy_from_slice(b"prefix"),
                        tag: if agg == &Aggregate::Max || agg == &Aggregate::Min {
                            // for testing purposes put the empty tag name for max aggregate
                            Bytes::new()
                        } else {
                            Bytes::copy_from_slice(b"aggtag")
                        },
                        tag_value: Bytes::copy_from_slice(agg.to_string().as_bytes()),
                        postfix: if agg == &Aggregate::Min {
                            // for testing purposes put the empty tag name for max aggregate
                            Bytes::new()
                        } else {
                            Bytes::copy_from_slice(agg.to_string().as_bytes())
                        },
                        destination: AggregationDestination::Both, // we can check both destination at once
                    },
                );
            }
        }

        for (ty, aggs) in all_aggs {
            for agg in aggs {
                let mut buf = BytesMut::new();
                let inagg = agg.to_string();
                // we obviously do not use format! in production code, but the behaviour is like here
                let expected = if let Aggregate::Value = agg {
                    // value aggregate has empty postfix, so no dot should be put at the end of the name
                    Bytes::copy_from_slice(format!("prefix.gorets.bobez;aggtag={agg};tag1=value1;tag2=val2", agg = &inagg).as_bytes())
                } else if let Aggregate::Max = agg {
                    // max aggregate has empty tag name, so no tag should be put
                    Bytes::copy_from_slice(format!("prefix.gorets.bobez.{agg};tag1=value1;tag2=val2", agg = &inagg).as_bytes())
                } else if let Aggregate::Min = agg {
                    // min aggregate has no tag and no postfix, so it should ne like metric, but
                    // with sorted tags
                    Bytes::copy_from_slice(b"prefix.gorets.bobez;tag1=value1;tag2=val2")
                } else if let Aggregate::Percentile(_, _) = agg {
                    // this is separated to check percentile naming
                    Bytes::copy_from_slice(b"prefix.gorets.bobez.percentile.99;aggtag=percentile.99;tag1=value1;tag2=val2")
                } else {
                    Bytes::copy_from_slice(format!("prefix.gorets.bobez.{agg};aggtag={agg};tag1=value1;tag2=val2", agg = &inagg).as_bytes())
                };
                name.put_with_options(&mut buf, ty, agg, &opts).expect("putting name failed");
                //dbg!(&ty, &expected, &buf);
                assert_eq!(expected, buf);
            }
        }
    }

    #[test]
    fn metric_aggregate_modes() {
        let typename = MetricTypeName::Timer;

        // create some replacements
        // for count replace prefix and tag value
        let mut opts: HashMap<(MetricTypeName, Aggregate<f64>), NamingOptions> = HashMap::new();
        let mut nopts = default_options(b"count");
        nopts.prefix = Bytes::from_static(b"counts");
        nopts.tag = Bytes::from_static(b"agg");
        nopts.tag_value = Bytes::from_static(b"cnt");
        let timer = MetricTypeName::Timer;
        opts.insert((timer, Aggregate::Count), nopts);

        // for percentile80 replace prefix with empty value
        let mut nopts = default_options(b"percentile80");
        nopts.prefix = Bytes::new();
        opts.insert((timer, Aggregate::Percentile(0.8f64, 80)), nopts);

        // for update count  replace prefix and tag name
        let mut nopts = default_options(b"updates");
        nopts.postfix = Bytes::from_static(b"");
        nopts.prefix = Bytes::from_static(b"updates");
        nopts.tag = Bytes::from_static(b"agg");

        opts.insert((timer, Aggregate::UpdateCount), nopts);

        let without_tags = new_name_graphite(b"gorets.bobez");
        let with_tags = new_name_graphite(b"gorets.bobez;tag=value");
        let with_semicolon = new_name_graphite(b"gorets.bobez;");

        // create the same options but with name mode aggregation
        let mut opts_mode_name = opts.clone();
        for (_, v) in &mut opts_mode_name {
            v.destination = AggregationDestination::Name;
        }
        // create the same options but with tag mode aggregation
        let mut opts_mode_tag = opts.clone();
        for (_, v) in &mut opts_mode_tag {
            v.destination = AggregationDestination::Tag;
        }

        // create the same options but with both mode aggregation
        let mut opts_mode_both = opts.clone();
        for (_, v) in &mut opts_mode_both {
            v.destination = AggregationDestination::Both;
        }

        // create 0-size buffer to make sure allocation counts work as intended
        let mut buf = BytesMut::new();

        // --------- without_tags, smart mode

        // max is not in replacements
        assert!(
            without_tags.put_with_options(&mut buf, typename, Aggregate::Max, &opts).is_err(),
            "non existing replacement gave no error"
        );

        without_tags.put_with_options(&mut buf, typename, Aggregate::UpdateCount, &opts).unwrap();
        assert_eq!(
            &buf.split()[..],
            &b"updates.gorets.bobez"[..],
            "update count should be aggregated only with prefix"
        );

        with_tags.put_with_options(&mut buf, typename, Aggregate::UpdateCount, &opts).unwrap();
        assert_buf(
            &mut buf,
            &b"updates.gorets.bobez;agg=updates;tag=value"[..],
            "aggregate is added to tagged metric in smart mode",
        );

        without_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts).unwrap();
        assert_buf(
            &mut buf,
            &b"counts.gorets.bobez.count"[..],
            "postfix and prefix are placed properly in smart mode",
        );

        let mut mopts = opts.clone();
        let mut nopts = default_options(b"");
        nopts.destination = AggregationDestination::Tag;
        nopts.tag = Bytes::new();
        mopts.insert((timer, Aggregate::Value), nopts);
        without_tags.put_with_options(&mut buf, typename, Aggregate::Value, &mopts).unwrap();
        assert_buf(&mut buf, &b"gorets.bobez"[..], "semicolon not added in tag mode");

        without_tags
            .put_with_options(&mut buf, typename, Aggregate::Percentile(0.8f64, 80), &opts)
            .unwrap();
        assert_buf(&mut buf, &b"gorets.bobez.percentile80"[..], "existing postfix replacement was not put");

        without_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts_mode_name).unwrap();
        assert_buf(&mut buf, &b"counts.gorets.bobez.count"[..], "name aggregation works properly");

        without_tags
            .put_with_options(&mut buf, typename, Aggregate::Percentile(0.8f64, 80), &opts_mode_name)
            .unwrap();

        // checks aggregate hashing
        assert_buf(&mut buf, &b"gorets.bobez.percentile80"[..], "existing postfix replacement was not put");

        without_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts_mode_tag).unwrap();
        assert_buf(&mut buf, &b"counts.gorets.bobez;agg=cnt"[..], "tag aggregated properly");

        without_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts_mode_both).unwrap();
        assert_buf(&mut buf, &b"counts.gorets.bobez.count;agg=cnt"[..], "both mode aggregations works properly");

        // --------- with_tags
        assert!(with_tags.put_with_options(&mut buf, typename, Aggregate::Max, &opts).is_err());

        let mut mopts = opts.clone();
        let mut nopts = default_options(b"");
        nopts.tag = Bytes::from_static(b"");
        mopts.insert((timer, Aggregate::Value), nopts);
        with_tags.put_with_options(&mut buf, typename, Aggregate::Value, &mopts).unwrap();
        assert_buf(
            &mut buf,
            &b"gorets.bobez;tag=value"[..],
            "tag is properly ignored for value aggregate in smart mode",
        );

        let mut mopts = opts.clone();
        let mut nopts = default_options(b"");
        nopts.tag = Bytes::from_static(b"agg");
        nopts.tag_value = Bytes::from_static(b"");
        mopts.insert((timer, Aggregate::Value), nopts);
        with_tags.put_with_options(&mut buf, typename, Aggregate::Value, &mopts).unwrap();
        assert_buf(
            &mut buf,
            &b"gorets.bobez;agg=;tag=value"[..],
            "tag with empty value is properly added in smart mode",
        );

        with_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts).unwrap();
        assert_buf(
            &mut buf,
            &b"counts.gorets.bobez;agg=cnt;tag=value"[..],
            "tag is at proper sorted place in smart mode",
        );

        with_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts_mode_name).unwrap();
        assert_buf(&mut buf, &b"counts.gorets.bobez.count;tag=value"[..], "put tagged metric in name mode");

        with_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts_mode_tag).unwrap();
        assert_buf(
            &mut buf,
            &b"counts.gorets.bobez;agg=cnt;tag=value"[..],
            "add aggregate to tagged metric in tag mode",
        );

        with_tags.put_with_options(&mut buf, typename, Aggregate::Count, &opts_mode_both).unwrap();
        assert_buf(
            &mut buf,
            &b"counts.gorets.bobez.count;agg=cnt;tag=value"[..],
            "add aggregate to tagged metric in both mode",
        );

        let mut mopts = opts.clone();
        let mut nopts = default_options(b"");
        nopts.tag = Bytes::new();
        mopts.insert((timer, Aggregate::Value), nopts);
        with_semicolon.put_with_options(&mut buf, typename, Aggregate::Value, &mopts).unwrap();
        assert_buf(&mut buf, &b"gorets.bobez;"[..], "trailing semicolon is not duplicated");

        with_semicolon.put_with_options(&mut buf, typename, Aggregate::Count, &opts).unwrap();
        assert_buf(&mut buf, &b"counts.gorets.bobez;agg=cnt"[..], "semicolon is added properly");
    }

    #[test]
    fn metric_name_sort_tags_graphite() {
        let mode = TagFormat::Graphite;

        let mut intermediate: Vec<u8> = Vec::new();
        let x = b"12345678";
        let y = b"abcdefgh";
        for i in 2..x.len() {
            for j in 2..y.len() {
                let mut name = BytesMut::from(&b"test_corrupted_tags;y="[..]);
                name.extend_from_slice(&y[..j + 1]);
                name.extend_from_slice(&b";x="[..]);
                name.extend_from_slice(&x[..i + 1]);
                intermediate.resize(name.len(), b'z');
                let tag_pos = find_tag_pos(&name, mode).unwrap();

                let newlen = sort_tags(&mut name[..], mode, &mut intermediate, tag_pos).unwrap();

                let mut expected = BytesMut::from(&b"test_corrupted_tags;x="[..]);
                expected.extend_from_slice(&x[..i + 1]);
                expected.extend_from_slice(&b";y="[..]);
                expected.extend_from_slice(&y[..j + 1]);

                assert_buf(&mut BytesMut::from(&name[..newlen]), &expected[..], "sorting does not miss last byte");
            }
        }

        let mut name = BytesMut::from(&b"gorets2;tag3=shit;t2=fuck"[..]);

        let mut intermediate: Vec<u8> = Vec::new();

        let tag_pos = find_tag_pos(&name, mode).unwrap();
        intermediate.resize(name.len() - tag_pos, b'z');
        let newlen = sort_tags(&mut name[..], mode, &mut intermediate, tag_pos).unwrap();
        assert_eq!(
            &name[..newlen],
            &b"gorets2;t2=fuck;tag3=shit"[..],
            "{}",
            String::from_utf8_lossy(&name[..newlen]),
        );

        let mut name = BytesMut::from(&b"gorets.bobez;t=y;a=b;;;c=e;u=v;c=d;c=b;aaa=z"[..]);
        let tag_pos = find_tag_pos(&name, mode).unwrap();
        let tag_len = name.len() - tag_pos;
        assert_eq!(tag_len, 32);
        assert_eq!(&name[..tag_pos], &b"gorets.bobez"[..]);
        assert_eq!(&name[tag_pos..], &b";t=y;a=b;;;c=e;u=v;c=d;c=b;aaa=z"[..]);

        let mut intermediate = BytesMut::new();
        intermediate.resize(tag_len - 2, 0u8); // intentionally resize to less bytes than required
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
