use std::hash::{Hash, Hasher};

use bytes::BytesMut;

// TODO: think error type. There is single possible error atm, so sort_tags returns () instead
// TODO: think if we need sorted tags in btreemap instead of string
// TODO: handle repeating same tags i.e. gorets;a=b;e=b;a=b:...

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

    /// find position where tags start, forcing re-search when it's already found
    pub fn find_tag_pos(&mut self, force: bool) -> bool {
        if force || self.tag_pos.is_none() {
            self.tag_pos = self.name.iter().position(|c| *c == b';')
        }
        self.tag_pos.is_some()
    }

    /// returns only name, without tags, considers tag position was already found before
    pub fn without_tags(&self) -> &[u8] {
        if let Some(pos) = self.tag_pos {
            &self.name[..pos]
        } else {
            &self.name[..]
        }
    }

    /// sort tags in place using intermediate buffer, buffer length MUST be at least
    /// `name.len() - tag_pos` bytes
    /// sorting is made lexicographically
    pub fn sort_tags<B: AsMut<[u8]>>(
        &mut self,
        mode: TagMode,
        intermediate: &mut B,
    ) -> Result<(), ()> {
        if self.tag_pos.is_none() {
            if !self.find_tag_pos(true) {
                return Err(());
            }
        }

        let intermediate: &mut [u8] = intermediate.as_mut();
        if intermediate.len() < self.name.len() - self.tag_pos.unwrap() {
            return Err(());
        }
        use lazysort::Sorted;
        match mode {
            TagMode::Graphite => {
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
    //pub fn fetch_tags<B: BufMut>(&mut self, mode: TagMode) {
    //let tag_pos = match self.tag_pos {
    //None => return,
    //Some(t) => t,
    //};

    //match mode {
    //TagMode::Graphite => unimplemented!(),
    //}
    //}
}

impl Hash for MetricName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // metric with tag position found and tag position not found should be the same
        self.name.hash(state);
        //if Some(pos) = self.tag_pos {
        //self.tags.hash(state);
        //}
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
    fn metric_name_no_tags() {
        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez"[..]), None);
        name.find_tag_pos(true);
        assert_eq!(name.without_tags(), &b"gorets.bobez"[..]);

        let mut name = MetricName::new(BytesMut::from(&b"gorets.bobez;a=b;c=d"[..]), None);
        name.find_tag_pos(true);
        assert_eq!(name.without_tags(), &b"gorets.bobez"[..]);
    }

    #[test]
    fn metric_name_sort_tags_graphite() {
        let mut name = MetricName::new(
            BytesMut::from(&b"gorets.bobez;t=y;a=b;c=e;u=v;c=d;c=b;aaa=z"[..]),
            None,
        );
        name.find_tag_pos(false);
        let tag_pos = name.tag_pos.unwrap();

        let tag_len = name.name.len() - tag_pos;
        assert_eq!(tag_len, 30);

        //let mut intermediate = Vec::with_capacity(tag_len);
        let mut intermediate = BytesMut::new();
        intermediate.resize(tag_len - 1, 0u8); // intentionally resize to less bytes than required
        assert!(name
            .sort_tags(TagMode::Graphite, &mut intermediate)
            .is_err());

        intermediate.put(0u8);
        assert!(name.sort_tags(TagMode::Graphite, &mut intermediate).is_ok());
        assert_eq!(
            &name.name[..],
            &b"gorets.bobez;a=b;aaa=z;c=b;c=d;c=e;t=y;u=v"[..]
        );
    }
}
