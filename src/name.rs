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
}

impl MetricName {
    pub fn new(name: BytesMut, tag_pos: Option<usize>) -> Self {
        Self { name, tag_pos }
    }

    pub fn find_tag_pos(&mut self) {
        if self.tag_pos.is_none() {
            self.tag_pos = self.name.iter().position(|c| *c == b';')
        }
    }

    pub fn normalize_tags(&mut self, mode: TagMode) {
        let tag_pos = match self.tag_pos {
            None => return,
            Some(t) => t,
        };

        // TODO: handle sorting
        // TODO: handle repeating same tags i.e. gorets;a=b;e=b;a=b:...
        match mode {
            TagMode::Graphite => unimplemented!(),
        }
    }
}

impl Hash for MetricName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // metric with tag position found and tag position not found should be the same
        self.name.hash(state);
    }
}
