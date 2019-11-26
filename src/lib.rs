pub mod aggregate;
pub mod metric;
pub mod name;
pub mod parser;

pub use crate::metric::*;
pub use crate::name::MetricName;

pub mod protocol_capnp {
    include!(concat!(env!("OUT_DIR"), "/schema/protocol_capnp.rs"));
}
