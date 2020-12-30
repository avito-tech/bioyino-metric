extern crate capnpc;

fn main() {
    capnpc::CompilerCommand::new()
        .file("schema/protocol.capnp")
        .file("schema/protocol-v2.capnp")
        .run()
        .expect("Failed compiling messages schema");
}
