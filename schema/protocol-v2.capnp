@0xa1bc8b0a8a38d5b3;

# This is a schema for bioyino peer messaging, used internally
# to pass preaggregated metric batches between agents and cluster nodes
#
# unlike version 1 or protocol, version 2 is not intended to be used as a
# public interfacee to bioyino
#
# If such a public API is required, feel free to create the issue or PR at the project repository

# capnproto allows to skip sending any of the fields
# if they are separate types, so there is no need to integrate
# option-like type into schema type system.

struct Message {
    # we want to keep this type extensible, and avoid the limitations for all new types in union
    # as capnproto specification requires
    # to make it a union out of the box, we introduce the noop message, wich can be used for
    # ping purpose or anything else
    union {
        noop @0 :Void;
        snapshot @1 :List(Metric);
    }
}

struct Metric {
    # everyone should have a name, even metrics
    name @0 :Text;

    # the internal values depends on metric type and is defined in a separate type
    value @1 :MetricValue;

    struct MetricValue {
        union {
            # gauge always replaces the previous value
            # statsd +value or -value are not allowed
            gauge @0 :Float64;

            # counter value is stored inside it's value
            counter @1 :Float64;

            # timer holds all values for further stats counting
            timer @2 :List(Float64);

            # set holds all value hashed for further cardinality estimation
            set @3 :List(UInt64);

            # custom range histogram holds buckets with boundaries and counters
            # the histogram is not cumulative
            # we count buckets using "right of or equal" rule
            #
            # example: 10 buckets in 0-10 range will look like
            # c, (0, c0), (1, c1), ... (10, c10)
            # where c0 will store number of values right of zero, including zero itself
            # and left of 1 NOT including 1 itself
            # the first value c is boundless and is the catch-all for all values < 0
            # the last bucket - c10 is catch-all bucket for all values >= 10
            #
            customHistogram @4 :CustomHistogram;
        }

        struct CustomHistogram {
            leftBucket @0 :UInt64;
            buckets @1 :List(RightOf);
            struct RightOf {
                value @0 :Float64;
                counter @1 :UInt64;
            }
        }
    }

    # a timesamp can optionally be sent and may be used for protocols other than statsd
    # the separate type is ised to make the posibility for timestamp to be optional
    timestamp @2 :Timestamp;

    struct Timestamp {
        ts @0 :UInt64;
    }

    # any other useful data about metric
    meta @3 :MetricMeta;

    struct MetricMeta {
        updateCounter @0 :UInt32;

        # as of now tags are stored as a part of metric name
        # there is no plans to change this at the moment because statsd doesn's seem
        # to be the right place for analyzing tags in any way except normalization
        # which is required for aggregation
        #
        # yet, some future implementations may still have tags included as a separate
        # structure, and we sacrifice some bytes for the sake of such compatibility
        tags :union {
            noTags @1 :Void;
            graphite @2 :UInt64;
            #tagList @3 :List(Tag);
        }

        #struct Tag {
        #    key @0 :Text;
        #    value @1 :Text;
        #}
    }
}
