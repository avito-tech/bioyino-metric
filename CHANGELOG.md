# 0.4.0
* sampling was not considered at all, now it is not a part of metric and is counted right away
* rate aggregation has been added as `update_counter / aggregation_period`, with period provided externally
* parser now parses floats from bytes without string conversion, giving around 10% speedup (see `benchmark-parser` branch for details)

# 0.2.0

* aggregation API has changed, supporting custom percentiles, specifying a list of aggregates, etc
* metric name is now a separate structure allowing to process tag information
* metrics can now have tags in (yet only one) graphite format
* tags are placed in sorted order during aggregation and parsing so they are always correct to be used without creating separate structure for them
* aggregation modes are not supported to put aggregate names in tags or name postfixes
* name prefixing is available per aggregate
