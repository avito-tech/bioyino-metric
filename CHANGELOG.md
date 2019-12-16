# 0.2.0

* aggregation API has changed, supporting custom percentiles, specifying a list of aggregates, etc
* metric name is now a separate structure allowing to process tag information
* metrics can now have tags in (yet only one) graphite format
* tags are placed in sorted order during aggregation and parsing so they are always correct to be used without creating separate structure for them
* aggregation modes are not supported to put aggregate names in tags or name postfixes
* name prefixing is available per aggregate
