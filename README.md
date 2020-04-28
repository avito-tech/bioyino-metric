# bioyino-metrics
This library contains useful types and methods for working with metrics in bioyino statsd server and some other
metric processing software. Features:

* a type for representing typed and timestamped metrics, generic over floating point format
* streaming parser of statsd format
* metric aggregation routines
* working with Graphite-compatible metric naming including basic tags support
* schema and functions for sending/receiving metrics in binary Cap'n'Proto format
 
