use crate::metric::Metric;
use crate::name::MetricName;

/// Metric with name and 64-bit value
#[derive(Debug, Clone, PartialEq)]
pub struct NamedMetric64 {
    pub name: MetricName,
    pub value: Metric<f64>,
}

impl NamedMetric64 {
    pub fn new(name: MetricName, value: Metric<f64>) -> Self {
        Self { name, value }
    }
}

/// Metric with name and 32-bit value
#[derive(Debug, Clone, PartialEq)]
pub struct NamedMetric32 {
    pub name: MetricName,
    pub value: Metric<f32>,
}

impl NamedMetric32 {
    pub fn new(name: MetricName, value: Metric<f32>) -> Self {
        Self { name, value }
    }
}
