use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use num_traits::{AsPrimitive, Float};
use serde_derive::{Deserialize, Serialize};

use crate::metric::{FromF64, Metric, MetricType};

// Percentile counter. Not safe. Requires at least two elements in vector
// vector must be sorted
pub fn percentile<F>(vec: &[F], nth: F) -> F
where
    F: Float + AsPrimitive<usize>,
{
    let last = F::from(vec.len() - 1).unwrap(); // usize to float should be ok for both f32 and f64
    if last == F::zero() {
        return vec[0];
    }

    let k = nth * last;
    let f = k.floor();
    let c = k.ceil();

    if c == f {
        // exact nth percentile have been found
        return vec[k.as_()];
    }

    let m0 = c - k;
    let m1 = k - f;
    let d0 = vec[f.as_()] * m0;
    let d1 = vec[c.as_()] * m1;
    d0 + d1
}

fn fill_cached_sum<F>(agg: &[F], sum: &mut Option<F>)
where
    F: Float,
{
    if sum.is_none() {
        let first = if let Some(first) = agg.first() { first } else { return };
        *sum = Some(agg.iter().skip(1).fold(*first, |acc, &v| acc + v))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields, try_from = "String")]
pub enum Aggregate<F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    #[serde(skip)]
    Value,
    Count,
    Last,
    Min,
    Max,
    Sum,
    Median,
    Mean,
    UpdateCount,
    #[serde(skip)]
    AggregateTag,
    Percentile(F),
}

impl<F> TryFrom<String> for Aggregate<F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "count" => Ok(Aggregate::Count),
            "last" => Ok(Aggregate::Last),
            "min" => Ok(Aggregate::Min),
            "max" => Ok(Aggregate::Max),
            "sum" => Ok(Aggregate::Sum),
            "median" => Ok(Aggregate::Median),
            "updates" => Ok(Aggregate::UpdateCount),
            s if s.starts_with("percentile-") => {
                // check in match guarantees minus char exists
                let pos = s.chars().position(|c| c == '-').unwrap() + 1;
                let num: u64 = u64::from_str(&s[pos..]).map_err(|_| "percentile value is not unsigned integer".to_owned())?;
                let mut divider = 10f64;

                let num = num as f64;
                // divider is f64, so it's always bigger than u64:MAX and therefore never
                // overflow
                while num > divider {
                    divider *= 10.0;
                }

                Ok(Aggregate::Percentile(F::from_f64(num / divider)))
            }
            _ => Err("".into()),
        }
    }
}

impl<F> Hash for Aggregate<F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Aggregate::Value => 0usize.hash(state),
            Aggregate::Count => 1usize.hash(state),
            Aggregate::Last => 2usize.hash(state),
            Aggregate::Min => 3usize.hash(state),
            Aggregate::Max => 4usize.hash(state),
            Aggregate::Sum => 5usize.hash(state),
            Aggregate::Median => 6usize.hash(state),
            Aggregate::Mean => 7usize.hash(state),
            Aggregate::UpdateCount => 8usize.hash(state),
            Aggregate::AggregateTag => 9usize.hash(state),
            // we need this for hashing and comparison, so we just use a value different from other
            // enum values
            // the second thing we need here is correctness, so nobody could send us some strange
            // percentile value like inf or nan (maybe there will be no such case, but just for the
            // sake of correctness we'd better do this
            Aggregate::Percentile(p) if !p.is_finite() => 11usize.hash(state), //std::usize::MAX,
            Aggregate::Percentile(p) => {
                let (mantissa, exponent, sign) = Float::integer_decode(*p);
                11usize.hash(state);
                mantissa.hash(state);
                exponent.hash(state);
                sign.hash(state);
            }
        }
    }
}

impl<F> Eq for Aggregate<F> where F: Float + Debug + FromF64 + AsPrimitive<usize> {}

// to be consistent with hasher, we do the comparison in the same way
impl<F> PartialEq for Aggregate<F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Aggregate::Count, Aggregate::Count) => true,
            (Aggregate::Last, Aggregate::Last) => true,
            (Aggregate::Min, Aggregate::Min) => true,
            (Aggregate::Max, Aggregate::Max) => true,
            (Aggregate::Sum, Aggregate::Sum) => true,
            (Aggregate::Median, Aggregate::Median) => true,
            (Aggregate::Mean, Aggregate::Mean) => true,
            (Aggregate::UpdateCount, Aggregate::UpdateCount) => true,
            (Aggregate::AggregateTag, Aggregate::AggregateTag) => true,
            // we need this for hashing and comparison, so we just use a value different from other
            // percentile value like inf or nan (maybe there will be no such case, but just for the
            // sake of correctness we'd better do this
            (Aggregate::Percentile(p), Aggregate::Percentile(o)) => {
                if p.is_normal() && o.is_normal() {
                    Float::integer_decode(*p) == Float::integer_decode(*o)
                } else {
                    p.classify() == o.classify()
                }
            }
            _ => false,
        }
    }
}

impl<F> Aggregate<F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    /// calculates the corresponding aggregate from the input metric
    /// returne None in all inapplicable cases, like zero-length vector or metric type mismatch
    /// cached_sum must relate to the same metric between calls, giving incorrect results or
    /// panics otherwise
    pub fn calculate(&self, metric: &Metric<F>, cached_sum: &mut Option<F>) -> Option<F> {
        // TODO: test
        match metric.mtype {
            // for sets calculate only count
            MetricType::Set(ref hs) if self == &Aggregate::Count => Some(F::from_f64(hs.len() as f64)),
            // don't count values for timers and sets
            MetricType::Set(_) if self == &Aggregate::Value => None,
            MetricType::Timer(_) if self == &Aggregate::Value => None,
            // for timers calculate all aggregates
            MetricType::Timer(ref agg) => match self {
                Aggregate::Value => None,
                Aggregate::Count => Some(F::from_f64(agg.len() as f64)),
                Aggregate::Last => agg.last().copied(),
                Aggregate::Min => Some(agg[0]),
                Aggregate::Max => Some(agg[agg.len() - 1]),
                Aggregate::Sum => {
                    fill_cached_sum(agg, cached_sum);
                    cached_sum.map(|sum| sum)
                }
                Aggregate::Median => Some(percentile(agg, F::from_f64(0.5))),
                Aggregate::Mean => {
                    // the case with len = 0 and sum != None is real here, but we intentinally let it
                    // panic on division by zero to get incorrect usage from code to be explicit
                    fill_cached_sum(agg, cached_sum);
                    cached_sum.map(|sum| {
                        let len: F = F::from_f64(agg.len() as f64);
                        sum / len
                    })
                }
                Aggregate::UpdateCount => Some(F::from_f64(f64::from(metric.update_counter))),
                Aggregate::AggregateTag => None,
                Aggregate::Percentile(ref p) => Some(percentile(agg, *p)),
            },
            // cout value for all types except timers (matched above)
            _ if self == &Aggregate::Value => Some(metric.value),
            // for other types calculate only update counter
            _ if self == &Aggregate::UpdateCount => Some(F::from_f64(f64::from(metric.update_counter))),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    #[test]
    fn aggregates_eq_and_hashing_f32() {
        let c32: Aggregate<f32> = Aggregate::Count;
        assert!(c32 != Aggregate::Min);

        assert!(Aggregate::Percentile(0.75f32) != Aggregate::Percentile(0.999));

        let mut hm = HashMap::new();
        // ensure hashing works good for typical percentiles up to 99999
        for p in 1..100_000u32 {
            hm.insert(Aggregate::Percentile(1f32 / p as f32), ());
            hm.insert(Aggregate::Percentile(1f32 / p as f32), ());
        }
        assert_eq!(hm.len(), 100000 - 1);
    }

    #[test]
    fn aggregates_eq_and_hashing_f64() {
        let c64: Aggregate<f64> = Aggregate::Count;
        assert!(c64 != Aggregate::Min);

        assert!(Aggregate::Percentile(0.75f64) != Aggregate::Percentile(0.999f64));

        let mut hm = HashMap::new();
        // ensure hashing works good for typical percentiles up to 99999
        for p in 1..100_000u32 {
            hm.insert(Aggregate::Percentile(1f64 / f64::from(p)), ());
            hm.insert(Aggregate::Percentile(1f64 / f64::from(p)), ());
        }
        assert_eq!(hm.len(), 100000 - 1);
    }
}
