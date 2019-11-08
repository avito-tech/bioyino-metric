use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use num_traits::{AsPrimitive, Float};
use serde_derive::{Deserialize, Serialize};

use crate::metric::FromF64;

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
            Aggregate::Percentile(f) => {
                let u: usize = f.as_(); //.rotate_left(5);
                u.hash(state)
            }
            p => p.as_usize().hash(state),
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
            (Aggregate::Percentile(f), Aggregate::Percentile(o)) => {
                let s: usize = f.as_().rotate_left(5);
                let o: usize = o.as_().rotate_left(5);
                s == o
            }
            (s, o) => s.as_usize() == o.as_usize(),
        }
    }
}

impl<F> Aggregate<F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    fn as_usize(&self) -> usize {
        match self {
            Aggregate::Count => 0,
            Aggregate::Last => 1,
            Aggregate::Min => 2,
            Aggregate::Max => 3,
            Aggregate::Sum => 4,
            Aggregate::Median => 5,
            Aggregate::Mean => 6,
            Aggregate::UpdateCount => 7,
            Aggregate::AggregateTag => 8,
            // we need this for hashing and comparison, so we just use a value different from other
            // enum values
            // the second thing we need here is correctness, so nobody could send us some strange
            // percentile value like inf or nan (maybe there will be no such case, but just for the
            // sake of correctness we'd better do this
            Aggregate::Percentile(p) if !p.is_finite() => std::usize::MAX,
            Aggregate::Percentile(p) if *p > FromF64::from_f64(std::usize::MAX as f64) => std::usize::MAX,
            Aggregate::Percentile(p) => (*p + FromF64::from_f64(10f64)).as_(),
        }
    }
    /// calculates the corresponding aggregate from the input vector
    /// may return none if length of agg if data is required for aggregate to be count
    /// agg and cached_sum must relate to the same metric between calls, giving incorrect results or
    /// panics otherwise
    pub fn calculate(&self, agg: &[F], cached_sum: &mut Option<F>, update_count: f64) -> Option<F> {
        match self {
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

            Aggregate::UpdateCount => Some(F::from_f64(update_count)),
            Aggregate::AggregateTag => None,
            Aggregate::Percentile(ref p) => Some(percentile(agg, *p)),
        }
    }
}
