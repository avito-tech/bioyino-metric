use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::metric::{FromF64, Metric, MetricType, MetricTypeName};

/// Percentile counter. Not safe. Requires at least two elements in vector
/// vector MUST be sorted
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
        let first = if let Some(first) = agg.first() {
            first
        } else {
            return;
        };
        *sum = Some(agg.iter().skip(1).fold(*first, |acc, &v| acc + v))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields, try_from = "String")]
/// Contains list of all possible aggregates and some additional data.
pub enum Aggregate<F>
where
    F: Copy + Float + Debug + FromF64 + AsPrimitive<usize>,
{
    #[serde(skip)]
    /// a dummy aggregate, containing metric's original value
    Value,
    Count,
    Last,
    Min,
    Max,
    Sum,
    Median,
    Mean,
    UpdateCount,
    // user will want the exact same number formatting of percentile like in config, but
    // float converstions from/to string may loose it, so
    // we prefer to keep the percentile as is, with the original integer parsed from config value
    // this is also VERY useful when comparing and hashing percentiles too
    // the only one downside is that i.e. 0.8th and 0.800 th percentile will be different metrics,
    // but same values, but this is easily acceptable and should be really very rare case
    Percentile(F, u64),
}

impl<F> TryFrom<String> for Aggregate<F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "value" => Ok(Aggregate::Value),
            "count" => Ok(Aggregate::Count),
            "last" => Ok(Aggregate::Last),
            "min" => Ok(Aggregate::Min),
            "max" => Ok(Aggregate::Max),
            "sum" => Ok(Aggregate::Sum),
            "median" => Ok(Aggregate::Median),
            "mean" => Ok(Aggregate::Mean),
            "updates" => Ok(Aggregate::UpdateCount),
            s if s.starts_with("percentile-") => {
                // check in match guarantees minus char exists
                let pos = s.chars().position(|c| c == '-').unwrap() + 1;
                let num: u64 = u64::from_str(&s[pos..]).map_err(|_| "percentile value is not unsigned integer".to_owned())?;
                let mut divider = 10f64;

                let numf = num as f64;
                // divider is f64, so it's always bigger than u64:MAX and therefore never
                // overflow
                while numf > divider {
                    divider *= 10.0;
                }

                Ok(Aggregate::Percentile(F::from_f64(numf / divider), num))
            }
            _ => Err("unknown aggregate name".into()),
        }
    }
}

impl<F> ToString for Aggregate<F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    fn to_string(&self) -> String {
        match self {
            Aggregate::Value => "".to_string(),
            Aggregate::Count => "count".to_string(),
            Aggregate::Last => "last".to_string(),
            Aggregate::Min => "min".to_string(),
            Aggregate::Max => "max".to_string(),
            Aggregate::Sum => "sum".to_string(),
            Aggregate::Median => "median".to_string(),
            Aggregate::Mean => "mean".to_string(),
            Aggregate::UpdateCount => "updates".to_string(),
            Aggregate::Percentile(p, _) if !p.is_finite() => "bad_percentile".to_string(),
            Aggregate::Percentile(_, num) => format!("percentile.{}", num),
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
            // we need this for hashing and comparison, so we just use a value different from other
            // enum values
            // the second thing we need here is correctness, so nobody could send us some strange
            // percentile value like inf or nan (maybe there will be no such case, but just for the
            // sake of correctness we'd better do this
            Aggregate::Percentile(p, _) if !p.is_finite() => 11usize.hash(state), //std::usize::MAX,
            Aggregate::Percentile(_, num) => {
                // it's ok to hash only number here because that's how we differ percentiles from
                // each other
                num.hash(state);
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
            (Aggregate::Value, Aggregate::Value) => true,
            (Aggregate::Count, Aggregate::Count) => true,
            (Aggregate::Last, Aggregate::Last) => true,
            (Aggregate::Min, Aggregate::Min) => true,
            (Aggregate::Max, Aggregate::Max) => true,
            (Aggregate::Sum, Aggregate::Sum) => true,
            (Aggregate::Median, Aggregate::Median) => true,
            (Aggregate::Mean, Aggregate::Mean) => true,
            (Aggregate::UpdateCount, Aggregate::UpdateCount) => true,
            // we need this for hashing and comparison, so we just use a value different from other
            // percentile value like inf or nan (maybe there will be no such case, but just for the
            // sake of correctness we'd better do this
            (Aggregate::Percentile(_, num), Aggregate::Percentile(_, o)) => num == o,
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
        match (&metric.mtype, self) {
            // for sets calculate only count
            (MetricType::Set(ref hs), &Aggregate::Count) => Some(F::from_f64(hs.len() as f64)),
            // don't count values for timers and sets
            (MetricType::Set(_), &Aggregate::Value) => None,
            (MetricType::Timer(_), &Aggregate::Value) => None,
            // for timers calculate all aggregates
            (MetricType::Timer(ref agg), s) => match s {
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
                Aggregate::Percentile(ref p, _) => Some(percentile(agg, *p)),
            },
            // cout value for all types except timers (matched above)
            (_, &Aggregate::Value) => Some(metric.value),
            // for other types calculate only update counter
            (_, &Aggregate::UpdateCount) => Some(F::from_f64(f64::from(metric.update_counter))),
            _ => None,
        }
    }
}

/// A state for calculating all aggregates over metric
/// Implements iterator returning the index of aggregate in the input and the aggregate value
/// if such value should exist for an aggregate
pub struct AggregateCalculator<'a, F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    metric: &'a Metric<F>,
    timer_sum: Option<F>,
    aggregates: &'a [Aggregate<F>],
    current: usize,
}

impl<'a, F> AggregateCalculator<'a, F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    pub fn new(metric: &'a mut Metric<F>, aggregates: &'a [Aggregate<F>]) -> Self {
        let timer_sum = if let MetricType::Timer(ref mut agg) = metric.mtype {
            agg.sort_unstable_by(|ref v1, ref v2| v1.partial_cmp(v2).unwrap());
            let first = agg.first().unwrap();
            Some(agg.iter().skip(1).fold(*first, |acc, &v| acc + v))
        } else {
            None
        };
        Self {
            metric,
            timer_sum,
            aggregates,
            current: 0,
        }
    }
}

impl<'a, F> Iterator for AggregateCalculator<'a, F>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    type Item = Option<(usize, F)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.aggregates.len() {
            return None;
        }

        let agg = &self.aggregates[self.current];
        let calc = agg.calculate(self.metric, &mut self.timer_sum).map(|result| (self.current, result));
        self.current += 1;
        Some(calc)
    }
}

/// A helper function giving all possible aggregates for each metric type name.
/// Includes ony one, 99th percentile for the sake of complenetes
pub fn possible_aggregates<F>() -> HashMap<MetricTypeName, Vec<Aggregate<F>>>
where
    F: Float + Debug + FromF64 + AsPrimitive<usize>,
{
    let mut map = HashMap::new();
    map.insert(MetricTypeName::Counter, vec![Aggregate::Value, Aggregate::UpdateCount]);
    map.insert(MetricTypeName::DiffCounter, vec![Aggregate::Value, Aggregate::UpdateCount]);

    map.insert(
        MetricTypeName::Timer,
        vec![
            Aggregate::Count,
            Aggregate::Last,
            Aggregate::Min,
            Aggregate::Max,
            Aggregate::Sum,
            Aggregate::Median,
            Aggregate::Mean,
            Aggregate::UpdateCount,
            Aggregate::Percentile(F::from_f64(0.99), 99),
        ],
    );
    map.insert(MetricTypeName::Gauge, vec![Aggregate::Value, Aggregate::UpdateCount]);
    map.insert(MetricTypeName::Set, vec![Aggregate::Count, Aggregate::UpdateCount]);
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::{HashMap, HashSet};

    #[test]
    fn aggregates_eq_and_hashing_f32() {
        let c32: Aggregate<f32> = Aggregate::Count;
        assert!(c32 != Aggregate::Min);

        assert!(Aggregate::Percentile(0.75f32, 75) != Aggregate::Percentile(0.999, 999));

        let mut hm = HashMap::new();
        // ensure hashing works good for typical percentiles up to 99999
        for p in 1..100_000u32 {
            hm.insert(Aggregate::Percentile(1f32 / p as f32, p as u64), ());
            hm.insert(Aggregate::Percentile(1f32 / p as f32, p as u64), ());
        }
        assert_eq!(hm.len(), 100000 - 1);
    }

    #[test]
    fn aggregates_eq_and_hashing_f64() {
        let c64: Aggregate<f64> = Aggregate::Count;
        assert!(c64 != Aggregate::Min);

        assert!(Aggregate::Percentile(0.75f64, 75) != Aggregate::Percentile(0.999f64, 999));

        let mut hm = HashMap::new();
        // ensure hashing works good for typical percentiles up to 99999
        for p in 1..100_000u32 {
            hm.insert(Aggregate::Percentile(1f64 / f64::from(p), p as u64), ());
            hm.insert(Aggregate::Percentile(1f64 / f64::from(p), p as u64), ());
        }
        assert_eq!(hm.len(), 100000 - 1);
    }

    #[test]
    fn percentile_to_string() {
        assert_eq!(&Aggregate::Percentile(0.75f64, 75).to_string(), "percentile.75");
        assert_eq!(&Aggregate::Percentile(0.009f64, 9).to_string(), "percentile.9");
        assert_eq!(&Aggregate::Percentile(0.8f64, 80).to_string(), "percentile.80");
        assert_eq!(&Aggregate::Percentile(0.800f64, 800).to_string(), "percentile.800");
    }

    #[test]
    fn aggregating_with_sampling() {
        // aggregate with sampling
        let samples = vec![12f64, 43f64, 1f64, 9f64, 84f64, 55f64, 31f64, 16f64, 64f64];
        // counter value is 1 after sampling considered
        let mut counter = Metric::new(0.01f64, MetricType::Counter, None, Some(0.01)).unwrap();
        samples
            .iter()
            .map(|t| {
                let counter2 = Metric::new(*t, MetricType::Counter, None, None).unwrap();
                counter.accumulate(counter2).unwrap();
            })
            .last();

        let expected = 316f64; // sum of samples + 1, 1 goes from the counter value itself

        //  there is a small precision loss due to float converions and sampling calculations
        //  so the resulting value will be close but not equal to expected one
        assert!((counter.value - expected).abs() < 0.0001, format!("{} ~= {}", counter.value, expected));
    }

    #[test]
    fn aggregating_with_iterator() {
        // create aggregates list
        // this also tests all aggregates are parsed
        let aggregates = vec!["count", "min", "updates", "max", "sum", "median", "percentile-85"];
        let mut aggregates = aggregates
            .into_iter()
            .map(|s| Aggregate::try_from(s.to_string()).unwrap())
            .collect::<Vec<Aggregate<f64>>>();
        // add hidden value aggregator
        aggregates.push(Aggregate::Value);

        let samples = vec![12f64, 43f64, 1f64, 9f64, 84f64, 55f64, 31f64, 16f64, 64f64];
        let mut expected = HashMap::new();

        let mut metrics = Vec::new();
        aggregates.iter().map(|agg| expected.insert(agg.clone(), Vec::new())).last();
        // NOTE: when modifying this test, push expected aggregates to vector in the same order as `aggregates` vector
        // (this is needed for equality test to work as intended)

        // TODO: gauge signs
        let mut gauge = Metric::new(1f64, MetricType::Gauge(None), None, None).unwrap();
        samples
            .iter()
            .map(|t| {
                let gauge2 = Metric::new(*t, MetricType::Gauge(None), None, None).unwrap();
                gauge.accumulate(gauge2).unwrap();
            })
            .last();

        metrics.push(gauge.clone());

        // gauges only consider last value
        expected.get_mut(&Aggregate::Value).unwrap().push((gauge.clone(), 64f64));
        expected.get_mut(&Aggregate::UpdateCount).unwrap().push((gauge.clone(), 10f64));

        // counters must be aggregated into two aggregates: value and update counter
        let mut counter = Metric::new(1f64, MetricType::Counter, None, None).unwrap();
        let mut sign = 1f64;
        samples
            .iter()
            .map(|t| {
                sign = -sign;
                let counter2 = Metric::new(*t * sign, MetricType::Counter, None, None).unwrap();
                counter.accumulate(counter2).unwrap();
            })
            .last();

        metrics.push(counter.clone());

        // aggregated value for counter is a sum of all incoming data considering signs
        expected.get_mut(&Aggregate::Value).unwrap().push((counter.clone(), -68f64));
        expected.get_mut(&Aggregate::UpdateCount).unwrap().push((counter.clone(), 10f64));

        // diff counter is when you send a counter value as is, but it's considered increasing and
        // only positive diff adds up
        let mut dcounter = Metric::new(1f64, MetricType::DiffCounter(0.0), None, None).unwrap();
        samples
            .iter()
            .map(|t| {
                let dcounter2 = Metric::new(*t, MetricType::DiffCounter(0.0), None, None).unwrap();
                dcounter.accumulate(dcounter2).unwrap();
            })
            .last();

        metrics.push(dcounter.clone());

        expected.get_mut(&Aggregate::Value).unwrap().push((dcounter.clone(), 278f64));
        expected.get_mut(&Aggregate::UpdateCount).unwrap().push((dcounter.clone(), 10f64));

        // timers should be properly aggregated into all agregates except value
        let mut timer = Metric::new(1f64, MetricType::Timer(Vec::new()), None, None).unwrap();

        samples
            .iter()
            .map(|t| {
                let timer2 = Metric::new(*t, MetricType::Timer(Vec::new()), None, None).unwrap();
                timer.accumulate(timer2).unwrap();
            })
            .last();

        expected.get_mut(&Aggregate::Count).unwrap().push((timer.clone(), 10f64));
        expected.get_mut(&Aggregate::Min).unwrap().push((timer.clone(), 1f64));
        expected.get_mut(&Aggregate::UpdateCount).unwrap().push((timer.clone(), 10f64));
        expected.get_mut(&Aggregate::Max).unwrap().push((timer.clone(), 84f64));
        expected.get_mut(&Aggregate::Sum).unwrap().push((timer.clone(), 316f64));
        // percentiles( 0.5 and 0.85 ) was counted in google spreadsheets
        expected.get_mut(&Aggregate::Median).unwrap().push((timer.clone(), 23.5f64));
        expected.get_mut(&Aggregate::Percentile(0.85, 85)).unwrap().push((timer.clone(), 60.85f64));
        metrics.push(timer);

        // sets should be properly aggregated into count of uniques and update count
        let mut set = Metric::new(1f64, MetricType::Set(HashSet::new()), None, None).unwrap();

        samples
            .iter()
            .map(|t| {
                let set2 = Metric::new(*t, MetricType::Set(HashSet::new()), None, None).unwrap();
                set.accumulate(set2).unwrap();
            })
            .last();

        expected.get_mut(&Aggregate::Count).unwrap().push((set.clone(), 9f64));
        expected.get_mut(&Aggregate::UpdateCount).unwrap().push((set.clone(), 10f64));
        // percentiles( 0.5 and 0.85 ) was counted in google spreadsheets
        metrics.push(set);

        let mut results = HashMap::new();
        metrics
            .into_iter()
            .map(|metric| {
                let mut calc_metric = metric.clone();
                let calculator = AggregateCalculator::new(&mut calc_metric, &aggregates);
                calculator
                    //                    .inspect(|res| {
                    //dbg!(res);
                    //})
                    // count all of them that are countable (filtering None) and leaving the aggregate itself
                    .filter_map(|result| result)
                    // set corresponding name
                    .map(|(idx, value)| {
                        // it would be better to use metric as a key, but it doesn't implement Eq,
                        // so atm we trick it this way
                        if !results.contains_key(&aggregates[idx]) {
                            results.insert(aggregates[idx].clone(), Vec::new());
                        }
                        results.get_mut(&aggregates[idx]).unwrap().push((metric.clone(), value));
                    })
                    .last();
            })
            .last();

        //dbg!(&expected, &results);
        assert_eq!(expected.len(), results.len(), "expected len does not match results len");
        for (ec, ev) in &expected {
            //dbg!(ec);
            let rv = results.get(ec).expect("expected key not found in results");
            assert_eq!(ev.len(), rv.len());
            // for percentile a value can be a bit not equal
            if ec == &Aggregate::Percentile(0.85f64, 85) {
                ev.iter()
                    .zip(rv.iter())
                    .map(|((_, e), (_, r))| {
                        let diff = (e - r).abs();
                        assert!(diff < 0.0001, format!("{} ~= {}: {}", e, r, diff));
                    })
                    .last();
            } else {
                assert_eq!(ev, rv, "\non {:?}: \n {:?} \n not equal to \n {:?}", &ec, &ev, &rv);
            }
        }
    }
}
