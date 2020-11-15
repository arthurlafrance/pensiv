use ndarray::prelude::*;
use ndarray::Array;

use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::utils::ComparableFloat;


pub fn factorial(n: i32) -> i32 {
    if n < 0 {
        return 0; // panic instead?
    }
    else if n == 0 {
        return 1;
    }
    else {
        return n * factorial(n - 1);
    }
}


pub fn permutations(n: i32, k: i32) -> i32 {
    factorial(n) / factorial(n - k)
}


pub fn choose(n: i32, k: i32) -> i32 {
    if k >= 0 && (n - k) >= 0 { // denominator can't be 0
        factorial(n) / (factorial(n - k) * factorial(k))
    }
    else {
        0
    }
}


/// Base trait for all discrete distributions
pub trait DiscreteDist<N> { // TODO: bound generic type to numerics
    fn pmf(&self, value: N) -> f64;
    fn cdf(&self, value: N) -> f64;

    fn interval_cdf(&self, lower_bound: N, upper_bound: N) -> f64 {
        self.cdf(upper_bound) - self.cdf(lower_bound) // need to subtract one so that you get the entire interval
    }

    fn mean(&self) -> f64;
    fn variance(&self) -> f64;

    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}


#[derive(Debug, PartialEq)]
pub struct DiscreteUniformDist {
    lower_bound: i32,
    upper_bound: i32
}

impl DiscreteUniformDist {
    pub fn new(lower_bound: i32, upper_bound: i32) -> Option<DiscreteUniformDist> {
        if lower_bound > upper_bound {
            return None;
        }

        Some(DiscreteUniformDist { lower_bound, upper_bound })
    }

    pub fn range(&self) -> i32 {
        self.upper_bound - self.lower_bound
    }

    pub fn upper_bound(&self) -> i32 {
        self.upper_bound
    }

    pub fn lower_bound(&self) -> i32 {
        self.lower_bound
    }
}

impl DiscreteDist<i32> for DiscreteUniformDist {
    fn pmf(&self, value: i32) -> f64 {
        if value < self.lower_bound || value > self.upper_bound {
            return 0.0;
        }

        1.0 / self.range() as f64
    }

    fn cdf(&self, value: i32) -> f64 {
        if value < self.lower_bound {
            return 0.0;
        }
        else if value > self.upper_bound {
            return 1.0;
        }

        (value - self.lower_bound + 1) as f64 / self.range() as f64
    }

    // TODO: better interval cdf implementation

    fn mean(&self) -> f64 {
        (self.upper_bound + self.lower_bound) as f64 / 2.0
    }

    fn variance(&self) -> f64 {
        (((self.upper_bound - self.upper_bound + 1) * (self.upper_bound - self.upper_bound + 1)) as f64 - 1.0) / 12.0
    }
}


#[derive(Debug, PartialEq)]
struct BernoulliDist {
    p_success: f64,
}

impl BernoulliDist {
    pub fn new(p_success: f64) -> Option<BernoulliDist> {
        if p_success < 0.0 || p_success > 1.0 {
            return None;
        }

        Some(BernoulliDist { p_success })
    }

    pub fn p_success(&self) -> f64 {
        self.p_success
    }

    pub fn p_failure(&self) -> f64 {
        1.0 - self.p_success
    }
}

impl DiscreteDist<i32> for BernoulliDist {
    fn pmf(&self, value: i32) -> f64 {
        if value == 0 {
            return self.p_failure();
        }
        else if value == 1 {
            return self.p_success;
        }
        else {
            return 0.0;
        }
    }

    fn cdf(&self, value: i32) -> f64 {
        if value < 0 {
            return 0.0;
        }
        else if value == 0 {
            return self.p_failure();
        }
        else {
            return 1.0;
        }
    }

    fn mean(&self) -> f64 {
        self.p_success
    }

    fn variance(&self) -> f64 {
        self.p_success * self.p_failure()
    }
}


#[derive(Debug, PartialEq)]
struct BinomDist {
    p_success: f64,
    trials: i32,
}

impl BinomDist {
    pub fn new(trials: i32, p_success: f64) -> Option<BinomDist> {
        if p_success < 0.0 || p_success > 1.0 {
            return None;
        }

        if trials < 0 {
            return None;
        }

        Some(BinomDist { p_success, trials })
    }

    pub fn p_success(&self) -> f64 {
        self.p_success
    }

    pub fn p_failure(&self) -> f64 {
        1.0 - self.p_success
    }

    pub fn trials(&self) -> i32 {
        self.trials
    }
}

impl DiscreteDist<i32> for BinomDist {
    fn pmf(&self, value: i32) -> f64 {
        choose(self.trials, value) as f64 * self.p_success.powi(value) * self.p_failure().powi(self.trials - value)
    }

    fn cdf(&self, value: i32) -> f64 {
        let mut cdf_values = Array::range(0.0, value as f64 + 1.0, 1.0);

        cdf_values.mapv_inplace(|n| { self.pmf(n as i32) }); // avoid using map because that allocates another array
        cdf_values.sum()
    }

    fn interval_cdf(&self, lower_bound: i32, upper_bound: i32) -> f64 {
        let mut cdf_values = Array::range(lower_bound as f64, upper_bound as f64 + 1.0, 1.0);

        cdf_values.mapv_inplace(|n| { self.pmf(n as i32) }); // avoid using map because that allocates another array
        cdf_values.sum()
    }

    fn mean(&self) -> f64 {
        self.trials as f64 * self.p_success
    }

    fn variance(&self) -> f64 {
        self.trials as f64 * self.p_success * (1.0 - self.p_success)
    }
}


#[derive(Debug, PartialEq)]
struct GeometricDist {
    p_success: f64,
}

impl GeometricDist {
    pub fn new(p_success: f64) -> Option<GeometricDist> {
        if p_success < 0.0 || p_success > 1.0 {
            return None;
        }

        Some(GeometricDist { p_success })
    }

    pub fn p_success(&self) -> f64 {
        self.p_success
    }

    pub fn p_failure(&self) -> f64 {
        1.0 - self.p_success
    }
}

impl DiscreteDist<i32> for GeometricDist {
    fn pmf(&self, value: i32) -> f64 {
        if value <= 0 {
            return 0.0;
        }

        self.p_success * self.p_failure().powi(value - 1)
    }

    fn cdf(&self, value: i32) -> f64 {
        1.0 - self.p_failure().powi(value)
    }

    fn mean(&self) -> f64 {
        1.0 / self.p_success
    }

    fn variance(&self) -> f64 {
        self.p_failure() / self.p_success.powi(2)
    }
}


#[derive(Debug)]
struct EmpiricalDist {
    // NOTE: if this dist is immutable, should cache stats directly
    //       on the other hand, if it's mutable, should add mutating methods
    counts: BTreeMap<ComparableFloat, i32>,
    data: Array<f64, Ix1>,
    data_len: usize,
}

impl EmpiricalDist {
    pub fn new(data: Array<f64, Ix1>) -> Option<EmpiricalDist> {
        let mut counts = BTreeMap::<ComparableFloat, i32>::new();

        for elem in data.iter() {
            match ComparableFloat::new(*elem) {
                Some(f) => {
                    if !counts.contains_key(&f) {
                        counts.insert(f, 0);
                    }

                    counts.insert(f, counts[&f] + 1);
                },
                None => return None,
            }
        }

        let data_len = data.len();

        Some(EmpiricalDist { counts, data, data_len })
    }

    pub fn data(&self) -> &Array<f64, Ix1> {
        &self.data
    }
}

impl DiscreteDist<f64> for EmpiricalDist {
    fn pmf(&self, value: f64) -> f64 {
        match ComparableFloat::new(value) {
            Some(f) => {
                self.counts[&f] as f64 / self.data_len as f64
            },
            None => panic!("Encountered NaN in empirical distribution PMF") // NOTE: needs review
        }
    }

    fn cdf(&self, value: f64) -> f64 {
        let mut cdf = 0.0;

        for key in self.counts.keys() {
            if key.value() <= value {
                cdf += self.counts[key] as f64;
            }
            else {
                break;
            }
        }

        cdf / self.data_len as f64
    }

    fn interval_cdf(&self, lower_bound: f64, upper_bound: f64) -> f64 {
        let mut cdf = 0.0;

        for key in self.counts.keys() {
            if key.value() >= lower_bound && key.value() <= upper_bound {
                cdf += self.counts[key] as f64;
            }
            else if key.value() > upper_bound {
                break;
            }
        }

        cdf / self.data_len as f64
    }

    fn mean(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    fn variance(&self) -> f64 {
        let data_squared = &self.data * &self.data;

        data_squared.mean().unwrap_or(0.0) - self.mean().powi(2)
    }
}


pub trait ContinuousDist<N> { // TODO: bound generic type to numerics
    fn pdf(&self, value: N) -> f64;
    fn cdf(&self, value: N) -> f64;

    fn interval_cdf(&self, lower_bound: N, upper_bound: N) -> f64 {
        self.cdf(upper_bound) - self.cdf(lower_bound)
    }

    fn mean(&self) -> f64;
    fn variance(&self) -> f64;

    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}


#[derive(Debug, PartialEq)]
pub struct ContinuousUniformDist {
    lower_bound: f64,
    upper_bound: f64,
}

impl ContinuousUniformDist {
    pub fn new(lower_bound: f64, upper_bound: f64) -> Option<ContinuousUniformDist> {
        if lower_bound > upper_bound {
            return None;
        }

        Some(ContinuousUniformDist { lower_bound, upper_bound })
    }

    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    pub fn upper_bound(&self) -> f64 {
        self.upper_bound
    }

    pub fn range(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }
}

impl ContinuousDist<f64> for ContinuousUniformDist {
    fn pdf(&self, value: f64) -> f64 {
        if value >= self.lower_bound && value <= self.upper_bound {
            return 1.0 / self.range();
        }
        else {
            return 0.0;
        }
    }

    fn cdf(&self, value: f64) -> f64 {
        if value >= self.lower_bound && value <= self.upper_bound {
            return (value - self.lower_bound) / self.range();
        }
        else if value < self.lower_bound {
            return 0.0;
        }
        else {
            return 1.0;
        }
    }

    // TODO: better interval cdf

    fn mean(&self) -> f64 {
        (self.lower_bound + self.upper_bound) / 2.0
    }

    fn variance(&self) -> f64 {
        (self.upper_bound - self.lower_bound) / 12.0_f64.sqrt()
    }
}


#[derive(Debug, PartialEq)]
struct ExponentialDist {
    rate_param: f64
}

impl ExponentialDist {
    pub fn new(rate_param: f64) -> Option<ExponentialDist> {
        if rate_param <= 0.0 {
            return None;
        }

        Some(ExponentialDist { rate_param })
    }

    pub fn rate_param(&self) -> f64 {
        self.rate_param
    }
}

impl ContinuousDist<f64> for ExponentialDist {
    fn pdf(&self, value: f64) -> f64 {
        self.rate_param * (-self.rate_param * value).exp()
    }

    fn cdf(&self, value: f64) -> f64 {
        1.0 - (-self.rate_param * value).exp()
    }

    fn mean(&self) -> f64 {
        1.0 / self.rate_param
    }

    fn variance(&self) -> f64 {
        1.0 / self.rate_param.powi(2)
    }
}


#[derive(Debug, PartialEq)]
struct NormalDist {
    // NOTE: distributions parameters are so named to reflect their conceptual purpose, rather than their relevance to 
    // the properties of the distribution
    loc: f64,
    scale: f64,
}

impl NormalDist {
    pub fn new(loc: f64, scale: f64) -> Option<NormalDist> {
        if scale < 0.0 {
            return None;
        }

        Some(NormalDist { loc, scale })
    }

    pub fn std() -> NormalDist {
        NormalDist { loc: 0.0, scale: 1.0 }
    }

    pub fn loc(&self) -> f64 {
        self.loc
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn z(&self, value: f64) -> f64 {
        (value - self.loc) / self.scale
    }
}

impl ContinuousDist<f64> for NormalDist {
    fn pdf(&self, value: f64) -> f64 {
        let t1 = 1.0 / (2.0 * PI).sqrt() * self.scale;
        let t2 = -1.0 * (value - self.loc).powi(2) / (2.0 * self.variance());

        t1 * t2.exp()
    }

    fn cdf(&self, value: f64) -> f64 {
        // NOTE: this approximation is currently somewhat naive, will be improved in the future
        1.0 / (1.0 + (-1.65451 * value).exp())
    }

    fn mean(&self) -> f64 {
        self.loc
    }

    fn variance(&self) -> f64 {
        self.scale.powi(2)
    }

    fn std(&self) -> f64 {
        self.scale
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use ndarray::Array;

    #[test]
    fn discrete_uniform_dist_created_correctly() {
        let dist = DiscreteUniformDist::new(0, 4).unwrap(); // panics if creation fails
        
        assert_eq!(dist.lower_bound(), 0);
        assert_eq!(dist.upper_bound(), 4);
        assert_eq!(dist.range(), 4);
    }

    #[test]
    fn discrete_uniform_dist_invalid_creation_fails() {
        let dist = DiscreteUniformDist::new(4, 0);
        assert_eq!(dist, None);
    }
            
    #[test]
    fn discrete_uniform_dist_gives_correct_pmf_inrange() {
        let lower_bound = 0;
        let upper_bound = 4;

        let dist = DiscreteUniformDist::new(lower_bound, upper_bound).unwrap();

        let mut values = Array::range(lower_bound as f64, upper_bound as f64 + 1.0, 1.0);
        values.mapv_inplace(|n| { dist.pmf(n as i32) });

        assert!(values.iter().all(|&p| p == 1.0 / dist.range() as f64));
    }

    #[test]
    fn discrete_uniform_dist_gives_correct_pmf_outofrange() {
        let lower_bound = 0;
        let upper_bound = 4;

        let dist = DiscreteUniformDist::new(lower_bound, upper_bound).unwrap();

        assert_eq!(dist.pmf(lower_bound - 1), 0.0);
        assert_eq!(dist.pmf(upper_bound + 1), 0.0);
    }

    #[test]
    fn discrete_uniform_dist_correct_cdf_outofrange() {
        let lower_bound = 0;
        let upper_bound = 4;

        let dist = DiscreteUniformDist::new(lower_bound, upper_bound).unwrap();

        assert_eq!(dist.cdf(lower_bound - 1), 0.0);
        assert_eq!(dist.cdf(upper_bound + 1), 1.0);
    }

    #[test]
    fn discrete_uniform_dist_correct_cdf_withinrange() {
        let lower_bound = 1;
        let upper_bound = 5;

        let dist = DiscreteUniformDist::new(lower_bound, upper_bound).unwrap();

        let values = Array::range(lower_bound as f64, upper_bound as f64 + 1.0, 1.0);
        let correct_cdf = values.mapv(|n| n as f64 / dist.range() as f64);
        let cdf = values.mapv(|n| dist.cdf(n as i32));

        assert_eq!(cdf, correct_cdf);
    }

    #[test]
    fn discrete_uniform_dist_correct_inverval_cdf() {
        let lower_bound = 1;
        let upper_bound = 5;

        let dist = DiscreteUniformDist::new(lower_bound, upper_bound).unwrap();

        assert_eq!(dist.interval_cdf(lower_bound + 1, upper_bound - 1), (dist.range() - 2) as f64 / dist.range() as f64);
    }

    #[test]
    fn discrete_uniform_dist_mean_calculated_correctly() {
        let lower_bound = 1;
        let upper_bound = 5;

        let dist = DiscreteUniformDist::new(lower_bound, upper_bound).unwrap();

        assert_eq!(dist.mean(), (upper_bound + lower_bound) as f64 / 2.0);
    }

        // variance (i'm literally just gonna plug in a formula but ok then)

        // std (again literally just gonna plug in a formula so idk if it's worth testing)
    #[test]
    fn bernoulli_dist_created_correctly() {
        let p = 0.5;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.p_success(), p);
        assert_eq!(dist.p_failure(), 1.0 - p);
    }

    #[test]
    fn bernoulli_dist_invalid_creation_fails() {
        let p = -0.5;
        let dist = BernoulliDist::new(p);

        assert_eq!(dist, None);
    }

    #[test]
    fn bernoulli_dist_correct_pmf_inrange() {
        let p = 0.5;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.pmf(0), dist.p_failure());
        assert_eq!(dist.pmf(1), dist.p_success());
    }

    #[test]
    fn bernoulli_dist_correct_pmf_outofrange() {
        let p = 0.5;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.pmf(-1), 0.0);
        assert_eq!(dist.pmf(2), 0.0);
    }

    #[test]
    fn bernoulli_dist_correct_cdf_inrange() {
        let p = 0.5;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.cdf(0), dist.p_failure());
        assert_eq!(dist.cdf(1), 1.0);
    }

    #[test]
    fn bernoulli_dist_correct_cdf_outofrange() {
        let p = 0.5;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.cdf(-1), 0.0);
        assert_eq!(dist.cdf(2), 1.0);
    }

    // i'm a little mathematically unsure of this one
    // #[test]
    // fn bernoulli_dist_correct_interval_cdf() {
    //     let p = 0.5;
    //     let dist = BernoulliDist::new(p).unwrap();

    //     assert_eq!(dist.interval_cdf(0, 1), 1.0);
    // }

    #[test]
    fn bernoulli_dist_mean_calculated_correctly() {
        let p = 0.4;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.mean(), p);
    }

    #[test]
    fn bernoulli_dist_variance_calculated_correctly() {
        let p = 0.4;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.variance(), p * (1.0 - p));
    }

    #[test]
    fn bernoulli_dist_std_dev_calculated_correctly() {
        let p = 0.4;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.std(), (p * (1.0 - p)).sqrt());
    }

    fn binom_dist_created_correctly() {
        let n = 4;
        let p = 0.4;

        let dist = BinomDist::new(n, p).unwrap();

        assert_eq!(dist.trials(), n);
        assert_eq!(dist.p_success(), p);
        assert_eq!(dist.p_failure(), 1.0 - p);
    }

    #[test]
    fn binom_dist_invalid_p_creation_fails() {
        let n = 4;
        let p = -0.5;

        let dist = BinomDist::new(n, p);

        assert_eq!(dist, None);
    }

    #[test]
    fn binom_dist_invalid_trials_creation_fails() {
        let n = -1;
        let p = 0.5;

        let dist = BinomDist::new(n, p);

        assert_eq!(dist, None);
    }

    #[test]
    fn binom_dist_correct_pmf_inrange() {
        let n = 4;
        let p = 0.4;

        let dist = BinomDist::new(n, p).unwrap();
        let pmf = Array::range(0.0, n as f64 + 1.0, 1.0).mapv(|k| dist.pmf(k as i32));

        assert!(pmf.all_close(&array![0.1296, 0.3456, 0.3456, 0.1536, 0.0256], 1e-10));
    }

    #[test]
    fn binom_dist_correct_pmf_outofrange() {
        let n = 4;
        let p = 0.4;

        let dist = BinomDist::new(n, p).unwrap();

        assert_eq!(dist.pmf(-1), 0.0);
        assert_eq!(dist.pmf(5), 0.0);
    }

        // cdf
    #[test]
    fn binom_dist_correct_cdf_inrange() {
        let n = 4;
        let p = 0.4;

        let dist = BinomDist::new(n, p).unwrap();

        let v = 4.0;
        let cdf = Array::range(0.0, v, 1.0).mapv(|k| dist.pmf(k as i32)).sum();

        assert_eq!(dist.cdf(v as i32 - 1), cdf);
    }

    #[test]
    fn binom_dist_correct_cdf_outofrange() {
        let n = 4;
        let p = 0.4;

        let dist = BinomDist::new(n, p).unwrap();

        assert!((dist.cdf(n + 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn binom_dist_correct_interval_cdf() {
        let n = 4;
        let p = 0.4;

        let dist = BinomDist::new(n, p).unwrap();

        let a = 2.0;
        let b = 3.0;
        let cdf = Array::range(a, b + 1.0, 1.0).mapv(|k| dist.pmf(k as i32)).sum();

        assert_eq!(dist.interval_cdf(a as i32, b as i32), cdf);
    }

    #[test]
    fn binom_dist_mean_calculated_correctly() {
        let n = 4;
        let p = 0.4;

        let dist = BinomDist::new(n, p).unwrap();

        assert_eq!(dist.mean(), n as f64 * p);
    }

    #[test]
    fn binom_dist_variance_calculated_correctly() {
        let n = 4;
        let p = 0.4;

        let dist = BinomDist::new(n, p).unwrap();

        assert_eq!(dist.variance(), n as f64 * p * (1.0 - p));
    }

    #[test]
    fn geometric_dist_valid_created_correctly() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        assert_eq!(dist.p_success(), p);
        assert_eq!(dist.p_failure(), 1.0 - p);
    }

    #[test]
    fn geometric_dist_invalid_p_creation_fails() {
        let p = -0.4;
        let dist = GeometricDist::new(p);

        assert_eq!(dist, None);
    }

    #[test]
    fn geometric_dist_correct_pmf_inrange() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        let pmfs = Array::range(1.0, 4.0, 1.0).mapv(|k| dist.pmf(k as i32));
        assert!(pmfs.all_close(&array![0.4, 0.24, 0.144], 1e10));
    }

    #[test]
    fn geometric_dist_correct_pmf_outofrange() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        assert_eq!(dist.pmf(0), 0.0);
    }

    #[test]
    fn geometric_dist_correct_cdf_inrange() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        let n = 3.0;
        let cdf = Array::range(1.0, n + 1.0, 1.0).mapv(|k| dist.pmf(k as i32)).sum();

        assert_eq!(dist.cdf(n as i32), cdf);
    }

    #[test]
    fn geometric_dist_correct_interval_cdf() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        let a = 3.0;
        let b = 5.0;
        let cdf = Array::range(a, b + 1.0, 1.0).mapv(|k| dist.pmf(k as i32)).sum();

        assert_eq!(dist.interval_cdf(a as i32, b as i32), cdf);
    }
        // interval_cdf
        // mean
        // variance
        // std
    // Empirical
        // new
            // good
            // bad float
        // pmf
        // cdf
        // interval_cdf
        // mean
        // variance
        // std

    // ContinuousUniform
        // new
            // good
            // bad bounds
        // pdf
        // cdf
        // interval_cdf
        // mean
        // variance
        // std
    // Exponential
        // new
            // good
            // bad rate param
        // pdf
        // cdf
        // interval_cdf
        // mean
        // variance
        // std
    // Normal
        // new
            // good
            // bad scale
        // std
        // z
        // pdf
        // cdf
        // interval_cdf
        // mean
        // variance
        // std
}