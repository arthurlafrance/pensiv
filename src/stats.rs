use ndarray::prelude::*;
use ndarray::Array;

use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::utils::ComparableFloat;


pub fn factorial(n: i32) -> i32 {
    if n < 0 {
        return 0;
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
    factorial(n) / (factorial(n - k) * factorial(k))
}


/// Base trait for all discrete distributions
pub trait DiscreteDist<N> { // TODO: bound generic type to numerics
    fn pmf(&self, value: N) -> f64;
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
}

impl DiscreteDist<i32> for BernoulliDist {
    fn pmf(&self, value: i32) -> f64 {
        if value == 0 {
            return 1.0 - self.p_success;
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
            return 1.0 - self.p_success;
        }
        else {
            return 1.0;
        }
    }

    fn mean(&self) -> f64 {
        self.p_success
    }

    fn variance(&self) -> f64 {
        self.p_success * (1.0 - self.p_success)
    }
}


struct BinomDist {
    p_success: f64,
    trials: i32,
}

impl BinomDist {
    pub fn new(p_success: f64, trials: i32) -> Option<BinomDist> {
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
        // TODO: vectorize with ndarray
        let mut cdf_value = 0.0;

        for n in 0..(value + 1) {
            cdf_value += self.pmf(n);
        }

        cdf_value
    }

    fn interval_cdf(&self, lower_bound: i32, upper_bound: i32) -> f64 {
        let mut cdf_value = 0.0;

        for n in lower_bound..(upper_bound + 1) {
            cdf_value += self.pmf(n);
        }

        cdf_value
    }

    fn mean(&self) -> f64 {
        self.trials as f64 * self.p_success
    }

    fn variance(&self) -> f64 {
        self.trials as f64 * self.p_success * (1.0 - self.p_success)
    }
}


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
        self.p_success * self.p_failure().powi(value - 1)
    }

    fn cdf(&self, value: i32) -> f64 {
        1.0 - self.p_failure().powi(value)
    }

    fn interval_cdf(&self, lower_bound: i32, upper_bound: i32) -> f64 {
        self.cdf(upper_bound) - self.cdf(lower_bound)
    }

    fn mean(&self) -> f64 {
        1.0 / self.p_success
    }

    fn variance(&self) -> f64 {
        self.p_failure() / self.p_success.powi(2)
    }
}


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
    // TODO: add tests
}