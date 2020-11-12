use ndarray::prelude::*;
use ndarray::Array;

use std::collections::BTreeMap;

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
pub trait DiscreteDist<Value> { // TODO: bound generic type to numerics
    fn pmf(&self, value: Value) -> f64;
    fn cdf(&self, value: Value) -> f64;

    fn interval_cdf(&self, lower_bound: Value, upper_bound: Value) -> f64 {
        self.cdf(upper_bound) - self.cdf(lower_bound)
    }

    fn mean(&self) -> f64;
    fn variance(&self) -> f64;

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}


pub struct DiscreteUniformDist {
    lower_bound: i32,
    upper_bound: i32
}

impl DiscreteUniformDist {
    pub fn new(lower_bound: i32, upper_bound: i32) -> DiscreteUniformDist {
        if lower_bound > upper_bound {
            panic!("Upper bound of discrete uniform distribution can't be less than lower bound");
        }

        DiscreteUniformDist { lower_bound, upper_bound }
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
    pub fn new(p_success: f64) -> BernoulliDist {
        if p_success < 0.0 || p_success > 1.0 {
            panic!("Bernoulli probability of success must be between 0 and 1");
        }

        BernoulliDist { p_success }
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
    pub fn new(p_success: f64, trials: i32) -> BinomDist {
        if p_success < 0.0 || p_success > 1.0 {
            panic!("Binomial probability of success must be between 0 and 1");
        }

        if trials < 0 {
            panic!("Binomial number of trials must be non-negative");
        }

        BinomDist { p_success, trials }
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
    pub fn new(p_success: f64) -> GeometricDist {
        if p_success < 0.0 || p_success > 1.0 {
            panic!("Geometric probability of success must be between 0 and 1");
        }

        GeometricDist { p_success }
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
    pub fn new(data: Array<f64, Ix1>) -> EmpiricalDist {
        let mut counts = BTreeMap::<ComparableFloat, i32>::new();

        for elem in data.iter() {
            match ComparableFloat::new(*elem) {
                Some(f) => {
                    if !counts.contains_key(&f) {
                        counts.insert(f, 0);
                    }

                    counts.insert(f, counts[&f] + 1);
                },
                None => panic!("Encountered NaN in empirical distribution dataset")
            }
        }

        let data_len = data.len();

        EmpiricalDist { counts, data, data_len }
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
            None => panic!("Encountered NaN in empirical distribution PMF")
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


#[cfg(test)]
mod tests {
    // TODO: add tests
}