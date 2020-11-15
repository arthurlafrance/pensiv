//! Commonly-used probability & statistics functionality.
//! 
//! Currently implemented are the properties of several common discrete and continuous distributions, as well as the 
//! basic combinatorics required to implement them.

use ndarray::prelude::*;
use ndarray::Array;

use num_traits::Num;
use num_traits::identities;

use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::utils::ComparableFloat;

/// Returns n!.
/// 
/// Note that the function returns 0 if n < 0.
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


/// Returns the number of k-permutations of n.
pub fn permutations(n: i32, k: i32) -> i32 {
    factorial(n) / factorial(n - k) // TODO: fix if n - k < 0
}


/// Returns n choose k.
///
/// Note that the function returns 0 if k < 0 or n - k < 0.
pub fn choose(n: i32, k: i32) -> i32 {
    if k >= 0 && (n - k) >= 0 { // denominator can't be 0
        factorial(n) / (factorial(n - k) * factorial(k))
    }
    else {
        0
    }
}


/// Base trait for all discrete distributions.
///
/// The `DiscreteDist` trait provides a general interface for distributions of discrete random variables, including PMF, CDF, mean/expectation, 
/// variance, and standard deviation. Default implementations of CDF of an interval and standard deviation are provided.
/// 
/// `DiscreteDist` is parametrized by one generic type, `N`, which must conform to the Num trait in the `num-traits` crate; 
/// it represents the type of the distribution's support.
pub trait DiscreteDist<N: Num> {
    /// Returns the probability mass function (PMF) of `value`; varies by distribution. Convention is to return `0.0` for values that are outside the 
    /// distribution's support.
    fn pmf(&self, value: N) -> f64;

    /// Returns the cumulative density function (CDF) of `value`; varies by distribution.
    fn cdf(&self, value: N) -> f64; // default implementation?

    /// Returns the probability that the discrete random variable described by this distribution will fall within the interval 
    /// `lower_bound <= n <= upper_bound` for some n value of the random variable, i.e. the "interval CDF".
    ///
    /// The default implementation of this function simply calculates the difference between the CDFs of `upper_bound` and 
    /// `lower_bound - 1`.
    fn interval_cdf(&self, lower_bound: N, upper_bound: N) -> f64 {
        self.cdf(upper_bound) - self.cdf(lower_bound - identities::one()) // need to subtract one so that you get the entire interval
    }

    /// Returns the mean (expectation) of the distribution.
    fn mean(&self) -> f64;

    /// Returns the variance (average square distance from the mean) of the distribution.
    fn variance(&self) -> f64;

    /// Returns the standard deviation of the distribution. The default implementation of this function returns the square
    /// root of the distribution's variance.
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}


/// A discrete uniform distribution.
///
/// A `DiscreteUniformDist` is described by its support's lower and upper bounds; all integer values within that range (inclusive)
/// have equal likelihood of occurring.
#[derive(Debug, PartialEq)]
pub struct DiscreteUniformDist {
    lower_bound: i32,
    upper_bound: i32
}

impl DiscreteUniformDist {
    /// Creates a new discrete uniform distribution with the given lower and upper bounds.
    ///
    /// Returns `None` if `lower_bound > upper_bound`, otherwise returns `Some` containing 
    /// the created `DiscreteUniformDist`.
    ///
    /// ```rust
    /// let a = 1;
    /// let b = 5;
    /// 
    /// let dist = DiscreteUniformDist::new(a, b).unwrap();
    /// ```
    ///
    /// ```rust
    /// let a = 5;
    /// let b = 1;
    /// 
    /// match DiscreteUniformDist::new(a, b) {
    ///     Some(_) => println!("got the distribution!"),
    ///     None => println!("creation failed"),
    /// } // prints "creation failed"
    /// ```
    pub fn new(lower_bound: i32, upper_bound: i32) -> Option<Self> {
        if lower_bound > upper_bound {
            return None;
        }

        Some(DiscreteUniformDist { lower_bound, upper_bound })
    }

    /// Returns the range of the distribution, i.e. the length of the support.
    /// 
    /// ```rust
    /// let a = 1;
    /// let b = 5;
    /// 
    /// let dist = DiscreteUniformDist::new(a, b).unwrap();
    /// println!("{}", dist.range()); // prints "5"
    /// ```
    pub fn range(&self) -> i32 {
        self.upper_bound - self.lower_bound + 1
    }

    /// Returns the upper bound of the distribution's support
    pub fn upper_bound(&self) -> i32 {
        self.upper_bound
    }

    /// Returns the lower bound of the distribution's support
    pub fn lower_bound(&self) -> i32 {
        self.lower_bound
    }
}

impl DiscreteDist<i32> for DiscreteUniformDist {
    /// Returns the PMF of `value` within the distribution.
    ///
    /// This method returns `1.0 / range` (where `range` is the range of the distributrion) uniformly for all elements 
    /// in the support, and returns `0.0` uniformly for elements outside of the support.
    ///
    /// ```rust
    /// let a = 1;
    /// let b = 5;
    /// let dist = DiscreteUniformDist::new(a, b).unwrap();
    /// 
    /// for i in a..(b + 1) {
    ///     println!("{}", dist.pmf(i)); // prints "0.2" 5 times
    /// }
    /// 
    /// println!("{}", dist.pmf(2 * b)); // prints "0"
    /// ```
    fn pmf(&self, value: i32) -> f64 {
        if value < self.lower_bound || value > self.upper_bound {
            return 0.0;
        }

        1.0 / self.range() as f64
    }

    /// Returns the CDF of `value` within the distribution.
    ///
    /// For values below the support, `0.0` is returned. For values above the support, `1.0` is returned. For values within 
    /// the support, the fraction of the support that's at or below `value` is returned, i.e. `(value - lower bound + 1) / range`.
    ///
    /// ```rust
    /// let a = 0;
    /// let b = 4;
    /// let dist = DiscreteUniformDist::new(a, b).unwrap();
    ///
    /// for i in a..(b + 1) {
    ///     // prints "0.2", "0.4", "0.6", "0.8", "1.0" (separated by newlines)
    ///     println!("{}", dist.cdf(i));
    /// }
    /// 
    /// println!("{}", dist.cdf(a - 1)); // prints "0.0"
    /// println!("{}", dist.cdf(b + 1)); // prints "1.0"
    /// ```
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

    /// Returns the mean of the uniform distribution, i.e. the average of the upper and lower bounds of the support
    fn mean(&self) -> f64 {
        (self.upper_bound + self.lower_bound) as f64 / 2.0
    }

    /// Returns the variance of the uniform distribution.
    ///
    /// The variance is calculated according to the following formula:
    /// `((upper bound - lower bound + 1)^2 - 1) / 12`
    fn variance(&self) -> f64 {
        (((self.upper_bound - self.upper_bound + 1) * (self.upper_bound - self.upper_bound + 1)) as f64 - 1.0) / 12.0
    }
}


/// A distribution describing a Bernoulli ("indicator") random variable.
///
/// Bernoulli distributions are parametrized by `p`, the probability that a "success" will occur; or equivalently the probability 
/// that the random variable will be the value `1`. The support of a Bernoulli random variable, therefore, is `{0, 1}`; the variable
/// essentially "indicates" a 1 with probability `p`.
#[derive(Debug, PartialEq)]
pub struct BernoulliDist {
    p_success: f64,
}

impl BernoulliDist {
    /// Creates and returns a new Bernoulli distribution with parameter `p = p_success`.
    ///
    /// Note that `p_success` must be a valid probability, so the function returns None if it's invalid, i.e. if 
    /// `p_success < 0` or `p_success > 1`. If `p_success` is valid, however, the created distribution is returned.
    ///
    /// ```rust
    /// let p = 0.4;
    /// let dist = BernoulliDist::new(p).unwrap();
    /// 
    /// println!("{}", dist.p_success()); // prints "0.4"
    /// ```
    ///
    /// ```rust
    /// let p1 = -1.0;
    /// match BernoulliDist::new(p1) {
    ///     Some(_) => println!("got it!"),
    ///     None => println!("don't got it!"),
    /// }
    ///
    /// let p2 = 1.7;
    /// match BernoulliDist::new(p2) {
    ///     Some(_) => println!("got it!"),
    ///     None => println!("don't got it!"),
    /// }
    /// 
    /// // both print "don't got it!"
    /// ```
    pub fn new(p_success: f64) -> Option<Self> {
        if p_success < 0.0 || p_success > 1.0 {
            return None;
        }

        Some(BernoulliDist { p_success })
    }

    /// Returns the probability of the Bernoulli random variable indicating success, i.e. the probability of the value `1`.
    ///
    /// Note that this is the same as the value of the parameter.
    pub fn p_success(&self) -> f64 {
        self.p_success
    }

    /// Returns the probability of the Bernoulli random variable indicating failure, i.e. the probability of the value `0`.
    ///
    /// Note that this is the same as `1 - p`, where `p` is the value of the parameter.
    pub fn p_failure(&self) -> f64 {
        1.0 - self.p_success
    }
}

impl DiscreteDist<i32> for BernoulliDist {
    /// Returns the Bernoulli PMF of `value`.
    ///
    /// Returns `p` for `value = 1`, `1 - p` for `value = 0`, and `0.0` otherwise (where `p` is the value of the parameter).
    ///
    /// ```rust
    /// let p = 0.4;
    /// let dist = BernoulliDist::new(p).unwrap();
    /// 
    /// println!("{}", dist.pmf(-1)); // prints "0.0"
    /// println!("{}", dist.pmf(0)); // prints "0.6"
    /// println!("{}", dist.pmf(1)); // prints "0.4"
    /// println!("{}", dist.pmf(2)); // prints "0.0"
    /// ```
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

    /// Returns the Bernoulli CDF of `value`.
    ///
    /// Returns `0.0` for values below the support (i.e. negative values), `1 - p` for `value = 0`, and `1.0` otherwise, 
    /// including for `value = 1.0` (where `p` is the value of the parameter).
    ///
    /// ```rust
    /// let p = 0.4;
    /// let dist = BernoulliDist::new(p).unwrap();
    /// 
    /// println!("{}", dist.cdf(-1)); // prints "0.0"
    /// println!("{}", dist.cdf(0)); // prints "0.6"
    /// println!("{}", dist.cdf(1)); // prints "1.0"
    /// println!("{}", dist.cdf(2)); // prints "1.0"
    /// ```
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

    /// Returns the mean of the Bernoulli random variable, equal to the parameter `p`.
    fn mean(&self) -> f64 {
        self.p_success
    }

    /// Returns the variance of the Bernoulli random variable, equal to `p(1 - p)` where `p` is the value of the parameter.
    fn variance(&self) -> f64 {
        self.p_success * self.p_failure()
    }
}


/// A binomial distribution, representing the number of successes that arise from some number of trials.
///
/// Binomial distributions are parameterized by `p`, the probability that each trial will be a success (i.e. the Bernoulli probability of each trial), 
/// and by `n`, the number of trials to perform. The support of the binomial distribution is the (infinite) set of non-negative integers.
#[derive(Debug, PartialEq)]
pub struct BinomDist {
    p_success: f64,
    trials: i32,
}

impl BinomDist {
    /// Creates and returns a new binomial distribution with parameters `n = trials` and `p = p_success`.
    ///
    /// Returns `None` if `p_success` is not a valid probability or if `trials < 0`, otherwise returns the created distribution.
    ///
    /// ```rust
    /// // rolling two fair coins 5 times, counting the number of times both are heads
    /// let p = 0.25;
    /// let n = 5;
    ///
    /// let dist = BinomDist::new(n, p).unwrap();
    /// println!("{}", dist.p_success()); // prints "0.25"
    /// println!("{}", dist.p_failure()); // prints "0.75"
    /// println!("{}", dist.trials()); // prints "5"
    /// ```
    ///
    /// ```rust
    /// let p = -0.25;
    /// let n = 5;
    ///
    /// let dist = BinomDist::new(n, p);
    /// println!("{}", dist == None); // prints "true"
    /// ```
    ///
    /// ```rust
    /// let p = 1.25;
    /// let n = 5;
    ///
    /// let dist = BinomDist::new(n, p);
    /// println!("{}", dist == None); // prints "true"
    /// ```
    ///
    /// ```rust
    /// let p = 0.25;
    /// let n = -5;
    ///
    /// let dist = BinomDist::new(n, p);
    /// println!("{}", dist == None); // prints "true"
    /// ```
    pub fn new(trials: i32, p_success: f64) -> Option<Self> {
        if p_success < 0.0 || p_success > 1.0 {
            return None;
        }

        if trials < 0 {
            return None;
        }

        Some(BinomDist { p_success, trials })
    }

    /// Returns the probability that each trial will succeed, i.e. the Bernoulli probability of each trial or the value of the parameter `p`.
    pub fn p_success(&self) -> f64 {
        self.p_success
    }

    /// Returns the probability that each trial will fail.
    pub fn p_failure(&self) -> f64 {
        1.0 - self.p_success
    }

    /// Returns the number of trials.
    pub fn trials(&self) -> i32 {
        self.trials
    }
}

impl DiscreteDist<i32> for BinomDist {
    /// Returns the binomial PMF of `value`.
    ///
    /// The binomial PMF is calculated as:
    /// `(n choose k)(p)^k(1 - p)^(n - k)` where `n` is the number of trials and `k = value`
    ///
    /// Note that because of how `choose()` is implemented, this function will (correctly) return `0.0` for values outside of
    /// the support of the distribution, i.e. negative values.
    ///
    /// ```rust
    /// // flipping a biased coin twice, counting the number of heads
    /// let n = 2;
    /// let p = 0.4;
    /// let dist = BinomDist::new(n, p).unwrap();
    ///
    /// for k in 0..3 {
    ///     // prints "0.36", "0.48", "0.16" separated by newlines
    ///     println!("{}", dist.pmf(k));
    /// }
    /// ```
    fn pmf(&self, value: i32) -> f64 {
        choose(self.trials, value) as f64 * self.p_success.powi(value) * self.p_failure().powi(self.trials - value)
    }

    /// Returns the binomial CDF of `value`.
    ///
    /// The binomial CDF is the sum of the binomial PDF of values from `0` to `value` (inclusive). Note that the CDF of 
    /// negative values returns `0.0`.
    ///
    /// ```rust
    /// // flipping a biased coin twice, counting the number of heads
    /// let n = 2;
    /// let p = 0.4;
    /// let dist = BinomDist::new(n, p).unwrap();
    ///
    /// for k in 0..3 {
    ///     // prints "0.36", "0.84", "1.0" separated by newlines
    ///     println!("{}", dist.cdf(k));
    /// }
    /// ```
    fn cdf(&self, value: i32) -> f64 {
        let mut cdf_values = Array::range(0.0, value as f64 + 1.0, 1.0);

        cdf_values.mapv_inplace(|n| { self.pmf(n as i32) }); // avoid using map because that allocates another array
        cdf_values.sum()
    }

    /// Returns the probability that the binomial random variable falls between `lower_bound` and `upper_bound`, inclusive.
    ///
    /// ```rust
    /// // flipping a biased coin three times, counting the number of heads
    /// let n = 3;
    /// let p = 0.4;
    /// let dist = BinomDist::new(n, p).unwrap();
    ///
    /// println!("{}", dist.interval_cdf(1, 2)); // prints "0.72"
    /// ```
    fn interval_cdf(&self, lower_bound: i32, upper_bound: i32) -> f64 {
        let mut cdf_values = Array::range(lower_bound as f64, upper_bound as f64 + 1.0, 1.0);

        cdf_values.mapv_inplace(|n| { self.pmf(n as i32) }); // avoid using map because that allocates another array
        cdf_values.sum()
    }

    /// Returns the mean of the binomial distribution.
    ///
    /// The mean of a binomial distribution is equivalent to `np`, i.e. the number of trials multiplied by the probability that 
    /// each trial will succeed.
    fn mean(&self) -> f64 {
        self.trials as f64 * self.p_success
    }

    /// Returns the variance of the binomial distribution.
    ///
    /// The variance of a binomial distribution is equivalent to `np(1 - p)`, i.e. the number of trials multiplied by the variance of the Bernoulli 
    /// random variable for each trial.
    fn variance(&self) -> f64 {
        self.trials as f64 * self.p_success * (1.0 - self.p_success)
    }
}


/// A geometric distribution, representing the number of trials required to achieve 1 success.
/// 
/// Geometric distributions are parameterized solely by `p`, the probability that each trial will be a success. The support 
/// of the geometric distribution is the (infinite) set of non-negative integers.
#[derive(Debug, PartialEq)]
pub struct GeometricDist {
    p_success: f64,
}

impl GeometricDist {
    /// Creates and returns a new geometric distribution parameterized by `p_success`.
    ///
    /// Returns `None` if `p_success` isn't a valid probability, otherwise returns the created distribution.
    ///
    /// ```rust
    /// // Flipping a biased coin until it comes up heads
    /// let p = 0.4;
    /// let dist = GeometricDist::new(p).unwrap();
    /// 
    /// println!("{}", dist.p_success()); // prints "0.4"
    /// println!("{}", dist.p_failure()); // prints "0.6"
    /// ```
    ///
    /// ```rust
    /// let p = -0.4;
    /// let dist = GeometricDist::new(p);
    /// 
    /// println!("{}", dist == None); // prints "true"
    /// ```
    ///
    /// ```rust
    /// let p = 1.4;
    /// let dist = GeometricDist::new(p);
    /// 
    /// println!("{}", dist == None); // prints "true"
    /// ```
    pub fn new(p_success: f64) -> Option<GeometricDist> {
        if p_success < 0.0 || p_success > 1.0 {
            return None;
        }

        Some(GeometricDist { p_success })
    }

    /// Returns the probability that each trial will be a success.
    pub fn p_success(&self) -> f64 {
        self.p_success
    }

    /// Returns the probability that each trial will be a failure.
    pub fn p_failure(&self) -> f64 {
        1.0 - self.p_success
    }
}

impl DiscreteDist<i32> for GeometricDist {
    /// Returns the geometric PMF of `value`.
    ///
    /// Returns `0.0` for `value <= 0`, otherwise returns:
    /// `p(1 - p)^(k - 1)` where `k = value`
    ///
    /// ```rust
    /// // Flipping a biased coin until it comes up heads
    /// let p = 0.4;
    /// let dist = GeometricDist::new(p).unwrap();
    /// 
    /// for k in 0..4 {
    ///     // prints "0.0", "0.4", "0.24", "0.144"
    ///     println!("{}", dist.pmf(k));
    /// }
    /// ```
    fn pmf(&self, value: i32) -> f64 {
        if value <= 0 {
            return 0.0;
        }

        self.p_success * self.p_failure().powi(value - 1)
    }

    /// Returns the geometric CDF of `value`.
    ///
    /// Returns `0.0` for `value <= 0`, otherwise returns the probability that the geometric random variable is less than 
    /// or equal to `value`.
    ///
    /// ```rust
    /// // Flipping a biased coin until it comes up heads
    /// let p = 0.4;
    /// let dist = GeometricDist::new(p).unwrap();
    /// 
    /// for k in 0..4 {
    ///     // prints "0.0", "0.4", "0.64", "0.784"
    ///     println!("{}", dist.cdf(k));
    /// }
    /// ```
    fn cdf(&self, value: i32) -> f64 {
        if value <= 0 {
            return 0.0;
        }
        
        1.0 - self.p_failure().powi(value)
    }

    /// Returns the mean of the geometric distribution.
    /// 
    /// The mean of a geometric distribution is equivalent to the inverse of the parameter `p`, i.e. `1 / p`.
    fn mean(&self) -> f64 {
        1.0 / self.p_success
    }

    /// Returns the variance of the geometric distribution.
    /// 
    /// The variance of a geometric distribution is equivalent to `(1 - p) / p^2`.
    fn variance(&self) -> f64 {
        self.p_failure() / self.p_success.powi(2)
    }
}


/// An empirical distribution describing a set of data.
///
/// Empirical distributions are "parameterized" by a data set containing real numbers; they describe the distribution of values within that data set. 
/// Therefore, the support of the empirical distribution is the set of real numbers present in the data.
#[derive(Debug)]
pub struct EmpiricalDist {
    // NOTE: if this dist is immutable, should cache stats directly
    //       on the other hand, if it's mutable, should add mutating methods
    counts: BTreeMap<ComparableFloat, i32>,
    data: Array<f64, Ix1>,
    data_len: usize,
}

impl EmpiricalDist {
    /// Creates and returns a new empirical distribution describing `data`.
    ///
    /// Note that `data` must be a 1-dimensional instance of `ndarray::Array` containing the dataset. To create the dataset, `data` is borrowed immutably; 
    /// because it's borrowed rather than moved, it can continue to be used after the distribution is created. If any element 
    /// in `data` is `NaN` or infinite, `None` is returned. Otherwise, the created empirical distribution instance is returned.
    ///
    /// ```rust
    /// let data = array![1.0, 2.0, 2.0, 3.0];
    /// let dist = EmpiricalDist::new(&data).unwrap();
    /// 
    /// println!("{}", dist.data() == data); // prints "true"
    /// ```
    /// 
    /// ```rust
    /// let data = array![1.0, 2.0, 3.0, 1.0 / 0.0, f64::NAN];
    /// match EmpiricalDist::new(&data) {
    ///     Some(_) => println!("how'd that happen?"),
    ///     None => println!("that looks right"),
    /// } // prints "that looks right"
    /// ```
    pub fn new(dataset: &Array<f64, Ix1>) -> Option<EmpiricalDist> {
        let mut counts = BTreeMap::<ComparableFloat, i32>::new();
        let data_len = dataset.len();
        let mut data = Array::<f64, Ix1>::zeros(data_len);

        for (i, elem) in dataset.iter().enumerate() {
            match ComparableFloat::new(*elem) {
                Some(f) => {
                    if !counts.contains_key(&f) {
                        counts.insert(f, 0);
                    }

                    counts.insert(f, counts[&f] + 1);
                    data[[i]] = *elem;
                },
                None => return None,
            }
        }

        Some(EmpiricalDist { counts, data, data_len })
    }

    /// Returns a reference to the distribution's data.
    /// 
    /// Note that this array will be a copy of the array that was originally provided when the distribution was created due 
    /// to the constructor's implementation.
    pub fn data(&self) -> &Array<f64, Ix1> {
        &self.data
    }
}

impl DiscreteDist<f64> for EmpiricalDist {
    /// Returns the empirical PMF of `value`.
    /// 
    /// The empirical PMF of a value is equivalent to the fraction of the data set that it occupies, i.e. the number of 
    /// occurrences of `value` in the data set divided by the number of element in the data set. For values that are not 
    /// present in the data set, the empirical PMF returned is `0.0`.
    /// 
    /// ```rust
    /// let data = array![1.0, 2.0, 2.0, 3.0];
    /// let dist = EmpiricalDist::new(&data).unwrap();
    /// 
    /// println!("{}", dist.pmf(1.0)); // prints "0.25"
    /// println!("{}", dist.pmf(2.0)); // prints "0.5"
    /// println!("{}", dist.pmf(3.0)); // prints "0.25"
    /// ```
    fn pmf(&self, value: f64) -> f64 {
        match ComparableFloat::new(value) {
            Some(f) => {
                if self.counts.contains_key(&f) {
                    self.counts[&f] as f64 / self.data_len as f64
                }
                else {
                    0.0
                }
            },
            None => 0.0,
        }
    }

    /// Returns the empirical CDF of `value`.
    /// 
    /// The empirical CDF of a value is equivalent to the sum of the empirical PMFs of all values in the data set that are 
    /// less than or equal to `value`.
    /// 
    /// ```rust
    /// let data = array![1.0, 2.0, 2.0, 3.0];
    /// let dist = EmpiricalDist::new(&data).unwrap();
    ///
    /// println!("{}", dist.cdf(1.0)); // prints "0.25"
    /// println!("{}", dist.cdf(2.0)); // prints "0.75"
    /// println!("{}", dist.cdf(3.0)); // prints "1.0"
    /// ```
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

    /// Returns the probability that a randomly chosen element of the data set will fall between `lower_bound` and `upper_bound`, inclusive.
    /// 
    /// This is equivalent to the sum of the PMFs of the elements of the data set that are in the interval `lower_bound <= element <= upper_bound`.
    /// 
    /// ```rust
    /// let data = array![1.0, 2.0, 2.0, 3.0];
    /// let dist = EmpiricalDist::new(&data).unwrap();
    ///
    /// println!("{}", dist.interval_cdf(1.0, 2.0)); // prints "0.75"
    /// println!("{}", dist.interval_cdf(2.0, 4.0)); // prints "0.75"
    /// ```
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

    /// Returns the mean of the empirical distribution.
    /// 
    /// The mean of an empirical distribution is equivalent to the mean of the data set.
    fn mean(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    /// Returns the variance of the empirical distribution.
    /// 
    /// The variance of an empirical distribution is equivalent to the difference between the mean of the squared data set 
    /// and the squared mean of the data set.
    fn variance(&self) -> f64 {
        let data_squared = &self.data * &self.data;

        data_squared.mean().unwrap_or(0.0) - self.mean().powi(2)
    }
}


/// Base trait for all continuous distributions.
/// 
/// `ContinuousDist` provides a general interface for implementing the functionality of continuous distributions; it differs 
/// from `DiscreteDist` only in its requirement of the `pdf()` function in place of `pmf()`, although this difference 
/// underscores a larger conceptual difference between the two traits.
/// 
/// Like `DiscreteDist`, `ContinuousDist` is also parameterized by a generic type `N` which conforms to the `Num` trait from 
/// `num-traits`.
pub trait ContinuousDist<N: Num> { // NOTE: are generics necessary?
    /// Returns the probability density function (PDF) of `value` according to the distribution. Convention is to return `0.0` for 
    /// values outside the distribution's support.
    fn pdf(&self, value: N) -> f64;

    /// Returns the cumulative distribution function (CDF) of `value` according to the distribution.
    fn cdf(&self, value: N) -> f64;

    /// Returns the probability that a value in the distribution will fall between `lower_bound` and `upper_bound`.
    /// 
    /// The default implementation simply finds the difference in the CDFs of the bounds.
    fn interval_cdf(&self, lower_bound: N, upper_bound: N) -> f64 {
        self.cdf(upper_bound) - self.cdf(lower_bound)
    }

    /// Returns the mean (expectation) of the distribution.
    fn mean(&self) -> f64;

    /// Returns the variance (average square distance from the mean) of the distribution.
    fn variance(&self) -> f64;

    /// Returns the standard deviation of the distribution.
    /// 
    /// The default implementation returns the square root of the variance.
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}


/// A continuous uniform distribution.
/// 
/// The continuous uniform distribution is the continuous adaptation of the discrete uniform distribution. Given a lower and 
/// upper bound of its support, the probabilitu of any real number inside the support is uniform. Any value outside the 
/// support has probability `0`.
#[derive(Debug, PartialEq)]
pub struct ContinuousUniformDist {
    // TODO: bad float values?
    lower_bound: f64,
    upper_bound: f64,
}

impl ContinuousUniformDist {
    /// Creates and returns a new continuous uniform distribution from `lower_bound` to `upper_bound`.
    /// 
    /// Returns `None` if `lower_bound > upper_bound`, otherwise returns the created distribution.
    /// 
    /// ```rust
    /// let a = 1.0;
    /// let b = 2.5;
    /// let dist = ContinuousUniformDist::new(a, b).unwrap();
    /// 
    /// println!("{}", dist.lower_bound()); // prints "1.0"
    /// println!("{}", dist.upper_bound()); // prints "2.5"
    /// ```
    ///
    /// ```rust
    /// let a = 1.0;
    /// let b = 2.5;
    /// let dist = ContinuousUniformDist::new(b, a);
    /// 
    /// println!("{}", dist == None); // prints "true"
    /// ```
    pub fn new(lower_bound: f64, upper_bound: f64) -> Option<ContinuousUniformDist> {
        if lower_bound > upper_bound {
            return None;
        }

        Some(ContinuousUniformDist { lower_bound, upper_bound })
    }

    /// Returns the lower bound of the support of the distribution.
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    /// Returns the upper bound of the support of the distribution.
    pub fn upper_bound(&self) -> f64 {
        self.upper_bound
    }

    /// Returns the size of the interval of the distribution's values.
    /// 
    /// This is equivalent to the difference between the bounds of the support.
    pub fn range(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }
}

impl ContinuousDist<f64> for ContinuousUniformDist {
    /// Returns the continuous uniform PDF of `value`.
    /// 
    /// If `value` is in the support of the distribution, then `1 / range` is returned, where `range` is the range of the 
    /// distribution. Otherwise, `0.0` is returned.
    /// 
    /// ```rust
    /// let a = 1.0;
    /// let b = 2.5;
    /// let dist = ContinuousUniformDist::new(a, b).unwrap();
    /// 
    /// println!("{}", dist.pdf(1.0)); // prints approximately "0.6667"
    /// println!("{}", dist.pdf(1.5)); // prints approximately "0.6667"
    /// println!("{}", dist.pdf(2.0)); // prints approximately "0.6667"
    /// println!("{}", dist.pdf(2.5)); // prints approximately "0.6667"
    /// 
    /// println!("{}", dist.pdf(0.0)); // prints "0.0"
    /// println!("{}", dist.pdf(3.0)); // prints "0.0"
    /// ```
    fn pdf(&self, value: f64) -> f64 {
        if value >= self.lower_bound && value <= self.upper_bound {
            return 1.0 / self.range();
        }
        else {
            return 0.0;
        }
    }

    /// Returns the continuous uniform CDF of `value`.
    /// 
    /// This is equivalent to the fraction of the support that `value` is greater than.
    /// 
    /// ```rust
    /// let a = 1.0;
    /// let b = 2.5;
    /// let dist = ContinuousUniformDist::new(a, b).unwrap();
    /// 
    /// println!("{}", dist.cdf(1.0)); // prints approximately "0.0"
    /// println!("{}", dist.cdf(1.5)); // prints approximately "0.3333"
    /// println!("{}", dist.cdf(2.0)); // prints approximately "0.6667"
    /// println!("{}", dist.cdf(2.5)); // prints approximately "1.0"
    /// 
    /// println!("{}", dist.cdf(0.0)); // prints "0.0"
    /// println!("{}", dist.cdf(3.0)); // prints "1.0"
    /// ```
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

    /// Returns the probability that the continuous uniform random variable falls between `lower_bound` and `upper_bound`.
    /// 
    /// This is equivalent to the fraction of the support that the interval occupies.
    /// 
    /// ```rust
    /// let a = 1.0;
    /// let b = 2.5;
    /// let dist = ContinuousUniformDist::new(a, b).unwrap();
    /// 
    /// println!("{}", dist.interval_cdf(1.5, 2.0)); // prints approximately "0.3333"
    /// ```
    fn interval_cdf(&self, lower_bound: f64, upper_bound: f64) -> f64 {
        let upper = if upper_bound < self.upper_bound { upper_bound } else { self.upper_bound };
        let lower = if lower_bound > self.lower_bound { lower_bound } else { self.lower_bound };

        (upper - lower) / self.range()
    }

    /// Returns the mean of the uniform distribution.
    /// 
    /// This is equivalent to the average of the support's bounds.
    fn mean(&self) -> f64 {
        (self.lower_bound + self.upper_bound) / 2.0
    }

    /// Returns the variance of the uniform distribution.
    /// 
    /// This is equivalent to: `(upper bound - lower bound)^2 / 12`.
    fn variance(&self) -> f64 {
        (self.upper_bound - self.lower_bound).powi(2) / 12.0
    }

    /// Returns the variance of the uniform distribution.
    /// 
    /// This is equivalent to the square root of the variance, or `(upper bound - lower bound) / sqrt(12)`.
    fn std(&self) -> f64 {
        (self.upper_bound - self.lower_bound) / 12.0_f64.sqrt()
    }
}


/// An exponential distribution.
/// 
/// The exponential distribution is parameterized by a rate parameter, which dictates the "stretch/shrink" of the distribution's PDF. 
/// The support of the exponential distribution is the set of non-negative real numbers.
#[derive(Debug, PartialEq)]
pub struct ExponentialDist {
    rate_param: f64
}

impl ExponentialDist {
    /// Creates and returns a new exponential distribution.
    /// 
    /// Because the rate parameter must be positive, the function returns `None` if `rate_param <= 0.0`. Otherwise, the 
    /// distribution is returned.
    /// 
    /// ```rust
    /// let r = 0.5;
    /// let dist = ExponentialDist::new(r).unwrap();
    /// 
    /// println!("{}", dist.rate_param()); // prints "0.5"
    /// ```
    /// 
    /// ```rust
    /// let r = -0.5;
    /// let dist = ExponentialDist::new(r);
    /// println!("{}", dist == None); // prints "true"
    /// ```
    pub fn new(rate_param: f64) -> Option<ExponentialDist> {
        if rate_param <= 0.0 {
            return None;
        }

        Some(ExponentialDist { rate_param })
    }

    /// Returns the rate parameter of the distribution.
    pub fn rate_param(&self) -> f64 {
        self.rate_param
    }
}

impl ContinuousDist<f64> for ExponentialDist {
    /// Returns the exponential PDF of `value`.
    /// 
    /// When `value` is outside the support (i.e. negative), `0.0` is returned. Otherwise, the correct PDF is returned.
    /// 
    /// ```rust
    /// let r = 0.5;
    /// let dist = ExponentialDist::new(r).unwrap();
    /// 
    /// println!("{}", dist.pdf(0.0)); // prints "0.0"
    /// println!("{}", dist.pdf(2.5)); // prints approximately "0.1432"
    /// println!("{}", dist.pdf(5.0)); // prints approximately "0.041"
    /// ```
    fn pdf(&self, value: f64) -> f64 {
        if value < 0.0 { 
            return 0.0;
        }

        self.rate_param * (-self.rate_param * value).exp()
    }

    /// Returns the exponential CDF of `value`.
    /// 
    /// When `value` is outside the support (i.e. negative), `0.0` is returned. Otherwise, the correct PDF is returned.
    /// 
    /// ```rust
    /// let r = 0.5;
    /// let dist = ExponentialDist::new(r).unwrap();
    /// 
    /// println!("{}", dist.cdf(0.0)); // prints "0.0"
    /// println!("{}", dist.cdf(2.5)); // prints approximately "0.7135"
    /// println!("{}", dist.cdf(5.0)); // prints approximately "0.9179"
    /// ```
    fn cdf(&self, value: f64) -> f64 {
        if value < 0.0 { 
            return 0.0;
        }

        1.0 - (-self.rate_param * value).exp()
    }

    /// Returns the mean of the distribution.
    /// 
    /// This is equivalent to the inverse of the rate parameter of the distribution.
    fn mean(&self) -> f64 {
        1.0 / self.rate_param
    }

    /// Returns the variance of the distribution.
    /// 
    /// This is equivalent to the inverse of the squared rate parameter of the distribution.
    fn variance(&self) -> f64 {
        1.0 / self.rate_param.powi(2)
    }

    /// Returns the standard of the distribution.
    /// 
    /// This is equivalent to the inverse of the rate parameter of the distribution (which, coincidentally, is also the mean).
    fn std(&self) -> f64 {
        1.0 / self.rate_param
    }
}


/// A normal (Gaussian) distribution.
/// 
/// The normal distribution is parameterized by a location and a scale (both real numbers). Those familiar with the normal 
/// distribution may recognize these parameters as the distribution's mean and standard deviation; `NormalDist` names its 
/// parameters to reflect their conceptual purpose rather than their relevance to the properties of the distribution.
/// 
/// The support of the normal distribution is all real numbers.
#[derive(Debug, PartialEq)]
pub struct NormalDist {
    loc: f64,
    scale: f64,
}

impl NormalDist {
    /// Creates and returns a new normal distribution.
    /// 
    /// If `scale` is negative, `None` is returned; otherwise, the created distribution with parameters 
    /// `loc` and `scale` is returned.
    /// 
    /// ```rust
    /// let loc = 5.0;
    /// let scale = 2.0;
    /// let dist = NormalDist::new(loc, scale).unwrap();
    /// 
    /// println!("{}", dist.loc()); // prints "5.0"
    /// println!("{}", dist.scale()); // prints "2.0"
    /// ```
    /// 
    /// ```rust
    /// let loc = 5.0;
    /// let scale = -2.0;
    /// let dist = NormalDist::new(loc, scale);
    /// 
    /// println!("{}", dist == None); // prints "true"
    /// ```
    pub fn new(loc: f64, scale: f64) -> Option<NormalDist> {
        if scale < 0.0 {
            return None;
        }

        Some(NormalDist { loc, scale })
    }

    /// Creates and returns a new standard normal distribution.
    /// 
    /// The standard normal distribution is a normal distribution with `loc = 0` and `scale = 1`.
    /// 
    /// ```rust
    /// let dist = NormalDist::std();
    /// 
    /// println!("{}", dist.loc()); // prints "0.0"
    /// println!("{}", dist.scale()); // prints "1.0"
    /// ```
    pub fn std() -> NormalDist {
        NormalDist { loc: 0.0, scale: 1.0 }
    }

    /// Returns the location of the normal distribution.
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Returns the scale of the normal distribution.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the Z-value of `value` within the distribution.
    /// 
    /// The Z-value is obtained by finding the distance from `value` to the mean of the distribution in units of its standard 
    /// deviation.
    /// 
    /// ```rust
    /// let loc = 5.0;
    /// let scale = 2.0;
    /// let dist = NormalDist::new(loc, scale).unwrap();
    /// 
    /// println!("{}", dist.z(7.0)); // prints "1.0"
    /// println!("{}", dist.z(3.0)); // prints "-1.0"
    /// ```
    pub fn z(&self, value: f64) -> f64 {
        (value - self.loc) / self.scale
    }
}

impl ContinuousDist<f64> for NormalDist {
    /// Returns the normal PDF of `value`.
    /// 
    /// ```rust
    /// let loc = 5.0;
    /// let scale = 2.0;
    /// let dist = NormalDist::new(loc, scale).unwrap();
    /// 
    /// println!("{}", dist.pdf(1.0)); // prints approximately "0.0269"
    /// println!("{}", dist.pdf(3.0)); // prints approximately "0.1209"
    /// println!("{}", dist.pdf(5.0)); // prints approximately "0.1995"
    /// println!("{}", dist.pdf(7.0)); // prints approximately "0.1209"
    /// println!("{}", dist.pdf(9.0)); // prints approximately "0.0269"
    /// ```
    fn pdf(&self, value: f64) -> f64 {
        let t1 = 1.0 / ((2.0 * PI).sqrt() * self.scale);
        let t2 = -(value - self.loc).powi(2) / (2.0 * self.variance());

        t1 * t2.exp()
    }

    /// Returns the normal CDF of `value`.
    /// 
    /// Note that a less naive approximation (and corresponding citation) are coming soon.
    /// 
    /// ```rust
    /// let loc = 5.0;
    /// let scale = 2.0;
    /// let dist = NormalDist::new(loc, scale).unwrap();
    /// 
    /// println!("{}", dist.cdf(1.0)); // prints approximately "0.025"
    /// println!("{}", dist.cdf(3.0)); // prints approximately "0.17"
    /// println!("{}", dist.cdf(5.0)); // prints approximately "0.5"
    /// println!("{}", dist.cdf(7.0)); // prints approximately "0.83"
    /// println!("{}", dist.cdf(9.0)); // prints approximately "0.975"
    /// ```
    fn cdf(&self, value: f64) -> f64 {
        let z = self.z(value);

        let b_values = array![0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429];
        let b_0 = 0.2316419;

        let t = 1.0 / (1.0 + b_0 * z);
        let t_values = array![t, t.powi(2), t.powi(3), t.powi(4), t.powi(5)];

        1.0 - NormalDist::std().pdf(z) * (t_values.dot(&b_values))
    }

    /// Returns the mean of the distribution, equivalent to the location.
    fn mean(&self) -> f64 {
        self.loc
    }

    /// Returns the variance of the distribution, equivalent to the scale squared.
    fn variance(&self) -> f64 {
        self.scale.powi(2)
    }

    /// Returns the standard deviation of the distribution, equivalent to the scale.
    fn std(&self) -> f64 {
        self.scale
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_uniform_dist_created_correctly() {
        let a = 0;
        let b = 4;
        let dist = DiscreteUniformDist::new(0, 4).unwrap(); // panics if creation fails
        
        assert_eq!(dist.lower_bound(), a);
        assert_eq!(dist.upper_bound(), b);
        assert_eq!(dist.range(), b - a + 1);
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
        
        let diff = dist.interval_cdf(lower_bound + 1, upper_bound - 1) - (dist.range() - 2) as f64 / dist.range() as f64;
        assert!(diff < 1e-10);
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

    #[test]
    fn bernoulli_dist_correct_interval_cdf() {
        let p = 0.5;
        let dist = BernoulliDist::new(p).unwrap();

        assert_eq!(dist.interval_cdf(0, 1), 1.0);
    }

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

    #[test]
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
        
        assert!((dist.cdf(-1) - 0.0).abs() < 1e-10);
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
        assert!(pmfs.all_close(&array![0.4, 0.24, 0.144], 1e-10));
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
    fn geometric_dist_correct_cdf_outofrange() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        assert_eq!(dist.cdf(-1), 0.0);
    }

    #[test]
    fn geometric_dist_correct_interval_cdf() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        let a = 3.0;
        let b = 5.0;
        let cdf = Array::range(a, b + 1.0, 1.0).mapv(|k| dist.pmf(k as i32)).sum();

        let diff = dist.interval_cdf(a as i32, b as i32) - cdf;

        assert!(diff.abs() < 1e-10);
    }

    #[test]
    fn geometric_dist_mean_calculated_correctly() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        assert_eq!(dist.mean(), 1.0 / p);
    }

    #[test]
    fn geometric_dist_variance_calculated_correctly() {
        let p = 0.4;
        let dist = GeometricDist::new(p).unwrap();

        let diff = dist.variance() - (1.0 - p) / p.powi(2);
        assert!(diff < 1e-10);
    }

    #[test]
    fn empirical_dist_valid_created_correctly() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        for (i, n) in data.iter().enumerate() {
            assert_eq!(dist.data[i], *n);
        }
    }

    #[test]
    fn empirical_dist_invalid_creation_fails() {
        let data = array![1.0, 2.0, f64::NAN, 3.0];

        match EmpiricalDist::new(&data) {
            Some(_) => panic!("found empirical dist instead of None"),
            None => {},
        }
    }

    #[test]
    fn empirical_dist_correct_pmf_indata() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        let mut counts = BTreeMap::new();
        counts.insert(0, 1.0);
        counts.insert(1, 2.0);
        counts.insert(3, 3.0);

        for (i, n) in counts.iter() {
            let v = data[[*i]];

            assert_eq!(dist.pmf(v), *n / data.len() as f64);
        }
    }

    #[test]
    fn empirical_dist_correct_pmf_notindata() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        assert_eq!(dist.pmf(4.0), 0.0);
    }

    #[test]
    fn empirical_dist_correct_pmf_invalid_value() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        assert_eq!(dist.pmf(f64::NAN), 0.0);
    }

    #[test]
    fn empirical_dist_correct_cdf_inrange() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        let mut counts = BTreeMap::new();
        counts.insert(0, 1.0);
        counts.insert(1, 3.0);
        counts.insert(3, 6.0);

        for (i, n) in counts.iter() {
            let v = data[[*i]];

            assert_eq!(dist.cdf(v), *n / data.len() as f64);
        }
    }

    #[test]
    fn empirical_dist_correct_cdf_outofrange() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        assert_eq!(dist.cdf(0.0), 0.0);
        assert_eq!(dist.cdf(5.0), 1.0);
    }

    #[test]
    fn empirical_dist_correct_interval_cdf() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        assert_eq!(dist.interval_cdf(2.0, 4.0), 5.0 / 6.0);
    }

    #[test]
    fn empirical_dist_mean_calculated_correctly() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        assert_eq!(dist.mean(), 2.5);
    }

    #[test]
    fn empirical_dist_variance_calculated_correctly() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let dist = EmpiricalDist::new(&data).unwrap();

        let diff = dist.variance() - 0.917;

        assert!(diff < 1e-10);
    }

    #[test]
    fn continuous_uniform_dist_valid_created_correctly() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        assert_eq!(dist.lower_bound(), a);
        assert_eq!(dist.upper_bound(), b);
        assert_eq!(dist.range(), b - a);
    }

    #[test]
    fn continuous_uniform_dist_invalid_bounds_creation_fails() {
        let a = 2.0;
        let b = 1.5;
        let dist = ContinuousUniformDist::new(a, b);

        assert_eq!(dist, None);
    }
            
    #[test]
    fn continuous_uniform_dist_correct_pdf_inrange() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        let pdfs = Array::range(a, b, 0.5).mapv(|x| dist.pdf(x));
        let expected = Array::from_elem(pdfs.shape(), 1.0 / (b - a));

        assert!(pdfs.all_close(&expected, 1e-10));
    }

    #[test]
    fn continuous_uniform_dist_correct_pdf_outofrange() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        assert_eq!(dist.pdf(0.0), 0.0);
        assert_eq!(dist.pdf(5.0), 0.0);
    }

        // cdf
    #[test]
    fn continuous_uniform_dist_correct_cdf_inrange() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        let cdfs = Array::range(a, b, 0.5).mapv(|x| dist.cdf(x));
        let expected = Array::range(a, b, 0.5).mapv(|x| (x - a) / (b - a));

        assert!(cdfs.all_close(&expected, 1e-10));
    }

    #[test]
    fn continuous_uniform_dist_correct_cdf_outofrange() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        assert_eq!(dist.cdf(0.0), 0.0);
        assert_eq!(dist.cdf(5.0), 1.0);
    }

    #[test]
    fn continuous_uniform_dist_correct_interval_cdf_inrange() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        let x = a + 0.5;
        let y = b - 0.5;

        assert_eq!(dist.interval_cdf(x, y), dist.cdf(y) - dist.cdf(x));
    }

    #[test]
    fn continuous_uniform_dist_correct_interval_cdf_outofrange() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        assert_eq!(dist.interval_cdf(a - 0.5, b + 0.5), 1.0);
    }

    #[test]
    fn continuous_uniform_dist_mean_calculated_correctly() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        assert_eq!(dist.mean(), (a + b) / 2.0);
    }

    #[test]
    fn continuous_uniform_dist_variance_calculated_correctly() {
        let a = 1.0;
        let b = 2.5;
        let dist = ContinuousUniformDist::new(a, b).unwrap();

        assert_eq!(dist.variance(), (b - a).powi(2) / 12.0);
    }

    #[test]
    fn exp_dist_valid_created_correctly() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        assert_eq!(dist.rate_param(), r);
    }
    
    #[test]
    fn exp_dist_invalid_rateparam_creation_fails() {
        let r = -0.5;
        let dist = ExponentialDist::new(r);

        assert_eq!(dist, None);
    }
        
    #[test]
    fn exp_dist_correct_pdf_inrange() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        let pdfs = Array::range(0.0, 2.0, 0.5).mapv(|k| dist.pdf(k));
        let expected = array![0.5, 0.3894, 0.3033, 0.2362];

        assert!(pdfs.all_close(&expected, 1e-4)); // error is relatively low because I rounded to 4 decimal places
    }

    #[test]
    fn exp_dist_correct_pdf_outofrange() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        assert_eq!(dist.pdf(-1.0), 0.0);
    }
    
    #[test]
    fn exp_dist_correct_cdf_inrange() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        let cdfs = Array::range(0.0, 4.0, 0.5).mapv(|x| dist.cdf(x));
        let expected = Array::range(0.0, 4.0, 0.5).mapv(|x| { 1.0 - (-r * x).exp() });

        assert!(cdfs.all_close(&expected, 1e-10));
    }

    #[test]
    fn exp_dist_correct_cdf_outofrange() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        assert_eq!(dist.cdf(-1.0), 0.0);
    }
        
    #[test]
    fn exp_dist_correct_interval_cdf() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        let a = 1.0;
        let b = 4.0;

        assert_eq!(dist.interval_cdf(a, b), dist.cdf(b) - dist.cdf(a));
    }
        
    #[test]
    fn exp_dist_mean_calculated_correctly() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        assert_eq!(dist.mean(), 1.0 / r);
    }
        
    #[test]
    fn exp_dist_variance_calculated_correctly() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        assert_eq!(dist.variance(), 1.0 / r.powi(2));
    }
        
    #[test]
    fn exp_dist_std_calculated_correctly() {
        let r = 0.5;
        let dist = ExponentialDist::new(r).unwrap();

        assert_eq!(dist.std(), 1.0 / r);
    }
    
    #[test]
    fn normal_dist_valid_created_correctly() {
        let loc = 5.0;
        let scale = 2.0;
        let dist = NormalDist::new(loc, scale).unwrap();

        assert_eq!(dist.loc(), loc);
        assert_eq!(dist.scale(), scale);
    }
    
    #[test]
    fn normal_dist_invalid_scale_creation_fails() {
        let loc = 5.0;
        let scale = -2.0;
        let dist = NormalDist::new(loc, scale);

        assert_eq!(dist, None);
    }

    #[test]
    fn normal_dist_std_created_correctly() {
        let dist = NormalDist::std();

        assert_eq!(dist.loc(), 0.0);
        assert_eq!(dist.scale(), 1.0);
    }

    #[test]
    fn normal_dist_zvalue_calculated_correctly() {
        let loc = 5.0;
        let scale = 2.0;
        let dist = NormalDist::new(loc, scale).unwrap();

        assert_eq!(dist.z(7.0), 1.0);
        assert_eq!(dist.z(5.0), 0.0);
        assert_eq!(dist.z(3.0), -1.0);
    }

    #[test]
    fn normal_dist_pdf_correct() {
        let loc = 5.0;
        let scale = 2.0;
        let dist = NormalDist::new(loc, scale).unwrap();

        let pdfs = Array::range(-5.0, 20.0, 5.0).mapv(|x| dist.pdf(x));
        let expected = array![0.00000074, 0.00876415, 0.19947114, 0.00876415, 0.00000074];

        assert!(pdfs.all_close(&expected, 1e-8));
    }

    #[test]
    fn normal_dist_cdf_correct() {
        let loc = 5.0;
        let scale = 2.0;
        let dist = NormalDist::new(loc, scale).unwrap();

        let cdfs = Array::range(0.0, 12.5, 2.5).mapv(|x| dist.cdf(x));
        let expected = array![0.0062, 0.1056, 0.5, 0.8944, 0.9938]; // NOTE: improve accuracy

        println!("{}", cdfs);
        assert!(cdfs.all_close(&expected, 1e-8));
    }
        
    #[test]
    fn normal_dist_mean_calculated_correctly() {
        let loc = 5.0;
        let scale = 2.0;
        let dist = NormalDist::new(loc, scale).unwrap();

        assert_eq!(dist.mean(), loc);
    }
    
    #[test]
    fn normal_dist_variance_calculated_correctly() {
        let loc = 5.0;
        let scale = 2.0;
        let dist = NormalDist::new(loc, scale).unwrap();

        assert_eq!(dist.variance(), scale.powi(2));
    }
    
    #[test]
    fn normal_dist_std_calculated_correctly() {
        let loc = 5.0;
        let scale = 2.0;
        let dist = NormalDist::new(loc, scale).unwrap();

        assert_eq!(dist.std(), scale);
    }

    #[test]
    fn tmp_code_example_runner() {
        let loc = 5.0;
        let scale = 2.0;
        let dist = NormalDist::new(loc, scale).unwrap();
        
        println!("{}", dist.pdf(1.0)); // prints "0.0269"
        println!("{}", dist.pdf(3.0)); // prints "0.1209"
        println!("{}", dist.pdf(5.0)); // prints "0.1995"
        println!("{}", dist.pdf(7.0)); // prints "0.1209"
        println!("{}", dist.pdf(9.0)); // prints "0.0269"

        panic!("");
    }
}