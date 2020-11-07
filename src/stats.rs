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
    success_prob: f64,
}

impl BernoulliDist {
    pub fn new(success_prob: f64) -> BernoulliDist {
        if success_prob < 0.0 || success_prob > 1.0 {
            panic!("Bernoulli probability of success must be between 0 and 1");
        }

        BernoulliDist { success_prob }
    }

    pub fn success_prob(&self) -> f64 {
        self.success_prob
    }
}

impl DiscreteDist<i32> for BernoulliDist {
    fn pmf(&self, value: i32) -> f64 {
        if value == 0 {
            return 1.0 - self.success_prob;
        }
        else if value == 1 {
            return self.success_prob;
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
            return 1.0 - self.success_prob;
        }
        else {
            return 1.0;
        }
    }

    fn mean(&self) -> f64 {
        self.success_prob
    }

    fn variance(&self) -> f64 {
        self.success_prob * (1.0 - self.success_prob)
    }
}


#[cfg(test)]
mod tests {
    // TODO: add discrete uniform dist tests
}