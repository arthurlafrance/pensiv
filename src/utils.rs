use std::cmp::{Eq, Ord, Ordering};
use std::convert::Into;


#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct ComparableFloat(f64);

impl ComparableFloat {
    pub fn new(f: f64) -> Option<ComparableFloat> {
        if f.is_nan() { None }
        else { Some(ComparableFloat(f)) }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Eq for ComparableFloat { }

impl Ord for ComparableFloat {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.partial_cmp(rhs).unwrap()
    }
}

impl Into<f64> for ComparableFloat {
    fn into(self) -> f64 {
        self.0
    }
}