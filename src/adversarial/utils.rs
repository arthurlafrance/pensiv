pub trait Identifiable<ID> {
    fn id(&self) -> ID;
}

impl Identifiable<usize> for usize {
    fn id(&self) -> usize {
        *self
    }
}

// TODO: could impl Identifiable<usize> for other primitives
