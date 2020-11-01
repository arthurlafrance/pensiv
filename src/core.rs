#[derive(Debug)]
pub struct Dimensions {
    axes: Box<[u32]>
}

impl Dimensions {
    pub fn axes(&self) -> &[u32] {
        &self.axes
    }

    pub fn count(&self) -> usize {
        self.axes().len()
    }
}

#[macro_export]
macro_rules! dim {
    ( $($axis:expr),* ) => {
        {
            use crate::core;
            let mut axes = Vec::new();
            $( 
                axes.push($axis); 
            )*

            core::Dimensions { axes: axes.into_boxed_slice() }
        }
    };
}


#[cfg(test)]
mod tests { 
    use crate::core;

    #[test]
    fn dim_creation_test() {
        let d = dim![1, 2, 3];

        assert_eq!(d.axes(), [1, 2, 3]);
        assert_eq!(d.count(), 3);
    }
}