use std::{ops, fmt, iter};


#[derive(Debug)]
pub struct Dimensions {
    axes: Box<[u32]>
}


impl Dimensions {
    pub fn axes(&self) -> &[u32] {
        &self.axes
    }

    pub fn count(&self) -> usize {
        self.axes.len()
    }
}


impl ops::Index<usize> for Dimensions {
    type Output = u32;

    fn index(&self, index: usize) -> &u32 {
        &self.axes[index]
    }
}


impl ops::IndexMut<usize> for Dimensions {
    fn index_mut(&mut self, index: usize) -> &mut u32 {
        &mut self.axes[index]
    }
}


impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dim_strings: Vec<String> = self.axes.iter().map(|d| format!("{}", d)).collect();
        write!(f, "({})", dim_strings.join(", "))
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


enum ArrayContents<Element> {
    OneDim { elements: Box<[Element]> },
    MultiDim { elements: Box<[ArrayContents<Element>]> }
}

pub struct Array<Element> {
    contents: ArrayContents<Element>,
    shape: Dimensions
}

impl<Element> Array<Element> {
    // pub fn new(shape: Dimensions) -> Array<Element> {
        // need dimensions iterator
        // for each dimension:
            // if this is last dimension, make OneDim
            // else, make vector of smaller array contents
    // }
}


#[cfg(test)]
mod tests { 
    use crate::core;

    #[test]
    fn dim_creation_sanitycheck() {
        let d = dim![1, 2, 3];

        assert_eq!(d.axes(), [1, 2, 3]);
        assert_eq!(d.count(), 3);
    }

    #[test]
    fn dim_index_sanitycheck() {
        let d = dim![1, 2, 3];
        let expected = vec![1, 2, 3];

        for i in 0..d.count() {
            assert_eq!(d[i], expected[i]);
        }
    }

    #[test]
    fn dim_display_sanitycheck() {
        let d = dim![1, 2, 3];
        let expected = "(1, 2, 3)";
        let dim_str = format!("{}", d);
        
        assert_eq!(dim_str, expected);
    }
}