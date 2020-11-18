//! Adversarial search & game trees.
//! 
//! This module provides methods for evaluating game trees through adversarial search. It was designed to allow for building 
//! game trees flexibly, and to provide a central method for evaluating these flexible game trees through adversarial search. 
//! This is accomplished by dividing the process of adversarial search into 3 steps:
//! 
//! NOTE: a complete example program using adversarial search is available [here](https://github.com/arthurlafrance/pensiv). 
//! See this example for examples of the code in this module.
//!
//! ## Defining an Adversarial Search Problem
//! 
//! In order to define an adversarial search problem, you must define the state that represents it. Specifically, you must 
//! define a type to represent this state, and implement the `GameTreeState` trait for it. This allows the custom state to be 
//! used in `pensiv`'s provided adversarial search implementation.
//! 
//! In order to implement this trait, you must define a type to represent the actions that can be taken between states; 
//! usually this takes the form of either a simple `struct` or enum. Additionally, you must define a method that returns 
//! a vector of the actions that can legally be taken from a given state; this will be used to identify child nodes in the 
//! game tree. You must also define a method that returns the successor state that arises from taking some action at some 
//! state. This will be used in tandem with the previous method to link actions to successor, or in other words to link parent 
//! and child nodes in the game tree.
//! 
//! In addition to defining a mechanism for performing actions, you must also define a mechanism for representing utility in 
//! your adversarial search problem through the `Utility` associated type. This will be used to judge the utility of states, 
//! or equivalently the utility of nodes in the game tree, in order to identify the optimal solution. In tandem with this, the 
//! `eval()` method must be implemented in order to provide an evaluation function by which to judge the utility of a state; it 
//! calculates and returns the utility of a given state according to the defined associated type.
//!
//! After defining a sufficient state representation for your adversarial search problem, you can move on to building your 
//! adversarial search agent (describebd in the following section) which will be used to perform adversarial search.
//! 
//! ## Building an Adversarial Search Agent
//! 
//! TBD
//! 
//! ## Performing Adversarial Search
//! 
//! TBD


/// Base trait for states in a game tree
/// 
/// Implement this trait for custom states that are specific to the adversarial search problem. Note that this trait defines 
/// only the public interface of the game tree state for use in adversarial search; it's up to the developer to define the 
/// internals of their state to be specific to the problem they're trying to solve. For example, a chess-playing 
/// program might define its own state to represent the board, and would implement this trait for that state in order to use 
/// the custom state in `pensiv`'s adversarial search functionality.
pub trait GameTreeState {
    /// Describes a legal action that can be taken during the game.
    type Action;

    /// Describes the utility achieved at the end of a game.
    /// 
    /// Usually this is either a numeric value or a tuple of values, but `pensiv` provides flexibility as to how to represent 
    /// utility; as long as you know how to handle your own utility type, you're free to use whatever type you prefer.
    type Utility;

    /// Returns a vector containing the legal actions that can be taken at the current state.
    fn actions(&self) -> Vec<Self::Action>;

    // TODO: i think this should return a result to indicate legality of action
    /// Returns the successor state that arises from taking the given action at the current state.
    /// 
    /// Note that this function assumes that the action being taken is a valid action to take from the current state; any 
    /// violation of this precondition is undefined behavior, and can be handled at the developer's discretion.
    fn successor(&self, action: Self::Action) -> Self;

    /// Returns the utility of the current state according to the evaluation function.
    /// 
    /// Use this function to define a custom evaluation function by which to determine the utility of a custom state.
    fn eval(&self) -> Self::Utility;

    /// Returns `true` if the state is a terminal state, `false` otherwise.
    /// 
    /// The default implementation of this method simply checks to see if there are any legal actions that can be taken; 
    /// override this if you would like to modify this functionality.
    fn is_terminal(&self) -> bool {
        self.actions().len() == 0
    }
}


/// Base trait for all strategies in a game tree.
/// 
/// `pensiv` defines a game tree strategy as the type of node to be used at a given layer in the game tree. The main purpose of 
/// this is to serve as a factory for creating nodes in a game tree in an idiomatic way. Thus, implement this trait only if you 
/// also implement a custom type of game tree node, or if you want to alter the creation of an existing type of node. 
/// You should not have to implement this trait in any other circumstance; implementations of this trait for nodes implemented by 
/// `pensiv` will be provided.
pub trait GameTreeStrategy<State: GameTreeState> {
    /// The type of game tree node to be used for the strategy.
    type Node: GameTreeNode<State>;

    /// Creates and returns a node of the proper type for the given state.
    fn node(&self, state: State) -> Self::Node;
}


/// Base trait for all nodes in a game tree.
/// 
/// This trait is used for defining a general public interface for nodes in a game tree (e.g. minimizer and maximizer nodes). 
/// Its main relevance to developers is through the provided types that implement it, unless you want to create a custom node 
/// type to use in a game tree, in which case you should implement this trait for that type (see also `GameTreeStrategy` if 
/// doing so).
pub trait GameTreeNode<State: GameTreeState> {
    /// Returns a reference to the state stored at this node.
    /// 
    /// This method is typically simply an accessor method of the node's internal state field.
    fn state(&self) -> &State;

    /// Returns a reference to a vector containing the node's children, if they exist.
    /// 
    /// Returns `None` if the node is terminal, i.e. if it has no children, otherwise return a reference to them.
    /// 
    /// Note that a node's children don't need to be of the same type; as long as they use the same state as this node, 
    /// they're valid children. As a concrete example, this means that a minimizer node can have maximizer nodes as its 
    /// children, as long as all nodes are generic to the same state.
    fn children(&self) -> Option<&Vec<Box<dyn GameTreeNode<State>>>>;

    /// Returns the node's utility and the action required to achieve that utility, if it exists.
    /// 
    /// Note that the optional action returned will be `None` if the node is a terminal node, otherwise it will correctly 
    /// return the action required to achieve the returned utility.
    /// 
    /// How a node's utility is calculated is determined by the type of node in this method. For example, minimizer nodes 
    /// will minimize the utility of their children; maximizer nodes do the opposite. Thus, this method is how to determine 
    /// the method by which a node calculates its own usility, potentially relative to its children's utility.
    fn utility(&self) -> (State::Utility, Option<State::Action>);
}


/// A terminal (i.e. leaf) node in a game tree.
/// 
/// This node type is used for leaf nodes in the game tree, based on two criteria: if the node's state is terminal, it's a 
/// terminal node; or alternatively, if it's part of the last layer of the game tree, it's a terminal node. A terminal node 
/// has no children by definition, so its utility is simply the evaluated utility of its state.
pub struct TerminalNode<State: GameTreeState> {
    state: State, // by reference?
}

impl<State: GameTreeState> TerminalNode<State> {
    /// Creates and returns a new terminal node for the given state.
    /// 
    /// Note that the state is moved as part of the creation, and will not be valid after creation. Unless, of course, it can 
    /// be copied, although be aware that that's a cost that arises from this constructor.
    pub fn new(state: State) -> TerminalNode<State> {
        TerminalNode { state }
    }
}

impl<State: GameTreeState> GameTreeNode<State> for TerminalNode<State> {
    /// Returns a reference to the state for this node.
    fn state(&self) -> &State {
        &self.state
    }

    /// Returns an optional reference to a vector of this node's children. Note that because this node is terminal, this 
    /// method always returns `None`.
    fn children(&self) -> Option<&Vec<Box<dyn GameTreeNode<State>>>> {
        None
    }

    /// Returns the utility of this node's (possibly terminal) state, calculated according to the state's evaluation function. 
    /// Note that the optional action will always be `None` because this is a terminal state.
    fn utility(&self) -> (State::Utility, Option<State::Action>) {
        (self.state.eval(), None)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct CountToTenState {
        count: i32,
        done: bool,
    }

    impl CountToTenState {
        pub fn start() -> CountToTenState {
            CountToTenState { count: 0, done: false }
        }

        pub fn new(count: i32, done: bool) -> CountToTenState {
            CountToTenState { count: count, done: done }
        }
    }

    impl GameTreeState for CountToTenState {
        type Action = CountToTenAction;
        type Utility = i32;

        fn actions(&self) -> Vec<Self::Action> {
            if self.done {
                Vec::new()
            }
            else if self.count < 10 {
                vec![CountToTenAction::Increment, CountToTenAction::Done]
            }
            else {
                vec![CountToTenAction::Done]
            }
        }

        fn successor(&self, action: Self::Action) -> Self {            
            match action {
                CountToTenAction::Increment => CountToTenState::new(self.count + 1, false),
                CountToTenAction::Done => CountToTenState::new(self.count, true),
            }
        }

        fn eval(&self) -> Self::Utility {
            self.count
        }

        fn is_terminal(&self) -> bool {
            self.done
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum CountToTenAction {
        Increment,
        Done,
    }


    #[test]
    fn terminal_node_created_with_correct_properties() {
        let state = CountToTenState::new(8, true);
        let node = TerminalNode::new(state);

        assert_eq!(*node.state(), state);
        assert_eq!(node.utility(), (state.eval(), None));

        match node.children() {
            Some(_) => panic!("found children for terminal node"),
            None => {},
        };
    }
}