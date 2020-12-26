//! Adversarial search & game trees.
//! 
//! This module provides methods for evaluating game trees through adversarial search. It was designed to allow for building 
//! game trees flexibly, and to provide a central method for evaluating these flexible game trees through adversarial search. 
//! This is accomplished by dividing the process of adversarial search into 3 steps:
//! 
//! NOTE: a complete example program using adversarial search is available [here](https://github.com/arthurlafrance/pensiv). 
//! See it for examples of the code in this module.
//!
//! ## Defining an Adversarial Search Problem
//! 
//! In order to define an adversarial search problem, you must define the state that represents it. Specifically, you must 
//! define a type to represent this state, and implement the `AdversarialSearchState` trait for it. This allows the custom state to be 
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


use num_traits::Num;
use num_traits::identities::{Zero, One};

use std::marker::PhantomData;


pub struct AdversarialSearchAgent<'a, State: AdversarialSearchState> {
    strategy: &'a fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a>,
    adversaries: Vec<&'a fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a>>,
    max_depth: Option<usize>, // NOTE: depth is defined slightly differently than the traditional tree depth property
}

impl<'a, State: 'a + AdversarialSearchState> AdversarialSearchAgent<'a, State> {
    pub fn new(strategy: &'a fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a>, adversaries: Vec<&'a fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a>>, max_depth: Option<usize>) -> AdversarialSearchAgent<'a, State> {
        if let Some(d) = max_depth {
            if d <= 0 {
                panic!("Adversarial search max depth must be a positive integer (or None)");
            }
        }

        AdversarialSearchAgent { strategy, adversaries, max_depth }
    }

    // pub fn minimax(adversaries: i32) -> AdversarialSearchAgent {

    // }

    // pub fn expectimax(adversaries: i32) -> AdversarialSearchAgent {

    // }

    pub fn action(&self, state: State) -> Option<State::Action> {
        let root = self.make_node(state, 0);
        let (_, action) = root.utility();

        match action {
            Some(a) => Some(a.clone()),
            None => None,
        }
    }

    fn make_node(&self, state: State, depth: usize) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        // if node should be terminal, make terminal
        // else, make according to strategy and add successors
        if state.is_terminal() || (self.max_depth != None && depth == self.max_depth.unwrap()) {
            TerminalNode::new(state)
        }
        else {
            let mut successors = vec![];

            for action in state.actions().iter() {
                let successor_state = state.successor(action);
                let child = self.make_node(successor_state, depth + 1);

                let successor = AdversarialSearchSuccessor::new(*action, child);
                successors.push(successor);
            }

            let layer = depth % (self.adversaries.len() + 1);

            let node_constructor = if layer == 0 {
                self.strategy
            }
            else {
                self.adversaries[layer - 1]
            };

            node_constructor(state, successors)
        }
    }
}


/// Base trait for states in a game tree
/// 
/// Implement this trait for custom states that are specific to the adversarial search problem. Note that this trait defines 
/// only the public interface of the game tree state for use in adversarial search; it's up to the developer to define the 
/// internals of their state to be specific to the problem they're trying to solve. For example, a chess-playing 
/// program might define its own state to represent the board, and would implement this trait for that state in order to use 
/// the custom state in `pensiv`'s adversarial search functionality.
pub trait AdversarialSearchState {
    /// Describes a legal action that can be taken during the game.
    /// 
    /// This type must implement the `Clone` trait so that it can be duplicated within adversarial search
    type Action: Copy;

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
    fn successor(&self, action: &Self::Action) -> Self;

    /// Returns the utility of the current state according to the evaluation function.
    /// 
    /// Use this function to define a custom evaluation function by which to determine the utility of a custom state.
    fn eval(&self) -> Self::Utility;

    /// Returns `true` if the state is a terminal state, `false` otherwise.
    /// 
    /// The default implementation of this method simply checks to see if there are any legal actions that can be taken; 
    /// override this if you would like to modify this functionality. Note that if this method returns `true`, it's expected that `self.actions()` 
    /// returns a non-empty list. Conversely, if it returns `false`, `self.actions()` should return an empty list, and will be treated as such in 
    /// adversarial search.
    fn is_terminal(&self) -> bool {
        self.actions().len() == 0
    }
}


/// A successor in the adversarial search tree.
/// 
/// Represents a path from state to successor in the game tree. It holds the successor state and the action required to travel to that successor. This struct 
/// is mainly intended for internal use in adversarial search, but is exposed for compatibility with `AdversarialSearchNode`.
pub struct AdversarialSearchSuccessor<'a, State: AdversarialSearchState> {
    action: State::Action,
    node: Box<dyn AdversarialSearchNode<'a, State> + 'a>,
}

impl<'a, State: AdversarialSearchState> AdversarialSearchSuccessor<'a, State> {
    /// Create and return a new successor.
    pub fn new(action: State::Action, node: Box<dyn AdversarialSearchNode<'a, State> + 'a>) -> AdversarialSearchSuccessor<'a, State> {
        AdversarialSearchSuccessor { action, node }
    }

    /// Return the action leading to the successor state.
    pub fn action(&self) -> &State::Action {
        &self.action
    }

    /// Return a reference to the child node.
    pub fn node(&self) -> &dyn AdversarialSearchNode<'a, State> {
        &(*(self.node)) // NOTE: this is kinda jank syntactically, is it acceptable?
    }
}


/// Base trait for all nodes in a game tree.
/// 
/// This trait is used for defining a general public interface for nodes in a game tree (e.g. minimizer and maximizer nodes). 
/// Its main relevance to developers is through the provided types that implement it, unless you want to create a custom node 
/// type to use in a game tree, in which case you should implement this trait for that type (see also `AdversarialSearchStrategy` if 
/// doing so).
pub trait AdversarialSearchNode<'a, State: AdversarialSearchState> {
    // /// Creates and returns a new adversarial search node for the given state.
    // /// 
    // /// Use this method as a public constructor for each node type; it takes in a state and creates the corresponding node (typically with no 
    // /// successors upon construction).
    // fn new(state: State) -> Box<dyn AdversarialSearchNode<State>>;

    /// Returns a reference to the state stored at this node.
    /// 
    /// This method is typically simply an accessor method of the node's internal state field.
    fn state(&self) -> &State;

    /// Returns a reference to a vector containing the node's successors (i.e. child nodes), if they exist.
    /// 
    /// Returns `None` if the node has no successors, otherwise return a reference to them. Note that there is a subtle difference between a node that
    /// _can't_ have successors and a node that _doesn't_ have successors. This method should return `None` only if it's unable to have successors (perhaps 
    /// the only practical purpose of this is terminal nodes, but the subtle difference is important to keep in mind).
    /// 
    /// Also note that a node's successors don't need to be of the same type; as long as they use the same state as this node, 
    /// they're valid successors. As a concrete example, this means that a minimizer node can have maximizer nodes as its 
    /// successors, as long as all nodes are generic to the same state.
    fn successors(&self) -> Option<&Vec<AdversarialSearchSuccessor<'a, State>>>;

    /// Returns the node's utility and the action required to achieve that utility, if it exists.
    /// 
    /// Note that the optional action returned will be `None` if there is no action required to achieve the returned utility, otherwise it will 
    /// return the action required to achieve the returned utility.
    /// 
    /// How a node's utility is calculated is determined by the type of node in this method. For example, minimizer nodes 
    /// will minimize the utility of their successors; maximizer nodes do the opposite. Thus, this method is how to determine 
    /// the method by which a node calculates its own usility, potentially relative to its successors's utility.
    fn utility(&self) -> (State::Utility, Option<&State::Action>);
}


/// A terminal (i.e. leaf) node in the game tree.
/// 
/// `TerminalNode` is private to the module because it has no practical value to users -- its only use is to internally represent terminal nodes in 
/// the game tree; third-party users can't (and shouldn't) create artibitrary terminal nodes in the game tree, therefore this functionality is not publicly exposed.
struct TerminalNode<'a, State: AdversarialSearchState> {
    state: State,
    _phantom: PhantomData<&'a i32>,
}

impl<'a, State: 'a + AdversarialSearchState> TerminalNode<'a, State> {
    /// Creates and returns a new terminal node for the given state.
    fn new(state: State) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        Box::new(TerminalNode { state, _phantom: PhantomData })
    }
}

impl<'a, State: AdversarialSearchState> AdversarialSearchNode<'a, State> for TerminalNode<'a, State> {
    fn state(&self) -> &State {
        &self.state
    }

    /// Return an optional reference to a vector containing the node's successors. Since this node is terminal, always returns `None`.
    fn successors(&self) -> Option<&Vec<AdversarialSearchSuccessor<'a, State>>> {
        None
    }

    /// Determine and return the utility of the node, and return the action required to achieve that utility, if it exists.
    /// 
    /// Since this node is terminal, the returned utility is the utility of the node's state as determined by the state's evaluation function; the 
    /// returned action is always `None`.
    fn utility(&self) -> (State::Utility, Option<&State::Action>) {
        (self.state.eval(), None)
    }
}


/// A minimizer node in the game tree.
/// 
/// This node minimizes the utilities of its successors (regardless of how they're determined) to determine its own utility. To that end, its generic 
/// state must be comparable, i.e. it must implement `PartialOrd`.
pub struct MinimizerNode<'a, State: AdversarialSearchState> where State::Utility: PartialOrd {
    state: State,
    successors: Vec<AdversarialSearchSuccessor<'a, State>>,
}

impl<'a, State: 'a + AdversarialSearchState> MinimizerNode<'a, State> where State::Utility: PartialOrd {
    /// Creates and returns a new minimizer node for the given state. Note that nodes are initialized with no successors; they are added sequentially 
    /// during game tree creation.
    fn new(state: State, successors: Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        Box::new(MinimizerNode { state, successors })
    }
}

impl<'a, State: AdversarialSearchState> AdversarialSearchNode<'a, State> for MinimizerNode<'a, State> where State::Utility: PartialOrd {
    fn state(&self) -> &State {
        &self.state
    }

    fn successors(&self) -> Option<&Vec<AdversarialSearchSuccessor<'a, State>>> {
        Some(&self.successors)
    }

    /// Determines and returns the utility of the node and the action required to achieve that utility.
    /// 
    /// For minimizer nodes, utility is defined as the minimum utility among the node's successors (also note that minimizer nodes are guaranteed to 
    /// have successors because they aren't `TerminalNode`s). Additionally, the returned action is guaranteed to exist, again because the node isn't 
    /// terminal.
    /// 
    /// Note that this method returns the _first_ action that achieves optimal utility, that is, in the event of a tie, the optimal action is not changed.
    fn utility(&self) -> (State::Utility, Option<&State::Action>) {
        let successor = &self.successors[0];

        let (mut min_utility, _) = successor.node().utility();
        let mut optimal_action = successor.action();

        for successor in self.successors[1..].iter() {
            let (utility, _) = successor.node().utility();
            let action = successor.action();

            if utility < min_utility {
                min_utility = utility;
                optimal_action = action;
            }
        }

        (min_utility, Some(optimal_action))
    }
}


/// A maximizer node in the game tree.
/// 
/// This node determines its optimal utility as the maximum utility between its successors. In order to perform this comparison, the generic type 
/// parameter's associated type `State::Utility` must implement `PartialOrd`.
pub struct MaximizerNode<'a, State: AdversarialSearchState> where State::Utility: PartialOrd {
    state: State,
    successors: Vec<AdversarialSearchSuccessor<'a, State>>,
}

impl<'a, State: 'a + AdversarialSearchState> MaximizerNode<'a, State> where State::Utility: PartialOrd {
    /// Creates and returns a new maximizer node for the given state. Note that nodes are initialized with no successors; they are added sequentially 
    /// during game tree creation.
    fn new(state: State, successors: Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        Box::new(MaximizerNode { state, successors })
    }
}

impl<'a, State: AdversarialSearchState> AdversarialSearchNode<'a, State> for MaximizerNode<'a, State> where State::Utility: PartialOrd {
    fn state(&self) -> &State {
        &self.state
    }

    fn successors(&self) -> Option<&Vec<AdversarialSearchSuccessor<'a, State>>> {
        Some(&self.successors)
    }

    /// Determines and returns the utility of the node and the action required to achieve that utility.
    /// 
    /// For maximizer nodes, utility is defined as the maximum utility among the node's successors (also note that maximizer nodes are guaranteed to 
    /// have successors because they aren't `TerminalNode`s). Additionally, the returned action is guaranteed to exist, again because the node isn't 
    /// terminal.
    /// 
    /// Note that this method returns the _first_ action that achieves optimal utility, that is, in the event of a tie, the optimal action is not changed.
    fn utility(&self) -> (State::Utility, Option<&State::Action>) {
        let successor = &self.successors[0];

        let (mut max_utility, _) = successor.node().utility();
        let mut optimal_action = successor.action();

        for successor in self.successors[1..].iter() {
            let (utility, _) = successor.node().utility();
            let action = successor.action();

            if utility > max_utility {
                max_utility = utility;
                optimal_action = action;
            }
        }

        (max_utility, Some(optimal_action))
    }
}


pub struct ChanceNode<'a, State: AdversarialSearchState> where State::Utility: Num {
    state: State,
    successors: Vec<AdversarialSearchSuccessor<'a, State>>,
}

impl<'a, State: 'a + AdversarialSearchState> ChanceNode<'a, State> where State::Utility: Num {
    pub fn new(state: State, successors: Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        Box::new(ChanceNode { state, successors })
    }
}

impl<'a, State: AdversarialSearchState> AdversarialSearchNode<'a, State> for ChanceNode<'a, State> where State::Utility: Num {
    fn state(&self) -> &State {
        &self.state
    }

    fn successors(&self) -> Option<&Vec<AdversarialSearchSuccessor<'a, State>>> {
        Some(&self.successors)
    }

    fn utility(&self) -> (State::Utility, Option<&State::Action>) {
        let mut total_utility = State::Utility::zero();
        let mut n_successors = State::Utility::zero();

        for successor in self.successors.iter() {
            let (utility, _) = successor.node().utility();

            // Can't use += because of the lack of additional trait requirement
            total_utility = total_utility + utility;
            n_successors = n_successors + State::Utility::one();
        }

        (total_utility / n_successors, None)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct CountToTenState<'a> {
        count: i32,
        done: bool,
        _test: &'a Vec<i32>,
    }

    impl<'a> CountToTenState<'a> {
        // pub fn start() -> CountToTenState {
        //     CountToTenState { count: 0, done: false, _test:  }
        // }

        pub fn new(count: i32, done: bool, r: &Vec<i32>) -> CountToTenState {
            CountToTenState { count, done, _test: r }
        }
    }

    // impl<'a> AdversarialSearchState for CountToTenState<'a> {
    //     type Action = CountToTenAction;
    //     type Utility = i32;

    //     fn actions(&self) -> Vec<Self::Action> {
    //         if self.done {
    //             Vec::new()
    //         }
    //         else if self.count < 10 {
    //             vec![CountToTenAction::Increment, CountToTenAction::Done]
    //         }
    //         else {
    //             vec![CountToTenAction::Done]
    //         }
    //     }

    //     fn successor(&self, action: Self::Action) -> Self {            
    //         match action {
    //             CountToTenAction::Increment => CountToTenState::new(self.count + 1, false, self._test),
    //             CountToTenAction::Done => CountToTenState::new(self.count, true, self._test),
    //         }
    //     }

    //     fn eval(&self) -> Self::Utility {
    //         self.count
    //     }

    //     fn is_terminal(&self) -> bool {
    //         self.done
    //     }
    // }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum CountToTenAction {
        Increment,
        Done,
    }
}