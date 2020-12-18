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


pub struct AdversarialSearchAgent<'a, State: AdversarialSearchState> {
    strategy: &'a fn(State) -> Box<dyn AdversarialSearchNode<State>>,
    adversaries: Vec<&'a fn(State) -> Box<dyn AdversarialSearchNode<State>>>,
    max_depth: Option<usize>, // NOTE: depth is defined slightly differently than the traditional tree depth property
}

impl<'a, State: AdversarialSearchState + 'static> AdversarialSearchAgent<'a, State> {
    pub fn new(strategy: &'a fn(State) -> Box<dyn AdversarialSearchNode<State>>, adversaries: Vec<&'a fn(State) -> Box<dyn AdversarialSearchNode<State>>>, max_depth: Option<usize>) -> AdversarialSearchAgent<'a, State> {
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
        let root = self.make_node(state, 0, 0);
        let (_, action) = root.utility();

        action
    }

    fn make_node(&self, state: State, depth: usize, layer: usize) -> Box<dyn AdversarialSearchNode<State>> {
        // if node should be terminal, make terminal
        // else, make according to strategy and add children
        if state.is_terminal() || (self.max_depth != None && depth == self.max_depth.unwrap()) {
            return TerminalNode::new(state);
        }
        else {
            let mut node = if layer == 0 {
                (self.strategy)(state)
            }
            else {
                self.adversaries[layer - 1](state)
            };

            for action in node.state().actions() {
                let successor = node.state().successor(action);

                let next_layer = layer + 1;
                let next_depth = if next_layer < self.adversaries.len() + 1 { depth } else { depth + 1 };
                let child = self.make_node(successor, next_depth, next_layer % (self.adversaries.len() + 1));

                if let Some(children) = node.children() {
                    children.push(child);
                }
            }

            return node;
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
    /// override this if you would like to modify this functionality. Note that if this method returns `true`, it's expected that `self.actions()` 
    /// returns a non-empty list. Conversely, if it returns `false`, `self.actions()` should return an empty list, and will be treated as such in 
    /// adversarial search.
    fn is_terminal(&self) -> bool {
        self.actions().len() == 0
    }
}


/// Base trait for all nodes in a game tree.
/// 
/// This trait is used for defining a general public interface for nodes in a game tree (e.g. minimizer and maximizer nodes). 
/// Its main relevance to developers is through the provided types that implement it, unless you want to create a custom node 
/// type to use in a game tree, in which case you should implement this trait for that type (see also `AdversarialSearchStrategy` if 
/// doing so).
pub trait AdversarialSearchNode<State: AdversarialSearchState> {
    // /// Creates and returns a new adversarial search node for the given state.
    // /// 
    // /// Use this method as a public constructor for each node type; it takes in a state and creates the corresponding node (typically with no 
    // /// children upon construction).
    // fn new(state: State) -> Box<dyn AdversarialSearchNode<State>>;

    /// Returns a reference to the state stored at this node.
    /// 
    /// This method is typically simply an accessor method of the node's internal state field.
    fn state(&self) -> &State;

    /// Returns a reference to a vector containing the node's children, if they exist.
    /// 
    /// Returns `None` if the node has no children, otherwise return a reference to them. Note that there is a subtle difference between a node that
    /// _can't_ have children and a node that _doesn't_ have children. This method should return `None` only if it's unable to have children (perhaps 
    /// the only practical purpose of this is terminal nodes, but the subtle difference is important to keep in mind).
    /// 
    /// Also note that a node's children don't need to be of the same type; as long as they use the same state as this node, 
    /// they're valid children. As a concrete example, this means that a minimizer node can have maximizer nodes as its 
    /// children, as long as all nodes are generic to the same state.
    fn children(&mut self) -> Option<&mut Vec<Box<dyn AdversarialSearchNode<State>>>>;

    /// Returns the node's utility and the action required to achieve that utility, if it exists.
    /// 
    /// Note that the optional action returned will be `None` if there is no action required to achieve the returned utility, otherwise it will 
    /// return the action required to achieve the returned utility.
    /// 
    /// How a node's utility is calculated is determined by the type of node in this method. For example, minimizer nodes 
    /// will minimize the utility of their children; maximizer nodes do the opposite. Thus, this method is how to determine 
    /// the method by which a node calculates its own usility, potentially relative to its children's utility.
    fn utility(&self) -> (State::Utility, Option<State::Action>);
}


/// A terminal (i.e. leaf) node in the game tree.
/// 
/// `TerminalNode` is private to the module because it has no practical value to users -- its only use is to internally represent terminal nodes in 
/// the game tree; third-party users can't (and shouldn't) create artibitrary terminal nodes in the game tree, therefore this functionality is not publicly exposed.
struct TerminalNode<State: AdversarialSearchState> {
    state: State,
}

// NOTE: I don't like that it's bound to 'static
impl<State: 'static + AdversarialSearchState> TerminalNode<State> {
    /// Creates and returns a new terminal node for the given state.
    fn new(state: State) -> Box<dyn AdversarialSearchNode<State>> {
        Box::new(TerminalNode { state })
    }
}

impl<State: AdversarialSearchState> AdversarialSearchNode<State> for TerminalNode<State> {
    fn state(&self) -> &State {
        &self.state
    }

    /// Return an optional reference to a vector containing the node's children. Since this node is terminal, always returns `None`.
    fn children(&mut self) -> Option<&mut Vec<Box<dyn AdversarialSearchNode<State>>>> {
        None
    }

    /// Determine and return the utility of the node, and return the action required to achieve that utility, if it exists.
    /// 
    /// Since this node is terminal, the returned utility is the utility of the node's state as determined by the state's evaluation function; the 
    /// returned action is always `None`.
    fn utility(&self) -> (State::Utility, Option<State::Action>) {
        (self.state.eval(), None)
    }
}


/// A minimizer node in the game tree.
/// 
/// This node minimizes the utilities of its children (regardless of how they're determined) to determine its own utility. To that end, its generic 
/// state must be comparable, i.e. it must implement `PartialOrd`.
pub struct MinimizerNode<State: AdversarialSearchState> where State::Utility: PartialOrd {
    state: State,
    children: Vec<Box<dyn AdversarialSearchNode<State>>>,
}

impl<State: 'static + AdversarialSearchState> MinimizerNode<State> where State::Utility: PartialOrd {
    /// Creates and returns a new minimizer node for the given state. Note that nodes are initialized with no children; they are added sequentially 
    /// during game tree creation.
    fn new(state: State) -> Box<dyn AdversarialSearchNode<State>> {
        Box::new(MinimizerNode { state, children: vec![] })
    }
}

impl<State: AdversarialSearchState> AdversarialSearchNode<State> for MinimizerNode<State> where State::Utility: PartialOrd {
    fn state(&self) -> &State {
        &self.state
    }

    fn children(&mut self) -> Option<&mut Vec<Box<dyn AdversarialSearchNode<State>>>> {
        Some(&mut self.children)
    }

    /// Determines and returns the utility of the node and the action required to achieve that utility.
    /// 
    /// For minimizer nodes, utility is defined as the minimum utility among the node's children (also note that minimizer nodes are guaranteed to 
    /// have children because they aren't `TerminalNode`s). Additionally, the returned action is guaranteed to exist, again because the node isn't 
    /// terminal.
    /// 
    /// Note that this method returns the _first_ action that achieves optimal utility, that is, in the event of a tie, the optimal action is not changed.
    fn utility(&self) -> (State::Utility, Option<State::Action>) {
        let (mut min_utility, mut optimal_action) = self.children[0].utility();

        for child in self.children[1..].iter() {
            let (utility, action) = child.utility();

            if utility < min_utility {
                min_utility = utility;
                optimal_action = action;
            }
        }

        (min_utility, optimal_action)
    }
}


/// A maximizer node in the game tree.
/// 
/// This node determines its optimal utility as the maximum utility between its children. In order to perform this comparison, the generic type 
/// parameter's associated type `State::Utility` must implement `PartialOrd`.
pub struct MaximizerNode<State: AdversarialSearchState> where State::Utility: PartialOrd {
    state: State,
    children: Vec<Box<dyn AdversarialSearchNode<State>>>,
}

impl<State: 'static + AdversarialSearchState> MaximizerNode<State> where State::Utility: PartialOrd {
    /// Creates and returns a new maximizer node for the given state. Note that nodes are initialized with no children; they are added sequentially 
    /// during game tree creation.
    fn new(state: State) -> Box<dyn AdversarialSearchNode<State>> {
        Box::new(MaximizerNode { state, children: vec![] })
    }
}

impl<State: AdversarialSearchState> AdversarialSearchNode<State> for MaximizerNode<State> where State::Utility: PartialOrd {
    fn state(&self) -> &State {
        &self.state
    }

    fn children(&mut self) -> Option<&mut Vec<Box<dyn AdversarialSearchNode<State>>>> {
        Some(&mut self.children)
    }

    /// Determines and returns the utility of the node and the action required to achieve that utility.
    /// 
    /// For maximizer nodes, utility is defined as the maximum utility among the node's children (also note that maximizer nodes are guaranteed to 
    /// have children because they aren't `TerminalNode`s). Additionally, the returned action is guaranteed to exist, again because the node isn't 
    /// terminal.
    /// 
    /// Note that this method returns the _first_ action that achieves optimal utility, that is, in the event of a tie, the optimal action is not changed.
    fn utility(&self) -> (State::Utility, Option<State::Action>) {
        let (mut max_utility, mut optimal_action) = self.children[0].utility();

        for child in self.children[1..].iter() {
            let (utility, action) = child.utility();

            if utility > max_utility {
                max_utility = utility;
                optimal_action = action;
            }
        }

        (max_utility, optimal_action)
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