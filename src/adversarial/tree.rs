// TODO: module docs

use num_traits::Float;
use num_traits::identities::{Zero, One};

use std::marker::PhantomData;

use super::state::AdversarialSearchState;

/// A successor in the adversarial search tree.
///
/// Represents a path from state to successor in the game tree; holds the successor state and the action required to travel to that successor. This type is exposed for compatibility with
/// the rest of the module's public interface.
pub struct AdversarialSearchSuccessor<'s, State: AdversarialSearchState> {
    action: State::Action,
    node: Box<dyn AdversarialSearchNode<'s, State> + 's>,
}

impl<'s, State: AdversarialSearchState> AdversarialSearchSuccessor<'s, State> {
    /// Create and return a new successor.
    pub fn new(action: State::Action, node: Box<dyn AdversarialSearchNode<'s, State> + 's>) -> AdversarialSearchSuccessor<'s, State> {
        AdversarialSearchSuccessor { action, node }
    }

    /// Return the action leading to the successor state.
    pub fn action(&self) -> State::Action {
        self.action
    }

    /// Return a reference to the child node.
    pub fn node(&self) -> &dyn AdversarialSearchNode<'s, State> {
        &*self.node
    }
}


/// Base trait for nodes in a game tree.
///
/// This trait is used for defining a general public interface for nodes in a game tree (e.g. minimizer and maximizer nodes).
/// Its main relevance to developers is through the provided types that implement it, unless you want to create a custom node
/// type to use in a game tree, in which case you should implement this trait for that type.
pub trait AdversarialSearchNode<'s, State: AdversarialSearchState> {
    /// Returns a reference to the state stored at this node.
    ///
    /// This method is typically simply an accessor method of the node's internal state field.
    fn state(&self) -> &State;

    /// Returns the node's utility and the action the agent must take to achieve that utility, if it exists.
    ///
    /// Note that the optional action returned will be `None` if there is no action required to achieve the returned utility, otherwise it will
    /// return the action required to achieve the returned utility.
    ///
    /// How a node's utility is calculated is determined by the type of node in this method. For example, minimizer nodes
    /// will minimize the utility of their successors; maximizer nodes do the opposite. Thus, this method is how to determine
    /// the method by which a node calculates its own usility, potentially relative to its successors' utility.
    fn eval(&self) -> (Vec<State::Utility>, Option<State::Action>);
}


/// A terminal (i.e. leaf) node in the game tree.
///
/// `TerminalNode` is private to the module because it has no practical value to users -- its only use is to internally represent terminal nodes in
/// the game tree; third-party users can't (and shouldn't) create artibitrary terminal nodes in the game tree, therefore this functionality is not publicly exposed.
pub(super) struct TerminalNode<'s, State: AdversarialSearchState> {
    state: State,
    _phantom: PhantomData<&'s i32>,
}

impl<'s, State: 's + AdversarialSearchState> TerminalNode<'s, State> {
    /// Creates and returns a new terminal node for the given state.
    pub(super) fn new(state: State) -> Box<dyn AdversarialSearchNode<'s, State> + 's> {
        Box::new(TerminalNode { state, _phantom: PhantomData })
    }
}

impl<'s, State: AdversarialSearchState> AdversarialSearchNode<'s, State> for TerminalNode<'s, State> {
    fn state(&self) -> &State {
        &self.state
    }

    /// Determine and return the utility of the node, and return the action required to achieve that utility, if it exists.
    ///
    /// Since this node is terminal, the returned utility is the utility of the node's state as determined by the state's evaluation function; the
    /// returned action is always `None`.
    fn eval(&self) -> (Vec<State::Utility>, Option<State::Action>) {
        (self.state.eval(), None)
    }
}


/// A minimizer node in the game tree.
///
/// This node minimizes the utilities of its successors (regardless of how they're determined) to determine its own utility. To that end, its generic
/// state must be comparable, i.e. it must implement `PartialOrd`.
pub struct MinimizerNode<'s, State: AdversarialSearchState> where State::Utility: PartialOrd {
    state: State,
    agent: usize,
    successors: Vec<AdversarialSearchSuccessor<'s, State>>,
}

impl<'s, State: 's + AdversarialSearchState> MinimizerNode<'s, State> where State::Utility: PartialOrd {
    /// Creates and returns a new minimizer node for the given state, agent, & successors.
    pub fn new(state: State, agent: usize, successors: Vec<AdversarialSearchSuccessor<'s, State>>) -> Box<dyn AdversarialSearchNode<'s, State> + 's> {
        Box::new(MinimizerNode { state, agent, successors })
    }
}

impl<'s, State: AdversarialSearchState> AdversarialSearchNode<'s, State> for MinimizerNode<'s, State> where State::Utility: PartialOrd {
    fn state(&self) -> &State {
        &self.state
    }

    /// Determines and returns the utility of the node and the action required to achieve that utility.
    ///
    /// For minimizer nodes, utility is defined as the minimum utility among the node's successors (also note that minimizer nodes are guaranteed to
    /// have successors because they aren't `TerminalNode`s). Additionally, the returned action is guaranteed to exist, again because the node isn't
    /// terminal.
    ///
    /// Note that this method returns the _first_ action that achieves optimal utility, that is, in the event of a tie, the first optimal action is chosen.
    fn eval(&self) -> (Vec<State::Utility>, Option<State::Action>) {
        let successor = &self.successors[0];

        let (mut result_utility, _) = successor.node().eval();
        let mut optimal_action = successor.action();
        let mut min_utility = result_utility[self.agent];

        for successor in self.successors[1..].iter() {
            let (utility, _) = successor.node().eval();
            let action = successor.action();
            let utility_val = utility[self.agent];

            if utility_val < min_utility {
                min_utility = utility_val;
                result_utility = utility;
                optimal_action = action;
            }
        }

        (result_utility, Some(optimal_action))
    }
}


/// A maximizer node in the game tree.
///
/// This node determines its optimal utility as the maximum utility between its successors. In order to perform this comparison, the generic type
/// parameter's associated type `State::Utility` must implement `PartialOrd`.
pub struct MaximizerNode<'s, State: AdversarialSearchState> where State::Utility: PartialOrd {
    state: State,
    agent: usize,
    successors: Vec<AdversarialSearchSuccessor<'s, State>>,
}

impl<'s, State: 's + AdversarialSearchState> MaximizerNode<'s, State> where State::Utility: PartialOrd {
    /// Creates and returns a new maximizer node for the given state & successors.
    pub fn new(state: State, agent: usize, successors: Vec<AdversarialSearchSuccessor<'s, State>>) -> Box<dyn AdversarialSearchNode<'s, State> + 's> {
        Box::new(MaximizerNode { state, agent, successors })
    }
}

impl<'s, State: AdversarialSearchState> AdversarialSearchNode<'s, State> for MaximizerNode<'s, State> where State::Utility: PartialOrd {
    fn state(&self) -> &State {
        &self.state
    }

    /// Determines and returns the utility of the node and the action required to achieve that utility.
    ///
    /// For maximizer nodes, utility is defined as the maximum utility among the node's successors (also note that maximizer nodes are guaranteed to
    /// have successors because they aren't `TerminalNode`s). Additionally, the returned action is guaranteed to exist, again because the node isn't
    /// terminal.
    ///
    /// Note that this method returns the _first_ action that achieves optimal utility, that is, in the event of a tie, the first optimal action is chosen.
    fn eval(&self) -> (Vec<State::Utility>, Option<State::Action>) {
        let successor = &self.successors[0];

        let (mut result_utility, _) = successor.node().eval();
        let mut optimal_action = successor.action();
        let mut max_utility = result_utility[self.agent];

        for successor in self.successors[1..].iter() {
            let (utility, _) = successor.node().eval();
            let action = successor.action();
            let utility_val = utility[self.agent];

            if utility_val > max_utility {
                max_utility = utility_val;
                result_utility = utility;
                optimal_action = action;
            }
        }

        (result_utility, Some(optimal_action))
    }
}


/// A chance node in the game tree.
///
/// This node determines its utility as the expected utility among its children. For simplicity, it currently assumes that its successors
/// are uniformly likely to be encountered (or, equivalently, that the "action to take" is selected uniformly at random). For this reason,
/// the `State` generic type must be numeric, i.e. it must be capable of performing basic arithmetic operations. Furthermore, it must also be
/// capable of floating point arithmetic, to ensure no information loss occurs when taking the average of the successors' utilities. Both of these
/// requirements are formalized by `num_traits::Float`.
pub struct ChanceNode<'s, State: AdversarialSearchState> where State::Utility: Float {
    state: State,
    successors: Vec<AdversarialSearchSuccessor<'s, State>>,
}

impl<'s, State: 's + AdversarialSearchState> ChanceNode<'s, State> where State::Utility: Float {
    /// Creates and returns a new chance node for the given state & successors.
    pub fn new(state: State, _agent: usize, successors: Vec<AdversarialSearchSuccessor<'s, State>>) -> Box<dyn AdversarialSearchNode<'s, State> + 's> {
        Box::new(ChanceNode { state, successors })
    }
}

impl<'s, State: AdversarialSearchState> AdversarialSearchNode<'s, State> for ChanceNode<'s, State> where State::Utility: Float {
    fn state(&self) -> &State {
        &self.state
    }

    /// Determines and returns the utility of the node. `None` is returned for the action because chance nodes predict only the expected utility, not an action to take.
    ///
    /// For chance nodes, utility is defined as the expected utility between the node's successors according to a uniform probability distribution. (Also note that chance nodes are guaranteed to
    /// have successors because they aren't `TerminalNode`s).
    fn eval(&self) -> (Vec<State::Utility>, Option<State::Action>) {
        let mut total_utility = vec![State::Utility::zero(); self.state.n_agents()];
        let mut n_successors = State::Utility::zero();

        for successor in self.successors.iter() {
            let (utility, _) = successor.node().eval();

            // Can't use += because of the lack of additional trait requirement
            for (i, u) in utility.into_iter().enumerate() {
                total_utility[i] = total_utility[i] + u;
            }

            n_successors = n_successors + State::Utility::one(); // I feel like this is kinda jank, there has to be a better way
        }

        (total_utility.into_iter().map(|u| u / n_successors).collect(), None)
    }
}
