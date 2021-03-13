//! Adversarial search & game-playing.
//!
//! This module provides methods for using game-playing agents through adversarial search. It's designed to allow for building
//! adversarial search trees flexibly, and to provide a central method for evaluating these flexibly-created trees through adversarial search.
//! This is accomplished by dividing the process of adversarial search into 2 steps:
//!
//! NOTE: a complete example program using adversarial search is available [here](https://github.com/arthurlafrance/Pensiv).
//! Refer to it for examples of the code in this module.
//!
//! ## Defining an Adversarial Search Problem
//!
//! Before performing adversarial search, you must define an adversarial search problem using the tools provided by `pensiv`. This
//! mainly entails defining the state representation for the problem. You must define a type that implements the
//! `AdversarialSearchState` trait which represents the state of the search problem. After doing so, you'll be ready to perform adversarial search based on the defined state.
//!
//! ## Performing Adversarial Search
//!
//! After defining a state representation, you can then perform adversarial search based on it. To do this, create an `AdversarialSearchAgent`
//! by defining a configuration for the game tree, either using minimax, "max-n" and expectimax or by specifying a custom configuration using the various
//! node constructor functions. After creating the agent, simply call the `optimal_action()` method to determine the optimal action for the
//! agent to take given some start state.
//!
//! ## Extensions
//!
//! The flexibility of `pensiv`'s adversarial search implementation makes it easy to extend if the existing implementation
//! doesn't quite fit your needs. There are two main ways to extend this module:
//! 1. Custom adversarial search nodes: While this module provides certain common node types out-of-the-box, it obviously can't
//! provide every possible node type imaginable. For that reason, it provides a flexible base trait, `AdversarialSearchNode`, which can be
//! implemented for any custom node type you may wish to define. In this way, you can bring customized functionality to your specific application
//! of adversarial search.
//! 2. Custom adversarial search tree configurations: In much the same way as the previous point, this module provides common configurations of
//! adversarial search trees (e.g. minimax), while also introducing the flexibility to define custom configurations that can be tailored to each
//! specific application of adversarial search. If the provided configurations aren't what you're looking for, you can use the custom constructor of
//! `AdversarialSearchAgent` to assign each layer (or equivalently, each agent in the game) to some arbitrary node type (including types not provided out-of-the-box).


use num_traits::Float;
use num_traits::identities::{Zero, One};

use std::marker::PhantomData;


/// An agent that performs adversarial search.
///
/// `AdversarialSearchAgent` is parameterized by the type parameter `State`, which represents the state of the adversarial search problem and must
/// implement `AdversarialSearchState`.
pub struct AdversarialSearchAgent<'a, State: AdversarialSearchState> {
    policies: Vec<AdversarialSearchPolicy<'a, State>>,
    n_policies: usize,
    max_depth: Option<usize>,
}

impl<'a, State: 'a + AdversarialSearchState> AdversarialSearchAgent<'a, State> {
    /// Creates and returns a new adversarial search agent.
    ///
    /// `AdversarialSearchAgent` is designed to provide flexible game tree creation and central adversarial search evaluation regardless of
    /// game tree layout. This is accomplished by specifying the agent's "policy", i.e. the type of node that the agent uses, as well as the
    /// strategies of its adversaries. In this way, the game tree is organized into layers whose node type is specified by the factory function
    /// parameters. Finally, an optional max depth can be specified to upper-bound adversarial search at some maximum depth in the game tree.
    ///
    /// Note that the array of policies must follow a specific order for adversarial search to work correctly: the agent's policy must be first,
    /// while the adversaries' policies can be second and after in whatever order you choose. This does mean that, because policies are evaluated
    /// sequentially, the adversaries' policies will be evaluated in the order in which they appear in the vector, so keep that in mind when using
    /// `AdversarialSearchAgent`.
    pub fn new(policies: Vec<AdversarialSearchPolicy<'a, State>>, max_depth: Option<usize>) -> AdversarialSearchAgent<State> {
        let n_policies = policies.len();

        AdversarialSearchAgent { policies, n_policies, max_depth }
    }

    /// Creates and returns a minimax agent.
    ///
    /// This is a convenience function for creating an adversarial search agent whose adversaries are assumed to use minizer nodes, and the agent is assumed to use maximizer nodes. Thus,
    /// the function requires you to specify the agents to include in the game tree as well as an optional maximum depth.
    ///
    /// Note that, just as the policies in the `new()` function appear in the vector in the order that they're evaluated, so too do the agents
    /// in the `agents` vector parameter. The player agent should appear first, and adversary agents should appear in whatever order you wish to
    /// evaluate them.
    pub fn minimax(agents: Vec<State::Agent>, max_depth: Option<usize>) -> AdversarialSearchAgent<'a, State> where State::Utility: PartialOrd {
        let n_policies = agents.len();
        let mut policies = Vec::with_capacity(n_policies);

        let agent_policy = AdversarialSearchPolicy::new(agents[0], MaximizerNode::new);
        policies.push(agent_policy);

        for agent in agents[1..].iter() {
            let policy = AdversarialSearchPolicy::new(*agent, MinimizerNode::new);
            policies.push(policy);
        }

        AdversarialSearchAgent { policies, n_policies, max_depth }
    }

    /// Creates and returns a expectimax agent.
    ///
    /// Like `minimax()`, this is a convenience function; it creates an adversarial search agent whose adversaries are assumed to use chance nodes, and the agent is assumed to use maximizer nodes. Thus,
    /// the function requires you to specify the agents to include in the game tree as well as an optional maximum depth.
    ///
    /// Note that, just as the agents in the `minimax()` function appear in the vector in the order that they're evaluated, so too do the agents
    /// in the `agents` vector parameter. The player agent should appear first, and adversary agents should appear in whatever order you wish to
    /// evaluate them.
    pub fn expectimax(agents: Vec<State::Agent>, max_depth: Option<usize>) -> AdversarialSearchAgent<'a, State> where State::Utility: PartialOrd + Float {
        let n_policies = agents.len();
        let mut policies = Vec::with_capacity(n_policies);

        let agent_policy = AdversarialSearchPolicy::new(agents[0], MaximizerNode::new);
        policies.push(agent_policy);

        for agent in agents[1..].iter() {
            let policy = AdversarialSearchPolicy::new(*agent, ChanceNode::new);
            policies.push(policy);
        }

        AdversarialSearchAgent { policies, n_policies, max_depth }
    }

    /// Return a reference to the policies for the agent.
    ///
    /// The player's policy will be the first element, and the adversaries' policies will appear sequentially after it.
    pub fn policies(&self) -> &Vec<AdversarialSearchPolicy<'a, State>> {
        &self.policies
    }

    /// Returns the number of policies in the game tree.
    ///
    /// This is equivalent to `self.policies().len()`
    pub fn n_policies(&self) -> usize {
        self.n_policies
    }

    /// Returns the (optional) maximum depth of adversarial search.
    ///
    /// If `None` is returned, there is no maximum depth, and adversarial search will proceed through the entire game tree.
    pub fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    /// Return the optimal action for the agent to take from the given state, if it exists.
    ///
    /// This function performs adversarial search -- it constructs a game tree starting at the current state according to the agent's known
    /// strategies, and evaluates it in order to determine which action will lead to optimal utility, if any.
    ///
    /// Note that this method assumes that `state` is a state from which the agent acts first; all adversaries will act in the order that their strategies
    /// were specified.
    ///
    /// One important note is that performing adversarial search with no maximum depth may lead to infinite recursion, if there exists
    /// some way to transition between states in a cycle. (Think of this as a cyclic state space graph, which would obviously result in a never-ending tree).
    /// Thus, be cognizant of this risk, and use infinite-depth adversarial search at your own risk.
    pub fn eval(&self, state: State) -> (State::Utility, Option<State::Action>) {
        let root = self.make_node(state, 0, 0);
        root.eval()
    }

    fn make_node(&self, state: State, policy_index: usize, depth: usize) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        let policy = &self.policies[policy_index];
        let agent = policy.agent();

        // if node should be terminal, make terminal
        // else, make according to policy and add successors
        if state.is_terminal() || (self.max_depth != None && depth == self.max_depth.unwrap()) {
            TerminalNode::new(state, agent)
        }
        else {
            let mut successors = vec![];
            let node_constructor = policy.node();

            for action in state.actions(agent).iter() {
                let successor_state = state.successor(agent, *action);
                let child = self.make_node(successor_state, (policy_index + 1) % self.n_policies, depth + 1);

                let successor = AdversarialSearchSuccessor::new(*action, child);
                successors.push(successor);
            }

            node_constructor(state, successors)
        }
    }
}


/// A policy in adversarial search: an agent and the node type that models it.
#[derive(Debug)]
pub struct AdversarialSearchPolicy<'a, State: AdversarialSearchState> {
    agent: State::Agent,
    node: fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a>,
}

impl<'a, State: AdversarialSearchState> AdversarialSearchPolicy<'a, State> {
    /// Creates and returns a new policy object.
    pub fn new(agent: State::Agent, node: fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a>) -> AdversarialSearchPolicy<State> {
        AdversarialSearchPolicy { agent, node }
    }

    /// Returns the agent to which the policy applies.
    pub fn agent(&self) -> State::Agent {
        self.agent
    }

    /// Returns a function pointer to the constructor for the policy's node type.
    pub fn node(&self) -> fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        self.node
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
    /// It's important to note that actions should be lightweight -- they should be small types that are easily copied rather than moved.
    /// As such, the `Action` associated type must implement the `Copy` trait.
    type Action: Copy;

    /// Describes the agents that are part of the state.
    ///
    /// Like `Action`, this type should be lightweight enough to copy; in most cases, it will be either a small enum or an integral primitive type.
    type Agent: Copy;

    /// Describes the utility achieved at the end of a game.
    ///
    /// Usually this is either a numeric value or a tuple of values, but `pensiv` provides flexibility as to how to represent
    /// utility; as long as you know how to handle your own utility type, you're free to use whatever type you prefer.
    type Utility;

    /// Returns a vector containing the legal actions that can be taken at the current state by the given agent.
    fn actions(&self, agent: Self::Agent) -> Vec<Self::Action>;

    // TODO: i think this should return a result to indicate legality of action
    /// Returns the successor state that arises from the given agent taking the given action at the current state.
    ///
    /// Note that this function assumes that the action being taken is a valid action to take from the current state; any
    /// violation of this precondition is undefined behavior, and can be handled at the developer's discretion.
    fn successor(&self, agent: Self::Agent, action: Self::Action) -> Self;

    /// Returns the evaluation function value of the current state for the specified agent.
    ///
    /// For terminal states, this should be the utility of the state for the given agent (i.e. their "score").
    /// For non-terminal states, this should be a heuristic estimate of the state's utility for the given agent, i.e. your
    /// best guess of their score.
    fn eval(&self, agent: Self::Agent) -> Self::Utility;

    /// Returns `true` if the state is a terminal state, `false` otherwise.
    ///
    /// This method must guarantee to be true when the current state has no successors, and false otherwise;
    /// if not, adversarial search may not work as intended.
    fn is_terminal(&self) -> bool;
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
    pub fn action(&self) -> State::Action {
        self.action
    }

    /// Return a reference to the child node.
    pub fn node(&self) -> &dyn AdversarialSearchNode<'a, State> {
        &*self.node
    }
}


/// Base trait for all nodes in a game tree.
///
/// This trait is used for defining a general public interface for nodes in a game tree (e.g. minimizer and maximizer nodes).
/// Its main relevance to developers is through the provided types that implement it, unless you want to create a custom node
/// type to use in a game tree, in which case you should implement this trait for that type.
pub trait AdversarialSearchNode<'a, State: AdversarialSearchState> {
    /// Returns a reference to the state stored at this node.
    ///
    /// This method is typically simply an accessor method of the node's internal state field.
    fn state(&self) -> &State;

    /// Returns the node's utility and the action required to achieve that utility, if it exists.
    ///
    /// Note that the optional action returned will be `None` if there is no action required to achieve the returned utility, otherwise it will
    /// return the action required to achieve the returned utility.
    ///
    /// How a node's utility is calculated is determined by the type of node in this method. For example, minimizer nodes
    /// will minimize the utility of their successors; maximizer nodes do the opposite. Thus, this method is how to determine
    /// the method by which a node calculates its own usility, potentially relative to its successors' utility.
    fn eval(&self) -> (State::Utility, Option<State::Action>);
}


/// A terminal (i.e. leaf) node in the game tree.
///
/// `TerminalNode` is private to the module because it has no practical value to users -- its only use is to internally represent terminal nodes in
/// the game tree; third-party users can't (and shouldn't) create artibitrary terminal nodes in the game tree, therefore this functionality is not publicly exposed.
struct TerminalNode<'a, State: AdversarialSearchState> {
    state: State,
    agent: State::Agent,
    _phantom: PhantomData<&'a i32>,
}

impl<'a, State: 'a + AdversarialSearchState> TerminalNode<'a, State> {
    /// Creates and returns a new terminal node for the given state.
    fn new(state: State, agent: State::Agent) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        Box::new(TerminalNode { state, agent, _phantom: PhantomData })
    }
}

impl<'a, State: AdversarialSearchState> AdversarialSearchNode<'a, State> for TerminalNode<'a, State> {
    fn state(&self) -> &State {
        &self.state
    }

    /// Determine and return the utility of the node, and return the action required to achieve that utility, if it exists.
    ///
    /// Since this node is terminal, the returned utility is the utility of the node's state as determined by the state's evaluation function; the
    /// returned action is always `None`.
    fn eval(&self) -> (State::Utility, Option<State::Action>) {
        (self.state.eval(self.agent), None)
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
    /// Creates and returns a new minimizer node for the given state & successors.
    fn new(state: State, successors: Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        Box::new(MinimizerNode { state, successors })
    }
}

impl<'a, State: AdversarialSearchState> AdversarialSearchNode<'a, State> for MinimizerNode<'a, State> where State::Utility: PartialOrd {
    fn state(&self) -> &State {
        &self.state
    }

    /// Determines and returns the utility of the node and the action required to achieve that utility.
    ///
    /// For minimizer nodes, utility is defined as the minimum utility among the node's successors (also note that minimizer nodes are guaranteed to
    /// have successors because they aren't `TerminalNode`s). Additionally, the returned action is guaranteed to exist, again because the node isn't
    /// terminal.
    ///
    /// Note that this method returns the _first_ action that achieves optimal utility, that is, in the event of a tie, the optimal action is not changed.
    fn eval(&self) -> (State::Utility, Option<State::Action>) {
        let successor = &self.successors[0];

        let (mut min_utility, _) = successor.node().eval();
        let mut optimal_action = successor.action();

        for successor in self.successors[1..].iter() {
            let (utility, _) = successor.node().eval();
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
    /// Creates and returns a new maximizer node for the given state & successors.
    fn new(state: State, successors: Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        Box::new(MaximizerNode { state, successors })
    }
}

impl<'a, State: AdversarialSearchState> AdversarialSearchNode<'a, State> for MaximizerNode<'a, State> where State::Utility: PartialOrd {
    fn state(&self) -> &State {
        &self.state
    }

    /// Determines and returns the utility of the node and the action required to achieve that utility.
    ///
    /// For maximizer nodes, utility is defined as the maximum utility among the node's successors (also note that maximizer nodes are guaranteed to
    /// have successors because they aren't `TerminalNode`s). Additionally, the returned action is guaranteed to exist, again because the node isn't
    /// terminal.
    ///
    /// Note that this method returns the _first_ action that achieves optimal utility, that is, in the event of a tie, the optimal action is not changed.
    fn eval(&self) -> (State::Utility, Option<State::Action>) {
        let successor = &self.successors[0];

        let (mut max_utility, _) = successor.node().eval();
        let mut optimal_action = successor.action();

        for successor in self.successors[1..].iter() {
            let (utility, _) = successor.node().eval();
            let action = successor.action();

            if utility > max_utility {
                max_utility = utility;
                optimal_action = action;
            }
        }

        (max_utility, Some(optimal_action))
    }
}


/// A chance node in the game tree.
///
/// This node determines its utility as the expected utility among its children. For simplicity, it currently assumes that its successors
/// are uniformly likely to be encountered (or, equivalently, that the "action to take" is selected uniformly at random). For this reason,
/// the `State` generic type must be numeric, i.e. it must be capable of performing basic arithmetic operations. Furthermore, it must also be
/// capable of floating point arithmetic, to ensure no information loss occurs when taking the average of the successors' utilities. Both of these
/// requirements are formalized by `num_traits::Float`.
pub struct ChanceNode<'a, State: AdversarialSearchState> where State::Utility: Float {
    state: State,
    successors: Vec<AdversarialSearchSuccessor<'a, State>>,
}

impl<'a, State: 'a + AdversarialSearchState> ChanceNode<'a, State> where State::Utility: Float {
    /// Creates and returns a new chance node for the given state & successors.
    fn new(state: State, successors: Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State> + 'a> {
        Box::new(ChanceNode { state, successors })
    }
}

impl<'a, State: AdversarialSearchState> AdversarialSearchNode<'a, State> for ChanceNode<'a, State> where State::Utility: Float {
    fn state(&self) -> &State {
        &self.state
    }

    /// Determines and returns the utility of the node. `None` is returned for the action because chance nodes predict only the expected utility, not an action to take.
    ///
    /// For chance nodes, utility is defined as the expected utility between the node's successors according to a uniform probability distribution. (Also note that chance nodes are guaranteed to
    /// have successors because they aren't `TerminalNode`s).
    fn eval(&self) -> (State::Utility, Option<State::Action>) {
        let mut total_utility = State::Utility::zero();
        let mut n_successors = State::Utility::zero();

        for successor in self.successors.iter() {
            let (utility, _) = successor.node().eval();

            // Can't use += because of the lack of additional trait requirement
            total_utility = total_utility + utility;
            n_successors = n_successors + State::Utility::one(); // I feel like this is kinda jank, there has to be a better way
        }

        (total_utility / n_successors, None)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::{Display, Formatter, Result as FmtResult};
    use std::str::FromStr;

    // new test "game":
        // simple game with two agents on a board where each spot has a numeric value
        // a collision between player & adversary ends the game
        // the player can:
            // move one spot right on the board
            // "call it" at any time, ie end the game at the current state
        // possible adversary movements:
            // move one spot left deterministically
            // move either left or right uniformly at random (how to make non-cyclic in this case?)
        // at the end of the game, the player's score is the numeric value of the spot that they're on
        // possible test cases by board config:
            // player moves forward to best spot and ends game
            // player moves forward to best spot that won't cause collision and ends game (even if that's not the global maximum value spot)
            // player chooses not to move
            // etc

    // TODO: set up adversarial search

    struct TestGameState {
        player_pos: usize,
        adversary_pos: usize,

        player_score: u32,
        adversary_score: u32,

        max_pos: usize,
        board: Vec<u32>,
    }

    impl TestGameState {
        fn start(board: Vec<u32>) -> TestGameState {
            let max_pos = board.len() - 1;

            TestGameState {
                player_pos: max_pos / 4usize, adversary_pos: max_pos * 3usize / 4usize,
                player_score: 0, adversary_score: 0,
                max_pos, board
            }
        }

        fn new(player_pos: usize, adversary_pos: usize, player_score: u32, adversary_score: u32, board: Vec<u32>) -> TestGameState {
            let max_pos = board.len() - 1;

            TestGameState {
                player_pos, adversary_pos,
                player_score, adversary_score,
                max_pos, board
            }
        }

        fn player_pos(&self) -> usize {
            self.player_pos
        }

        fn adversary_pos(&self) -> usize {
            self.adversary_pos
        }

        fn player_score(&self) -> u32 {
            self.player_score
        }

        fn adversary_score(&self) -> u32 {
            self.adversary_score
        }

        fn board(&self) -> &[u32] {
            &self.board
        }

        fn player_spot(&self) -> u32 {
            self.board[self.player_pos]
        }

        fn adversary_spot(&self) -> u32 {
            self.board[self.adversary_pos]
        }
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    enum TestGameAgent {
        Player,
        Adversary,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    enum TestGameAction {
        Left(usize),
        Right(usize),
        Peek, // only valid when you have no other moves
    }

    impl AdversarialSearchState for TestGameState {
        type Action = TestGameAction;
        type Agent = TestGameAgent;
        type Utility = f64;

        fn actions(&self, agent: TestGameAgent) -> Vec<TestGameAction> {
            if self.is_terminal() {
                return vec![];
            }

            let mut actions = Vec::with_capacity(3);

            let pos = match agent {
                TestGameAgent::Player => self.player_pos,
                TestGameAgent::Adversary => self.adversary_pos,
            };

            if pos > 0 {
                let mut left_neighbor = pos - 1;

                while self.board[left_neighbor] <= 0 {
                    if left_neighbor > 0 {
                        left_neighbor -= 1;
                    }
                    else {
                        break;
                    }
                }

                if self.board[left_neighbor] > 0 {
                    let action = TestGameAction::Left(pos - left_neighbor);

                    actions.push(action);
                }
            }

            if pos < self.max_pos {
                let mut right_neighbor = pos + 1;

                while right_neighbor <= self.max_pos && self.board[right_neighbor] <= 0 {
                    right_neighbor += 1;
                }

                if right_neighbor <= self.max_pos {
                    let action = TestGameAction::Right(right_neighbor - pos);
                    actions.push(action);
                }
            }

            if actions.len() == 0 {
                actions.push(TestGameAction::Peek);
            }

            actions
        }

        fn successor(&self, agent: TestGameAgent, action: TestGameAction) -> TestGameState {
            let mut player_pos = self.player_pos;
            let mut adversary_pos = self.adversary_pos;
            let mut player_score = self.player_score;
            let mut adversary_score = self.adversary_score;

            let pos;
            let score;
            let mut board = self.board.clone();

            match agent {
                TestGameAgent::Player => {
                    pos = &mut player_pos;
                    score = &mut player_score;
                },
                TestGameAgent::Adversary => {
                    pos = &mut adversary_pos;
                    score = &mut adversary_score;
                },
            };

            *score += board[*pos];
            board[*pos] = 0;

            match action {
                TestGameAction::Left(dx) => {
                    *pos -= dx;
                },
                TestGameAction::Right(dx) => {
                    *pos += dx;
                },
                TestGameAction::Peek => {},
            };

            if player_pos == adversary_pos {
                // if they collide, adversary "eats" player
                adversary_score += player_score;
                player_score = 0;
            }

            TestGameState { player_pos, adversary_pos, player_score, adversary_score, max_pos: self.max_pos, board }
        }

        fn eval(&self) -> f64 {
            self.player_score as f64 - self.adversary_score as f64
        }

        fn is_terminal(&self) -> bool {
            self.player_pos == self.adversary_pos || self.board.iter().all(|&spot| spot <= 0)
        }
    }

    impl Display for TestGameState {
        fn fmt(&self, f: &mut Formatter) -> FmtResult {
            let mut board_str = String::from_str("[ ").unwrap();

            for (i, v) in self.board.iter().enumerate() {
                board_str.push_str(format!("{}", v).as_str());

                if i == self.player_pos {
                    board_str.push_str("(P)");
                }
                else if i == self.adversary_pos {
                    board_str.push_str("(A)");
                }

                board_str.push_str(" ");
            }

            board_str.push_str("]");
            write!(f, "{}", board_str)
        }
    }

    type NodeConstructor<'a, State> = fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State>>;

    #[test]
    fn teststate_player_both_actions_returned_when_valid() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let state = TestGameState::start(board);

        assert_eq!(state.actions(TestGameAgent::Player), vec![TestGameAction::Left(1), TestGameAction::Right(1)]);
    }

    #[test]
    fn teststate_adversary_both_actions_returned_when_valid() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let state = TestGameState::start(board);

        assert_eq!(state.actions(TestGameAgent::Adversary), vec![TestGameAction::Left(1), TestGameAction::Right(1)]);
    }

    #[test]
    fn teststate_player_only_left_returned_when_valid() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(max_pos, 0, 0, 0, board);

        assert_eq!(state.actions(TestGameAgent::Player), vec![TestGameAction::Left(1)]);
    }

    #[test]
    fn teststate_adversary_only_left_returned_when_valid() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(0, max_pos, 0, 0, board);

        assert_eq!(state.actions(TestGameAgent::Adversary), vec![TestGameAction::Left(1)]);
    }

    #[test]
    fn teststate_player_only_right_returned_when_valid() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(0, max_pos, 0, 0, board);

        assert_eq!(state.actions(TestGameAgent::Player), vec![TestGameAction::Right(1)]);
    }

    #[test]
    fn teststate_adversary_only_right_returned_when_valid() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(max_pos, 0, 0, 0, board);

        assert_eq!(state.actions(TestGameAgent::Adversary), vec![TestGameAction::Right(1)]);
    }

    #[test]
    fn teststate_player_only_peek_returned_when_valid() {
        let board: Vec<u32> = vec![4, 0, 0, 0, 0, 0];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(0, max_pos, 0, 0, board);

        assert_eq!(state.actions(TestGameAgent::Player), vec![TestGameAction::Peek]);
    }

    #[test]
    fn teststate_adversary_only_peek_returned_when_valid() {
        let board: Vec<u32> = vec![0, 0, 0, 0, 0, 3];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(0, max_pos, 0, 0, board);

        assert_eq!(state.actions(TestGameAgent::Adversary), vec![TestGameAction::Peek]);
    }

    #[test]
    fn teststate_player_no_actions_returned_when_none_valid() {
        let board: Vec<u32> = vec![0, 0, 0, 0, 0, 0];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(0, max_pos, 0, 0, board);

        assert_eq!(state.actions(TestGameAgent::Player).len(), 0);
    }

    #[test]
    fn teststate_adversary_no_actions_returned_when_none_valid() {
        let board: Vec<u32> = vec![0, 0, 0, 0, 0, 0];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(0, max_pos, 0, 0, board);

        assert_eq!(state.actions(TestGameAgent::Adversary).len(), 0);
    }

    #[test]
    fn teststate_player_left_successor_returned_correctly() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let state = TestGameState::start(board);
        let successor = state.successor(TestGameAgent::Player, TestGameAction::Left(1));

        // player moved 1 spot left
        assert_eq!(successor.player_pos(), state.player_pos() - 1);

        // adversary stayed the same
        assert_eq!(successor.adversary_pos(), state.adversary_pos());

        // player score went up
        assert_eq!(successor.player_score(), state.player_score() + state.player_spot());

        // adversary score stayed same
        assert_eq!(successor.adversary_score(), state.adversary_score());

        // spot was overwritten
        assert_eq!(successor.board()[state.player_pos()], 0);
    }

    #[test]
    fn teststate_player_right_successor_returned_correctly() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let state = TestGameState::start(board);
        let successor = state.successor(TestGameAgent::Player, TestGameAction::Right(1));

        // player moved 1 spot left
        assert_eq!(successor.player_pos(), state.player_pos() + 1);

        // adversary stayed the same
        assert_eq!(successor.adversary_pos(), state.adversary_pos());

        // player score went up
        assert_eq!(successor.player_score(), state.player_score() + state.player_spot());

        // adversary score stayed same
        assert_eq!(successor.adversary_score(), state.adversary_score());

        // spot was overwritten
        assert_eq!(successor.board()[state.player_pos()], 0);
    }

    #[test]
    fn teststate_player_peek_successor_returned_correctly() {
        let board: Vec<u32> = vec![4, 0, 0, 0, 0, 0];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(0, max_pos, 0, 0, board);
        let successor = state.successor(TestGameAgent::Player, TestGameAction::Peek);

        // player moved 1 spot left
        assert_eq!(successor.player_pos(), state.player_pos());

        // adversary stayed the same
        assert_eq!(successor.adversary_pos(), state.adversary_pos());

        // player score went up
        assert_eq!(successor.player_score(), state.player_score() + state.player_spot());

        // adversary score stayed same
        assert_eq!(successor.adversary_score(), state.adversary_score());

        // spot was overwritten
        assert_eq!(successor.player_spot(), 0);
    }

    #[test]
    fn teststate_adversary_left_successor_returned_correctly() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let state = TestGameState::start(board);
        let successor = state.successor(TestGameAgent::Adversary, TestGameAction::Left(1));

        // player moved 1 spot left
        assert_eq!(successor.adversary_pos(), state.adversary_pos() - 1);

        // adversary stayed the same
        assert_eq!(successor.player_pos(), state.player_pos());

        // player score went up
        assert_eq!(successor.adversary_score(), state.adversary_score() + state.adversary_spot());

        // adversary score stayed same
        assert_eq!(successor.player_score(), state.player_score());

        // spot was overwritten
        assert_eq!(successor.board()[state.adversary_pos()], 0);
    }

    #[test]
    fn teststate_adversary_right_successor_returned_correctly() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let state = TestGameState::start(board);
        let successor = state.successor(TestGameAgent::Adversary, TestGameAction::Right(1));

        // player didn't move
        assert_eq!(successor.adversary_pos(), state.adversary_pos() + 1);

        // adversary stayed the same
        assert_eq!(successor.player_pos(), state.player_pos());

        // player score went up
        assert_eq!(successor.adversary_score(), state.adversary_score() + state.adversary_spot());

        // adversary score stayed same
        assert_eq!(successor.player_score(), state.player_score());

        // spot was overwritten
        assert_eq!(successor.board()[state.adversary_pos()], 0);
    }

    #[test]
    fn teststate_adversary_peek_successor_returned_correctly() {
        let board: Vec<u32> = vec![4, 0, 0, 0, 0, 0];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(max_pos, 0, 0, 0, board);
        let successor = state.successor(TestGameAgent::Adversary, TestGameAction::Peek);

        // player didn't move
        assert_eq!(successor.player_pos(), state.player_pos());

        // adversary stayed the same
        assert_eq!(successor.adversary_pos(), state.adversary_pos());

        // player score stayed the same
        assert_eq!(successor.adversary_score(), state.adversary_score() + state.adversary_spot());

        // adversary score went up
        assert_eq!(successor.player_score(), state.player_score());

        // spot was overwritten
        assert_eq!(successor.adversary_spot(), 0);
    }

    #[test]
    fn teststate_eval_function_calculated_correctly_positive() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let player_score = 9.0;
        let adversary_score = 15.0;
        let state = TestGameState::new(0, 0, player_score as u32, adversary_score as u32, board);

        assert_eq!(state.eval(), player_score - adversary_score);
    }

    #[test]
    fn teststate_eval_function_calculated_correctly_zero() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let player_score = 9.0;
        let adversary_score = 15.0;
        let state = TestGameState::new(0, 0, player_score as u32, adversary_score as u32, board);

        assert_eq!(state.eval(), player_score - adversary_score);
    }

    #[test]
    fn teststate_eval_function_calculated_correctly_negative() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let player_score = 9.0;
        let adversary_score = 15.0;
        let state = TestGameState::new(0, 0, player_score as u32, adversary_score as u32, board);

        assert_eq!(state.eval(), player_score - adversary_score);
    }

    #[test]
    fn teststate_correctly_nonterminal_when_game_continues() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let state = TestGameState::start(board);

        assert!(!state.is_terminal());
    }

    #[test]
    fn teststate_correctly_terminal_for_collision() {
        let board: Vec<u32> = vec![4, 2, 5, 1, 2, 3];
        let pos = 2;
        let state = TestGameState::new(pos, pos, 0, 0, board);

        assert!(state.is_terminal());
    }

    #[test]
    fn teststate_correctly_terminal_when_no_moves() {
        let board: Vec<u32> = vec![0, 0, 0, 0, 0, 0];
        let max_pos = board.len() - 1;
        let state = TestGameState::new(0, max_pos, 0, 0, board);

        assert!(state.is_terminal());
    }

    #[test]
    fn minimax_agent_created_correctly_no_max_depth() {
        let agent = AdversarialSearchAgent::<TestGameState>::minimax(
            vec![TestGameAgent::Player, TestGameAgent::Adversary],
            None
        );

        assert_eq!(agent.n_policies(), 2);
        assert_eq!(agent.max_depth(), None);

        let expected_policies: Vec<NodeConstructor<TestGameState>> = vec![MaximizerNode::new, MinimizerNode::new];

        for (i, policy) in agent.policies().iter().enumerate() {
            let expected_policy = &expected_policies[i];
            assert!(policy.node() == *expected_policy);
        }
    }

    #[test]
    fn minimax_agent_created_correctly_with_max_depth() {
        let max_depth = 5;
        let agent = AdversarialSearchAgent::<TestGameState>::minimax(
            vec![TestGameAgent::Player, TestGameAgent::Adversary],
            Some(max_depth)
        );

        assert_eq!(agent.n_policies(), 2);
        assert_eq!(agent.max_depth(), Some(max_depth));

        let expected_policies: Vec<NodeConstructor<TestGameState>> = vec![MaximizerNode::new, MinimizerNode::new];

        for (i, policy) in agent.policies().iter().enumerate() {
            let expected_policy = &expected_policies[i];
            assert!(policy.node() == *expected_policy);
        }
    }

    #[test]
    fn expectimax_agent_created_correctly_no_max_depth() {
        let agent = AdversarialSearchAgent::<TestGameState>::expectimax(
            vec![TestGameAgent::Player, TestGameAgent::Adversary],
            None
        );

        assert_eq!(agent.n_policies(), 2);
        assert_eq!(agent.max_depth(), None);

        let expected_policies: Vec<NodeConstructor<TestGameState>> = vec![MaximizerNode::new, ChanceNode::new];

        for (i, policy) in agent.policies().iter().enumerate() {
            let expected_policy = &expected_policies[i];
            assert!(policy.node() == *expected_policy);
        }
    }

    #[test]
    fn expectimax_agent_created_correctly_with_max_depth() {
        let max_depth = 5;
        let agent = AdversarialSearchAgent::<TestGameState>::expectimax(
            vec![TestGameAgent::Player, TestGameAgent::Adversary],
            Some(max_depth)
        );

        assert_eq!(agent.n_policies(), 2);
        assert_eq!(agent.max_depth(), Some(max_depth));

        let expected_policies: Vec<NodeConstructor<TestGameState>> = vec![MaximizerNode::new, ChanceNode::new];

        for (i, policy) in agent.policies().iter().enumerate() {
            let expected_policy = &expected_policies[i];
            assert!(policy.node() == *expected_policy);
        }
    }

    #[test]
    fn custom_policy_agent_created_correctly_no_max_depth() {
        let agent = AdversarialSearchAgent::<TestGameState>::new(
            vec![AdversarialSearchPolicy::new(TestGameAgent::Player, ChanceNode::new), AdversarialSearchPolicy::new(TestGameAgent::Adversary, MinimizerNode::new)],
            None
        );

        assert_eq!(agent.n_policies(), 2);
        assert_eq!(agent.max_depth(), None);

        let expected_policies: Vec<NodeConstructor<TestGameState>> = vec![ChanceNode::new, MinimizerNode::new];

        for (i, policy) in agent.policies().iter().enumerate() {
            let expected_policy = &expected_policies[i];
            assert!(policy.node() == *expected_policy);
        }
    }

    #[test]
    fn custom_policy_agent_created_correctly_with_max_depth() {
        let max_depth = 5;
        let agent = AdversarialSearchAgent::<TestGameState>::new(
            vec![AdversarialSearchPolicy::new(TestGameAgent::Player, ChanceNode::new), AdversarialSearchPolicy::new(TestGameAgent::Adversary, MinimizerNode::new)],
            Some(max_depth)
        );

        assert_eq!(agent.n_policies(), 2);
        assert_eq!(agent.max_depth(), Some(max_depth));

        let expected_policies: Vec<NodeConstructor<TestGameState>> = vec![ChanceNode::new, MinimizerNode::new];

        for (i, policy) in agent.policies().iter().enumerate() {
            let expected_policy = &expected_policies[i];
            assert!(policy.node() == *expected_policy);
        }
    }
/*
    #[test]
    fn minimax_agent_adv_search_returns_correct_result_no_max_depth() {

    }

    #[test]
    fn minimax_agent_adv_search_returns_correct_result_with_max_depth() {
        // TODO
    }

    #[test]
    fn expectimax_agent_adv_search_returns_correct_result_no_max_depth() {
        // TODO
    }

    #[test]
    fn expectimax_agent_adv_search_returns_correct_result_with_max_depth() {
        // TODO
    }

    #[test]
    fn custom_policy_agent_adv_search_returns_correct_result_no_max_depth() {
        // TODO
    }

    #[test]
    fn custom_policy_agent_adv_search_returns_correct_result_with_max_depth() {
        // TODO
    }

    #[test]
    fn terminal_node_created_correctly() {
        let state = TestGameState::new();
        let node = TerminalNode::new(state.clone());

        assert_eq!(*(node.state()), state);

        match node.successors() {
            Some(_) => panic!("terminal node has successors"),
            None => {},
        }
    }

    #[test]
    fn terminal_node_utility_calculated_correctly() {
        let state = TestGameState::new();
        let node = TerminalNode::new(state.clone());

        let (utility, action) = node.utility();

        assert_eq!(utility, state.eval());
        assert_eq!(action, None);
    }

    #[test]
    fn min_node_created_correctly() {
        let state = TestGameState::new();
        let agent = 'P';
        let successors: Vec<AdversarialSearchSuccessor<TestGameState>> = state.actions(agent).iter().map(
            |a| AdversarialSearchSuccessor::new(
                *a,
                TerminalNode::new(state.successor(agent, *a))
            )
        ).collect();
        let n_successors = successors.len();

        let node = MinimizerNode::new(state.clone(), successors);

        assert_eq!(*(node.state()), state);
        assert_eq!(node.successors().unwrap().len(), n_successors);
    }

    #[test]
    fn min_node_utility_correct_one_successor() {
        let state = TestGameState::new();

        let agent = 'P';
        let action = 1;
        let successor_state = state.successor(agent, action);

        let successors = vec![
            AdversarialSearchSuccessor::new(
                action,
                TerminalNode::new(successor_state.clone())
            )
        ];

        let node = MinimizerNode::new(state.clone(), successors);
        let (utility, optimal_action) = node.utility();

        assert_eq!(utility, successor_state.eval());
        assert_eq!(optimal_action.unwrap(), action);
    }

    #[test]
    fn min_node_utility_correct_many_successors_no_tie() {
        let state = TestGameState::new();

        let agent = 'P';
        let successors = vec![
            AdversarialSearchSuccessor::new(-1, TerminalNode::new(state.successor(agent, -1))),
            AdversarialSearchSuccessor::new(
                1,
                TerminalNode::new(state.successor(agent, 1).successor(agent, 1))
            )
        ];

        let min_utility = -2.0;
        let action = 1;

        let node = MinimizerNode::new(state.clone(), successors);
        let (utility, optimal_action) = node.utility();

        assert_eq!(utility, min_utility);
        assert_eq!(optimal_action.unwrap(), action);
    }

    #[test]
    fn min_node_utility_correct_many_successors_tie() {
        let state = TestGameState::new();
        let agent = 'P';
        let successors: Vec<AdversarialSearchSuccessor<TestGameState>> = state.actions(agent).iter().map(
            |a| AdversarialSearchSuccessor::new(
                *a,
                TerminalNode::new(state.successor(agent, *a))
            )
        ).collect();

        let min_successor = &successors[0];
        let (min_utility, _) = min_successor.node().utility();
        let optimal_action = min_successor.action();

        let node = MinimizerNode::new(state.clone(), successors);
        let (utility, action) = node.utility();

        assert_eq!(utility, min_utility);
        assert_eq!(action.unwrap(), optimal_action);
    }

    #[test]
    fn max_node_created_correctly() {
        let state = TestGameState::new();
        let agent = 'P';
        let successors: Vec<AdversarialSearchSuccessor<TestGameState>> = state.actions(agent).iter().map(
            |a| AdversarialSearchSuccessor::new(
                *a,
                TerminalNode::new(state.successor(agent, *a))
            )
        ).collect();
        let n_successors = successors.len();

        let node = MaximizerNode::new(state.clone(), successors);

        assert_eq!(*(node.state()), state);
        assert_eq!(node.successors().unwrap().len(), n_successors);
    }

    #[test]
    fn max_node_utility_correct_one_successor() {
        let state = TestGameState::new();

        let agent = 'P';
        let action = 1;
        let successor_state = state.successor(agent, action);

        let successors = vec![
            AdversarialSearchSuccessor::new(
                action,
                TerminalNode::new(successor_state.clone())
            )
        ];

        let node = MaximizerNode::new(state.clone(), successors);
        let (utility, optimal_action) = node.utility();

        assert_eq!(utility, successor_state.eval());
        assert_eq!(optimal_action.unwrap(), action);
    }

    #[test]
    fn max_node_utility_correct_many_successors_no_tie() {
        let state = TestGameState::new();

        let agent = 'P';
        let successors = vec![
            AdversarialSearchSuccessor::new(-1, TerminalNode::new(state.successor(agent, -1))),
            AdversarialSearchSuccessor::new(
                1,
                TerminalNode::new(state.successor(agent, 1).successor(agent, 1))
            )
        ];

        let max_utility = -1.0;
        let optimal_action = -1;

        let node = MaximizerNode::new(state.clone(), successors);
        let (utility, action) = node.utility();

        assert_eq!(utility, max_utility);
        assert_eq!(action.unwrap(), optimal_action);
    }

    #[test]
    fn max_node_utility_correct_many_successors_tie() {
        let state = TestGameState::new();
        let agent = 'P';
        let successors: Vec<AdversarialSearchSuccessor<TestGameState>> = state.actions(agent).iter().map(
            |action| AdversarialSearchSuccessor::new(
                *action,
                TerminalNode::new(state.successor(agent, *action))
            )
        ).collect();

        let max_successor = &successors[0];
        let (max_utility, _) = max_successor.node().utility();
        let optimal_action = max_successor.action();

        let node = MaximizerNode::new(state.clone(), successors);
        let (utility, action) = node.utility();

        assert_eq!(utility, max_utility);
        assert_eq!(action.unwrap(), optimal_action);
    }

    #[test]
    fn chance_node_created_correctly() {
        let state = TestGameState::new();
        let agent = 'P';
        let successors: Vec<AdversarialSearchSuccessor<TestGameState>> = state.actions(agent).iter().map(
            |a| AdversarialSearchSuccessor::new(
                *a,
                TerminalNode::new(state.successor(agent, *a))
            )
        ).collect();
        let n_successors = successors.len();

        let node = ChanceNode::new(state.clone(), successors);

        assert_eq!(*(node.state()), state);
        assert_eq!(node.successors().unwrap().len(), n_successors);
    }

    #[test]
    fn chance_node_utility_correct_one_successor() {
        let state = TestGameState::new();

        let agent = 'P';
        let action = 1;
        let successor_state = state.successor(agent, action);

        let successors = vec![
            AdversarialSearchSuccessor::new(
                action,
                TerminalNode::new(successor_state.clone())
            )
        ];

        let node = ChanceNode::new(state.clone(), successors);
        let (utility, optimal_action) = node.utility();

        assert_eq!(utility, successor_state.eval());
        assert_eq!(optimal_action, None);
    }

    #[test]
    fn chance_node_utility_correct_many_successors() {
        let state = TestGameState::new();

        let agent = 'P';
        let successors = vec![
            AdversarialSearchSuccessor::new(-1, TerminalNode::new(state.successor(agent, -1))),
            AdversarialSearchSuccessor::new(
                1,
                TerminalNode::new(state.successor(agent, 1).successor(agent, 1))
            )
        ];

        let exp_utility = -1.5;

        let node = ChanceNode::new(state.clone(), successors);
        let (utility, action) = node.utility();

        assert_eq!(utility, exp_utility);
        assert_eq!(action, None);
    }*/
}
