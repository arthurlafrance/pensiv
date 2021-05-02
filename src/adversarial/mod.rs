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
//! You may notice that the `AdversarialSearchState` trait interacts with agents through `usize`. This is more generally true of
//! the entire adversarial search implementation -- rather than worry about the specific agent type(s) that may be involved with a particular
//! search problem, you only need to assign some `usize` to each agent such that it can be used as an identifier during adversarial search. Basically,
//! each agent needs a `usize` that can be used to uniquely identify it within the state representation you define (the documentation for the module will
//! cover where exactly this identifier needs to be valid).
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

pub mod tree;
pub mod state;

use self::state::AdversarialSearchState;
use self::tree::{
    AdversarialSearchNode,
    AdversarialSearchSuccessor,
    MinimizerNode,
    MaximizerNode,
    ChanceNode,
    TerminalNode
};

use num_traits::Float;


/// An agent that performs adversarial search.
///
/// `AdversarialSearchAgent` is parameterized by the `State` type, which represents the state of the adversarial search problem and thus must
/// implement `AdversarialSearchState`.
pub struct AdversarialSearchAgent<'s, State: AdversarialSearchState> {
    policies: Vec<AdversarialSearchPolicy<'s, State>>,
    n_policies: usize,
    max_depth: Option<usize>,
}

impl<'s, State: 's + AdversarialSearchState> AdversarialSearchAgent<'s, State> {
    /// Creates and returns a new adversarial search agent.
    ///
    /// `AdversarialSearchAgent` is designed to provide flexible game tree creation and central adversarial search evaluation regardless of
    /// game tree layout. This is accomplished by specifying the agent's "policy", i.e. the type of node that the agent uses, as well as the
    /// strategies of its adversaries. In this way, the game tree is organized into layers whose node type is specified by the factory function
    /// parameters. Finally, an optional max depth can be specified to upper-bound adversarial search at some maximum depth in the game tree.
    ///
    /// Note that the array of policies must follow a specific order for adversarial search to work correctly: the agent's policy must be first,
    /// followed by the adversaries' policies in whatever order you choose. This does mean that, because policies are evaluated
    /// sequentially, the adversaries' policies will be evaluated in the order in which they appear in the vector, so keep that in mind when using
    /// `AdversarialSearchAgent`.
    pub fn new(policies: Vec<AdversarialSearchPolicy<'s, State>>, max_depth: Option<usize>) -> AdversarialSearchAgent<State> {
        let n_policies = policies.len();

        AdversarialSearchAgent { policies, n_policies, max_depth }
    }

    // NOTE: since minimax only really works when you have one single utility value, is it even worth including?

    /// Creates and returns a minimax agent.
    ///
    /// This is a convenience function for creating an adversarial search agent that performs minimax adversarial search. Thus,
    /// the function requires you to specify the agents to include in the game tree as well as an optional maximum depth.
    ///
    /// NOTE: minimax works by assigning both policies to the same utility -- the agent's policy maximizes it, the adversary's policy minimizes it.
    /// So, your state representation should return a single utility value "shared" by both agents, so that minimax search works correctly (any other
    /// utility values will be ignored).
    pub fn minimax(max_depth: Option<usize>) -> AdversarialSearchAgent<'s, State> where State::Utility: PartialOrd {
        let agent_policy = AdversarialSearchPolicy::new(0, MaximizerNode::new);
        let adversary_policy = AdversarialSearchPolicy::new(0, MinimizerNode::new);
        let policies = vec![agent_policy, adversary_policy];

        AdversarialSearchAgent { policies, n_policies: 2, max_depth }
    }

    /// Creates and returns a "max-n" agent, a generalization of minimax to many player games.
    ///
    /// Max-n generalizes minimax to more than 2 players by assigning each player with a maximizer node -- each player attempts to maximize its own utility value.
    ///
    /// Note that agents are given priority in adversarial search sequentially based on the order that they appear in the vector parameter. Adversarial search will be performed
    /// with respect to the first agent (i.e. the optimal utility & action for that agent will be returned), and all following agents will be treated as its adversaries.
    pub fn max_n(agents: Vec<usize>, max_depth: Option<usize>) -> AdversarialSearchAgent<'s, State> where State::Utility: PartialOrd {
        let n_policies = agents.len();
        let policies = agents.into_iter().map(|a| AdversarialSearchPolicy::new(a, MaximizerNode::new)).collect();

        AdversarialSearchAgent { policies, n_policies, max_depth }
    }

    /// Creates and returns a expectimax agent.
    ///
    /// Like `minimax()` and `max_n()`, this is a convenience function; it creates an adversarial search agent whose adversaries are assumed to use chance nodes, and the agent is assumed to use maximizer nodes. Thus,
    /// the function requires you to specify the agents to include in the game tree as well as an optional maximum depth.
    ///
    /// Note that, just as the agents in the `minimax()` function appear in the vector in the order that they're evaluated, so too do the agents
    /// in the `agents` vector parameter. The player agent should appear first, and adversary agents should appear in whatever order you wish to
    /// evaluate them.
    pub fn expectimax(agent: usize, adversaries: Vec<usize>, max_depth: Option<usize>) -> AdversarialSearchAgent<'s, State> where State::Utility: PartialOrd + Float {
        let n_policies = adversaries.len() + 1;
        let mut policies = Vec::with_capacity(n_policies);

        let agent_policy = AdversarialSearchPolicy::new(agent, MaximizerNode::new);
        policies.push(agent_policy);

        for adversary in adversaries.into_iter() {
            let policy = AdversarialSearchPolicy::new(adversary, ChanceNode::new);
            policies.push(policy);
        }

        AdversarialSearchAgent { policies, n_policies, max_depth }
    }

    /// Return a reference to the policies for the agent.
    ///
    /// The player's policy will be the first element, and the adversaries' policies will appear sequentially after it.
    pub fn policies(&self) -> &[AdversarialSearchPolicy<'s, State>] {
        &self.policies
    }

    /// Returns the number of policies in the game tree, equivalent to the number of players in the game.
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

    /// Return the optimal utility & action for the agent to take from the given state, if it exists.
    ///
    /// This function performs adversarial search -- it constructs a game tree starting at the current state according to the agent's known
    /// policies, and evaluates it in order to determine which action will lead to optimal utility, if any.
    ///
    /// Note that this method assumes that `state` is a state from which the agent acts first; all adversaries will act in the order that their strategies
    /// were specified.
    ///
    /// One important note is that performing adversarial search with no maximum depth may lead to infinite recursion, if there exists
    /// some way to transition between states in a cycle. (Think of this as a cyclic state space graph, which would obviously result in a never-ending tree).
    /// Thus, be cognizant of this, and use infinite-depth adversarial search at your own risk.
    pub fn eval(&self, state: State) -> (State::Utility, Option<State::Action>) {
        let root = self.make_node(state, 0, 0);
        let (utility, optimal_action) = root.eval();
        let first_agent = self.policies[0].agent();

        (utility[first_agent], optimal_action)
    }

    fn make_node(&self, state: State, policy_index: usize, depth: usize) -> Box<dyn AdversarialSearchNode<'s, State> + 's> {
        // if node should be terminal, make terminal
        // else, make according to policy and add successors
        if state.is_terminal() || (self.max_depth != None && depth == self.max_depth.unwrap()) {
            TerminalNode::new(state)
        }
        else {
            let policy = &self.policies[policy_index];
            let agent = policy.agent();

            let mut successors = vec![];
            let node_constructor = policy.node_constructor();

            for action in state.actions(agent).into_iter() {
                let successor_state = state.successor(agent, action);
                let child = self.make_node(successor_state, (policy_index + 1) % self.n_policies, depth + 1);

                let successor = AdversarialSearchSuccessor::new(action, child);
                successors.push(successor);
            }

            node_constructor(state, agent, successors)
        }
    }
}


/// A policy in adversarial search: an agent and the node type that models it.
#[derive(Debug)]
pub struct AdversarialSearchPolicy<'s, State: AdversarialSearchState> {
    agent: usize,
    node_constructor: fn(State, usize, Vec<AdversarialSearchSuccessor<'s, State>>) -> Box<dyn AdversarialSearchNode<'s, State> + 's>,
}

impl<'s, State: AdversarialSearchState> AdversarialSearchPolicy<'s, State> {
    /// Creates and returns a new policy object.
    pub fn new(agent: usize, node_constructor: fn(State, usize, Vec<AdversarialSearchSuccessor<'s, State>>) -> Box<dyn AdversarialSearchNode<'s, State> + 's>) -> AdversarialSearchPolicy<State> {
        AdversarialSearchPolicy { agent, node_constructor }
    }

    /// Returns the agent to which the policy applies.
    pub fn agent(&self) -> usize {
        self.agent
    }

    /// Returns a function pointer to the constructor for the policy's node type.
    pub fn node_constructor(&self) -> fn(State, usize, Vec<AdversarialSearchSuccessor<'s, State>>) -> Box<dyn AdversarialSearchNode<'s, State> + 's> {
        self.node_constructor
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::{Display, Formatter, Result as FmtResult};
    use std::str::FromStr;

    // ok new test search problem idea:
        // given n players (1 agent + (n - 1) adversaries), say there are 3n - 1 "bins", each of which stores some number
        // each player is given some "cards", each of which has a numerical value
        // on any player's turn, they can choose to place their card on top of one of the bins
        //   only if it's greater than the value of the current top card (maybe add more actions idk)
        // another potential action: allow for splitting a card in half -- can split a card into two cards whose values sum to the original card's value, roughly splitting it in half
        // game ends when no players have cards left -- each player's score is the sum of the values of all of their cards on
        //   top of a bin
    // good because it's deterministic, always has a fixed depth, and can be extended to an arbitrary number of players

    struct TestGameState {
        cards: Vec<Vec<bool>>, // true means still available to use
        remaining_cards: Vec<usize>,
        bins: Vec<(usize, usize)>, // format is (player, card value)
    }

    impl TestGameState {
        const N_CARDS: usize = 6;

        fn start(n_players: usize) -> TestGameState {
            if n_players == 0 {
                panic!("must have at least one player");
            }

            let n_bins = 3 * n_players - 1;

            TestGameState {
                cards: vec![vec![true; TestGameState::N_CARDS]; n_players],
                remaining_cards: vec![TestGameState::N_CARDS; n_players],
                bins: vec![(0, 0); n_bins]
            }
        }

        fn new(cards: Vec<Vec<bool>>, remaining_cards: Vec<usize>, bins: Vec<(usize, usize)>) -> TestGameState {
            TestGameState { cards, remaining_cards, bins }
        }
    }

    impl Display for TestGameState {
        fn fmt(&self, f: &mut Formatter) -> FmtResult {
            // write bins
            write!(f, "Bins: {:?}\n\n", self.bins)?;

            // write each player's remaining cards
            for (player, cards) in self.cards.iter().enumerate() {
                write!(f, "P{} remaining: ", player + 1)?;

                for (card, available) in cards.iter().enumerate() {
                    if *available {
                        write!(f, "{}", card + 1)?;
                    }
                    else {
                        write!(f, "_")?;
                    }

                    write!(f, " ")?;
                }

                write!(f, "\n")?;
            }

            Ok(())
        }
    }

    impl AdversarialSearchState for TestGameState {
        type Action = TestGameAction;
        type Utility = usize;

        fn n_agents(&self) -> usize {
            self.cards.len()
        }

        fn is_terminal(&self) -> bool {
            self.remaining_cards.iter().copied().any(|cards_left| cards_left > 0)
        }

        fn actions(&self, agent: usize) -> Vec<TestGameAction> {
            if agent > self.n_agents() - 1 {
                panic!("invalid agent (ID too large)");
            }

            //

            self.cards[agent].iter().copied()
                .enumerate() // add index to iterator
                .filter(|(_, card)| *card) // filter for only available cards
                .map(|(val, _)| val) // extract index only
                .collect() // collect into vector
        }

        fn successor(&self, agent: usize, action: usize) -> TestGameState {
            if agent > self.n_agents() - 1 {
                panic!("invalid agent (ID too large)");
            }

            if action > TestGameState::N_CARDS - 1 {
                panic!("invalid action (too large)");
            }

            // validate that agent can actually take that action
            if !self.cards[agent][action] {
                panic!("card already used for agent");
            }

            // clone cards, remaining_cards, bins
            let mut cards = self.cards.clone();
            let mut remaining_cards = self.remaining_cards.clone();
            let mut bins = self.bins.clone();

            // modify game state & return successor
            cards[agent][action] = false;
            remaining_cards[agent] -= 1;
            // bins[]
        }
    }

    struct TestGameAction {
        pub bin: usize,
        pub card: usize,
    }

    impl TestGameAction {
        pub fn new(bin: usize, card: usize) -> TestGameAction {
            TestGameAction { bin, card }
        }
    }

    type NodeConstructor<'a, State> = fn(State, Vec<AdversarialSearchSuccessor<'a, State>>) -> Box<dyn AdversarialSearchNode<'a, State>>;
/*
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
