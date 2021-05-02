// TODO: module docs

/// Base trait for states in a game tree
///
/// Implement this trait for your adversarial search problem's custom state representation. Note that this trait defines
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

    /// Describes the utility achieved at the end of a game _by any single player_.
    ///
    /// This should almost always be some kind of numeric type -- note the important requirement that the utility type describes
    /// utility for a single player only. The `eval()` method returns a vector of utilities (one per player), where each element is an instance
    /// of this utility associated type.
    type Utility: Copy;

    /// Returns a vector containing the legal actions that can be taken at the current state by the given agent, determined
    /// by its ID.
    fn actions(&self, agent: usize) -> Vec<Self::Action>;

    // TODO: i think this should return a result to indicate legality of action
    /// Returns the successor state that arises from the given agent taking the given action at the current state.
    ///
    /// Note that this function assumes that the action being taken is a valid action to take from the current state; any
    /// violation of this precondition is undefined behavior, and can be handled at the developer's discretion.
    fn successor(&self, agent: usize, action: Self::Action) -> Self;

    /// Returns the evaluation function value of the current state for each player in the game.
    ///
    /// For terminal states, this should be the utility of the state for each agent (i.e. their "score").
    /// For non-terminal states, this should be a heuristic estimate of the state's utility for each agent, i.e. your
    /// best guess of their score.
    fn eval(&self) -> Vec<Self::Utility>;

    /// Returns `true` if the state is a terminal state, `false` otherwise.
    ///
    /// This method must guarantee to be true when the current state has no successors, and false otherwise;
    /// if not, adversarial search may not work as intended.
    fn is_terminal(&self) -> bool;

    /// Returns the number of agents in the game.
    fn n_agents(&self) -> usize;
}
