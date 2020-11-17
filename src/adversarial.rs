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
    pub fn actions(&self) -> Vec<Self::Action>;

    // TODO: i think this should return a result to indicate legality of action
    /// Returns the successor state that arises from taking the given action at the current state.
    pub fn successor(&self, action: Self::Action) -> Self;

    /// Returns the utility of the current state according to the evaluation function.
    /// 
    /// Use this function to define a custom evaluation function by which to determine the utility of a custom state.
    pub fn eval(&self) -> Self::Utility;
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
    pub fn node(&self, state: State) -> Self::Node;
}