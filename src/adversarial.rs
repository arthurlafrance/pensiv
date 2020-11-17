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
    fn successor(&self, action: Self::Action) -> Self;

    /// Returns the utility of the current state according to the evaluation function.
    /// 
    /// Use this function to define a custom evaluation function by which to determine the utility of a custom state.
    fn eval(&self) -> Self::Utility;
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
    /// Returns the state stored at this node.
    /// 
    /// This method is typically simply an accessor method of the node's internal state field.
    fn state(&self) -> State;

    /// Returns the node's children.
    /// 
    /// Note that a node's children don't need to be of the same type; as long as they use the same state as this node, 
    /// they're valid children. As a concrete example, this means that a minimizer node can have maximizer nodes as its 
    /// children, as long as all nodes are generic to the same state.
    fn children(&self) -> Vec<Box<dyn GameTreeNode<State>>>;

    /// Returns the node's utility and the action required to achieve that utility.
    /// 
    /// How a node's utility is calculated is determined by the type of node in this method. For example, minimizer nodes 
    /// will minimize the utility of their children; maximizer nodes do the opposite. Thus, this method is how to determine 
    /// the method by which a node calculates its own usility, potentially relative to its children's utility.
    fn utility(&self) -> (State::Utility, State::Action);
}