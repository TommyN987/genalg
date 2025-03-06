pub mod elitist;
pub mod rank;
pub mod roulette;
pub mod selection_strategy;
pub mod tournament;

pub use selection_strategy::SelectionStrategy;
pub use elitist::ElitistSelection;
pub use tournament::TournamentSelection;
pub use roulette::RouletteWheelSelection;
pub use rank::RankBasedSelection;