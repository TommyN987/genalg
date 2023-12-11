use rand::{rngs::ThreadRng, Rng};
use std::collections::VecDeque;

pub struct RandomNumberGenerator {
    pub rng: ThreadRng,
}

impl RandomNumberGenerator {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    pub fn fetch_uniform(&mut self, from: f32, to: f32, num: usize) -> VecDeque<f32> {
        let mut uniform_numbers = VecDeque::new();
        uniform_numbers.extend((0..num).map(|_| self.rng.gen_range(from..to)));
        uniform_numbers
    }
}
