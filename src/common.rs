use bincode::{Decode, Encode};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode, Serialize)]
pub struct Label {
    pub index: usize,
}
