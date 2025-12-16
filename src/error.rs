use std::fmt;

/// Result type alias for pldag operations
pub type Result<T> = std::result::Result<T, PldagError>;

/// Error types that can occur during pldag operations
#[derive(Debug, Clone, PartialEq)]
pub enum PldagError {
    /// A cycle was detected in the DAG, which violates the acyclic property
    CycleDetected {
        node_id: String,
    },

    /// A referenced node was not found in the storage
    NodeNotFound {
        node_id: String,
    },

    // Node out of bounds error
    NodeOutOfBounds {
        node_id: String,
        got_bound: (i32, i32),
        expected_bound: (i32, i32),
    },

    // Max iterations exceeded during tightening
    MaxIterationsExceeded {
        max_iters: usize,
    },

    // When deleting a node that is referenced by other nodes
    NodeReferenced {
        node_id: String,
        referencing_nodes: Vec<String>,
    },
}

impl fmt::Display for PldagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PldagError::CycleDetected { node_id } => {
                write!(f, "Cycle detected in DAG at node '{}'", node_id)
            }
            PldagError::NodeNotFound { node_id } => {
                write!(f, "Node '{}' not found in storage", node_id)
            }
            PldagError::NodeOutOfBounds {
                node_id,
                got_bound,
                expected_bound,
            } => {
                write!(
                    f,
                    "Node '{}' out of bounds: got {:?}, expected {:?}",
                    node_id, got_bound, expected_bound
                )
            }
            PldagError::MaxIterationsExceeded { max_iters } => {
                write!(f, "Max iterations exceeded during tightening: {}", max_iters)
            }
            PldagError::NodeReferenced {
                node_id,
                referencing_nodes,
            } => {
                write!(
                    f,
                    "Cannot delete node '{}'; it is referenced by nodes: {:?}",
                    node_id, referencing_nodes
                )
            }
        }
    }
}

impl std::error::Error for PldagError {}