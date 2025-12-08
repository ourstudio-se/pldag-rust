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

    /// An operation was attempted on an invalid node type
    InvalidNodeType {
        node_id: String,
        expected: String,
        found: String,
    },

    /// Propagation failed due to missing dependencies
    MissingDependencies {
        node_id: String,
        missing: Vec<String>,
    },

    /// Storage operation failed
    StorageError {
        message: String,
    },

    /// Invalid bounds or constraint parameters
    InvalidConstraint {
        message: String,
    },

    /// Polyhedron conversion error
    ConversionError {
        message: String,
    },

    /// Solver-related errors (when GLPK feature is enabled)
    #[cfg(feature = "glpk")]
    SolverError {
        message: String,
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
            PldagError::InvalidNodeType { node_id, expected, found } => {
                write!(
                    f,
                    "Invalid node type for '{}': expected {}, found {}",
                    node_id, expected, found
                )
            }
            PldagError::MissingDependencies { node_id, missing } => {
                write!(
                    f,
                    "Missing dependencies for node '{}': {:?}",
                    node_id, missing
                )
            }
            PldagError::StorageError { message } => {
                write!(f, "Storage error: {}", message)
            }
            PldagError::InvalidConstraint { message } => {
                write!(f, "Invalid constraint: {}", message)
            }
            PldagError::ConversionError { message } => {
                write!(f, "Conversion error: {}", message)
            }
            #[cfg(feature = "glpk")]
            PldagError::SolverError { message } => {
                write!(f, "Solver error: {}", message)
            }
        }
    }
}

impl std::error::Error for PldagError {}

// Conversions from common error types
impl From<std::io::Error> for PldagError {
    fn from(err: std::io::Error) -> Self {
        PldagError::StorageError {
            message: err.to_string(),
        }
    }
}
