use std::fmt;

/// Result alias used by storage backends.
pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Result alias for pure-compute operations on a [`crate::CompiledDag`].
pub type ComputeResult<T> = std::result::Result<T, ComputeError>;

/// Result alias for operations that touch the model's storage.
pub type ModelResult<T> = std::result::Result<T, ModelError>;

/// Errors raised by storage backend implementations.
///
/// Backends should convert their native errors (database, network, serialization
/// failures) into one of these variants. The `Backend` variant is intended as a
/// catch-all for transport-level failures; prefer the more specific variants
/// when applicable.
#[derive(Debug, Clone, PartialEq)]
pub enum StorageError {
    /// A backend-level error such as a connection failure or query error.
    Backend {
        /// Human-readable description of the failure.
        message: String,
    },

    /// A value retrieved from the backend could not be deserialized into the
    /// expected shape (e.g. malformed JSON, schema drift).
    Deserialization {
        /// The storage key whose value failed to deserialize.
        key: String,
        /// Human-readable description of the failure.
        message: String,
    },

    /// A value could not be serialized for storage.
    Serialization {
        /// The storage key whose value failed to serialize.
        key: String,
        /// Human-readable description of the failure.
        message: String,
    },
}

impl StorageError {
    /// Convenience constructor for [`StorageError::Backend`] from any string-like value.
    pub fn backend(message: impl Into<String>) -> Self {
        StorageError::Backend { message: message.into() }
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Backend { message } => {
                write!(f, "Storage backend error: {}", message)
            }
            StorageError::Deserialization { key, message } => {
                write!(f, "Failed to deserialize value for key '{}': {}", key, message)
            }
            StorageError::Serialization { key, message } => {
                write!(f, "Failed to serialize value for key '{}': {}", key, message)
            }
        }
    }
}

impl std::error::Error for StorageError {}

/// Errors raised by pure-compute operations on a [`crate::CompiledDag`].
///
/// These never involve storage. Functions returning [`ComputeResult`] operate
/// entirely in-memory on an already-fetched DAG.
#[derive(Debug, Clone, PartialEq)]
pub enum ComputeError {
    /// A propagation assignment violated a primitive's declared inherent bound.
    NodeOutOfBounds {
        /// The id of the offending primitive node.
        node_id: String,
        /// The bound supplied by the caller.
        got_bound: (i32, i32),
        /// The inherent bound declared on the primitive.
        expected_bound: (i32, i32),
    },

    /// The bound-tightening loop did not converge before the iteration cap.
    MaxIterationsExceeded {
        /// The configured iteration cap that was exceeded.
        max_iters: usize,
    },

    /// A cycle was detected in the DAG, which violates the acyclic property.
    CycleDetected {
        /// The id of a node that participates in the cycle.
        node_id: String,
    },
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeError::NodeOutOfBounds {
                node_id,
                got_bound,
                expected_bound,
            } => write!(
                f,
                "Node '{}' out of bounds: got {:?}, expected {:?}",
                node_id, got_bound, expected_bound
            ),
            ComputeError::MaxIterationsExceeded { max_iters } => {
                write!(f, "Max iterations exceeded during tightening: {}", max_iters)
            }
            ComputeError::CycleDetected { node_id } => {
                write!(f, "Cycle detected in DAG at node '{}'", node_id)
            }
        }
    }
}

impl std::error::Error for ComputeError {}

/// Errors raised by operations that touch the model's storage.
///
/// Covers both model-level validation (e.g. referencing an unknown node when
/// building a constraint) and underlying backend failures.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// A constraint was created with no coefficient variables, which is invalid.
    EmptyConstraint,

    /// A referenced node was not found in the storage.
    NodeNotFound {
        /// The id that was looked up but not present.
        node_id: String,
    },

    /// A delete was attempted on a node that other nodes still reference.
    NodeReferenced {
        /// The id of the node whose deletion was rejected.
        node_id: String,
        /// Ids of nodes that still hold references to `node_id`.
        referencing_nodes: Vec<String>,
    },

    /// An error originating from the storage backend.
    Backend(StorageError),
}

impl From<StorageError> for ModelError {
    fn from(err: StorageError) -> Self {
        ModelError::Backend(err)
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::EmptyConstraint => write!(
                f,
                "Constraint cannot be empty; at least one coefficient is required"
            ),
            ModelError::NodeNotFound { node_id } => {
                write!(f, "Node '{}' not found in storage", node_id)
            }
            ModelError::NodeReferenced {
                node_id,
                referencing_nodes,
            } => write!(
                f,
                "Cannot delete node '{}'; it is referenced by nodes: {:?}",
                node_id, referencing_nodes
            ),
            ModelError::Backend(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for ModelError {}
