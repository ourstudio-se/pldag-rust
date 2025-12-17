use crate::pldag::Node;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Abstract interface for key-value storage backends
pub trait KeyValueStore: Send + Sync {
    /// Get all key-value pairs
    fn get_all(&self) -> HashMap<String, Value>;

    /// Get value for key
    fn get(&self, key: &str) -> Option<Value>;

    /// Set value for key
    fn set(&self, key: &str, value: Value);

    /// Check if key exists
    fn exists(&self, key: &str) -> bool;

    /// Get all keys matching pattern
    fn keys(&self) -> Vec<String>;

    /// Batch get multiple keys, returns map with found key-value pairs
    fn mget(&self, keys: &[String]) -> HashMap<String, Value>;

    /// Delete a key
    fn delete(&self, key: &str);

    /// Get all key-value pairs with keys starting with the given prefix
    fn get_prefix(&self, prefix: &str) -> HashMap<String, Value>;
}

/// Abstract interface for key-value storage backends
pub trait NodeStoreTrait: Send + Sync {
    // Get all nodes
    fn get_all_nodes(&self) -> HashMap<String, Node>;

    /// Batch get multiple ids, returns list with None for missing ids
    fn get_nodes(&self, ids: &[String]) -> HashMap<String, Node>;

    /// Set node for id
    fn set_node(&self, id: &str, node: Node);

    /// Check if node exists
    fn node_exists(&self, id: &str) -> bool;

    /// Get all ids
    fn node_ids(&self) -> Vec<String>;

    /// Delete a node by id
    /// NOTE: Make sure get_parents won't return any deleted node IDs
    fn delete(&self, id: &str);

    /// Get all parent node ids that has an edge pointing to the given id
    fn get_parent_ids(&self, ids: &[String]) -> HashMap<String, Vec<String>>;

    /// Get all children node ids that the given id points to
    fn get_children_ids(&self, ids: &[String]) -> HashMap<String, Vec<String>>;

    /// Get a reference to the underlying KeyValueStore for custom storage needs
    fn get_kv_store(&self) -> &dyn KeyValueStore;
}

pub struct NodeStore {
    data: Arc<dyn KeyValueStore>,
}

impl NodeStore {
    pub fn new(store: Arc<dyn KeyValueStore>) -> Self {
        Self { data: store }
    }

    /// Get a reference to the underlying KeyValueStore
    pub fn store(&self) -> &dyn KeyValueStore {
        &*self.data
    }
}

impl NodeStoreTrait for NodeStore {
    fn get_all_nodes(&self) -> HashMap<String, Node> {
        self.data
            .get_all()
            .into_iter()
            .filter_map(|(id, value)| {
                serde_json::from_value::<Node>(value)
                    .ok()
                    .map(|node| (id, node))
            })
            .collect()
    }

    fn get_nodes(&self, ids: &[String]) -> HashMap<String, Node> {
        self.data
            .mget(&ids.iter().map(|s| s.to_string()).collect::<Vec<String>>())
            .into_iter()
            .filter_map(|(id, value)| {
                serde_json::from_value::<Node>(value)
                    .ok()
                    .map(|node| (id.clone(), node))
            })
            .collect()
    }

    fn set_node(&self, id: &str, node: Node) {
        match node {
            Node::Primitive(p) => {
                // Insert the primitive variable as a node
                self.data
                    .set(id, serde_json::to_value(Node::Primitive(p)).unwrap());
            }
            Node::Composite(c) => {
                let value = serde_json::to_value(&Node::Composite(c.clone())).unwrap();
                self.data.set(id, value);

                // Update outgoing references for this composite node
                // For each coefficient variable, add this id as an incoming reference
                let coef_ids: Vec<String> = c
                    .coefficients
                    .iter()
                    .map(|(coef_id, _)| coef_id.to_string())
                    .collect();
                let mut coefficient_current_references: HashMap<String, Vec<String>> =
                    self.get_parent_ids(&coef_ids);

                for (coef_id, current_references) in coefficient_current_references.iter_mut() {
                    if !current_references.contains(&id.to_string()) {
                        current_references.push(id.to_string());
                        self.data.set(
                            &format!("__outgoing__{}", coef_id),
                            serde_json::to_value(current_references).unwrap(),
                        );
                    }
                }
            }
        }
    }

    fn node_exists(&self, id: &str) -> bool {
        self.data.exists(id)
    }

    fn node_ids(&self) -> Vec<String> {
        self.data
            .keys()
            .iter()
            .filter(|key| !key.starts_with("__outgoing__"))
            .cloned()
            .collect()
    }

    fn delete(&self, id: &str) {
        self.data.delete(id);
    }

    fn get_parent_ids(&self, ids: &[String]) -> HashMap<String, Vec<String>> {
        // Placeholder implementation
        let mut result = self
            .data
            .mget(
                &ids.iter()
                    .map(|id| format!("__outgoing__{}", id))
                    .collect::<Vec<String>>(),
            )
            .into_iter()
            .map(|(id, refs)| {
                (
                    id["__outgoing__".len()..].to_string(),
                    serde_json::from_value(refs).unwrap_or_else(|_| Vec::new()),
                )
            })
            .collect::<HashMap<String, Vec<String>>>();

        // Ensure all requested ids are present in the result with at least an empty vector
        ids.iter().for_each(|id| {
            if !result.contains_key(id) {
                result.insert(id.clone(), Vec::new());
            }
        });

        result
    }

    fn get_children_ids(&self, ids: &[String]) -> HashMap<String, Vec<String>> {
        // Placeholder implementation
        self.data
            .mget(ids)
            .into_iter()
            .map(|(id, val)| {
                (
                    id,
                    match serde_json::from_value::<Node>(val) {
                        Ok(Node::Composite(c)) => c
                            .coefficients
                            .iter()
                            .map(|(child_id, _)| child_id.clone())
                            .collect(),
                        _ => Vec::new(),
                    },
                )
            })
            .collect::<HashMap<String, Vec<String>>>()
    }

    fn get_kv_store(&self) -> &dyn KeyValueStore {
        &*self.data
    }
}

/// In-memory storage implementation (for testing/development)
pub struct InMemoryStore {
    data: RwLock<HashMap<String, Value>>,
}

impl InMemoryStore {
    /// Create a new in-memory store
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl KeyValueStore for InMemoryStore {
    fn get_all(&self) -> HashMap<String, Value> {
        let data = self.data.read().unwrap();
        data.clone()
    }

    fn get(&self, key: &str) -> Option<Value> {
        let data = self.data.read().unwrap();
        data.get(key).cloned()
    }

    fn set(&self, key: &str, value: Value) {
        let mut data = self.data.write().unwrap();
        data.insert(key.to_string(), value);
    }

    fn exists(&self, key: &str) -> bool {
        let data = self.data.read().unwrap();
        data.contains_key(key)
    }

    fn keys(&self) -> Vec<String> {
        let data = self.data.read().unwrap();
        data.keys().cloned().collect()
    }

    fn mget(&self, keys: &[String]) -> HashMap<String, Value> {
        let data = self.data.read().unwrap();
        let mut result = HashMap::with_capacity(keys.len());

        for key in keys {
            if let Some(value) = data.get(key) {
                result.insert(key.clone(), value.clone());
            }
        }

        result
    }

    fn delete(&self, key: &str) {
        let mut data = self.data.write().unwrap();
        data.remove(key);
    }

    fn get_prefix(&self, prefix: &str) -> HashMap<String, Value> {
        let data = self.data.read().unwrap();
        data.iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
}
