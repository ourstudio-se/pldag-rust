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
        // Also remove __outgoing__ references to this node's children
        if let Some(node_value) = self.data.get(id) {
            if let Ok(node) = serde_json::from_value::<Node>(node_value) {
                if let Node::Composite(c) = node {
                    for (coef_id, _) in c.coefficients {
                        let mut current_references = self
                            .get_parent_ids(&[coef_id.clone()])
                            .get(&coef_id)
                            .cloned()
                            .unwrap_or_else(Vec::new);
                        current_references.retain(|ref_id| ref_id != id);
                        self.data.set(
                            &format!("__outgoing__{}", coef_id),
                            serde_json::to_value(current_references).unwrap(),
                        );
                    }
                }
            }
            self.data.delete(id);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pldag::{Constraint, Node};

    #[test]
    fn test_delete_removes_backward_references() {
        // Create a store with composite node that references children
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Set up child nodes
        node_store.set_node("child1", Node::Primitive((0, 1)));
        node_store.set_node("child2", Node::Primitive((0, 1)));

        // Create a composite node that references child1 and child2
        let parent = Node::Composite(Constraint {
            coefficients: vec![
                ("child1".to_string(), 2),
                ("child2".to_string(), 3),
            ],
            bias: (0, 0),
        });
        node_store.set_node("parent", parent);

        // Verify that backward references were created
        let parent_ids = node_store.get_parent_ids(&["child1".to_string(), "child2".to_string()]);
        assert_eq!(parent_ids.get("child1").unwrap(), &vec!["parent".to_string()]);
        assert_eq!(parent_ids.get("child2").unwrap(), &vec!["parent".to_string()]);

        // Delete the parent node
        node_store.delete("parent");

        // Verify that the parent node is deleted
        assert!(!node_store.node_exists("parent"));

        // Verify that backward references are cleaned up
        let parent_ids_after = node_store.get_parent_ids(&["child1".to_string(), "child2".to_string()]);
        assert_eq!(parent_ids_after.get("child1").unwrap(), &Vec::<String>::new());
        assert_eq!(parent_ids_after.get("child2").unwrap(), &Vec::<String>::new());

        // Verify that child nodes still exist
        assert!(node_store.node_exists("child1"));
        assert!(node_store.node_exists("child2"));
    }

    #[test]
    fn test_delete_with_multiple_parents() {
        // Test that deleting one parent doesn't affect other parents' references
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Set up child node
        node_store.set_node("child", Node::Primitive((0, 1)));

        // Create two parent nodes that both reference the same child
        let parent1 = Node::Composite(Constraint {
            coefficients: vec![("child".to_string(), 1)],
            bias: (0, 0),
        });
        let parent2 = Node::Composite(Constraint {
            coefficients: vec![("child".to_string(), 2)],
            bias: (0, 0),
        });
        node_store.set_node("parent1", parent1);
        node_store.set_node("parent2", parent2);

        // Verify both parents are in the backward references
        let parent_ids = node_store.get_parent_ids(&["child".to_string()]);
        let mut parents = parent_ids.get("child").unwrap().clone();
        parents.sort();
        assert_eq!(parents, vec!["parent1".to_string(), "parent2".to_string()]);

        // Delete parent1
        node_store.delete("parent1");

        // Verify only parent2 remains in backward references
        let parent_ids_after = node_store.get_parent_ids(&["child".to_string()]);
        assert_eq!(parent_ids_after.get("child").unwrap(), &vec!["parent2".to_string()]);

        // Verify parent1 is gone but parent2 still exists
        assert!(!node_store.node_exists("parent1"));
        assert!(node_store.node_exists("parent2"));
    }

    #[test]
    fn test_delete_primitive_node() {
        // Test that deleting a primitive node (no children) works
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("prim", Node::Primitive((0, 10)));
        assert!(node_store.node_exists("prim"));

        node_store.delete("prim");
        assert!(!node_store.node_exists("prim"));
    }

    #[test]
    fn test_delete_nonexistent_node() {
        // Test that deleting a nonexistent node doesn't cause issues
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Should not panic
        node_store.delete("nonexistent");
    }

    #[test]
    fn test_set_and_get_nodes() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Set primitive nodes
        node_store.set_node("prim1", Node::Primitive((0, 5)));
        node_store.set_node("prim2", Node::Primitive((-10, 10)));

        // Set composite node
        let composite = Node::Composite(Constraint {
            coefficients: vec![("prim1".to_string(), 2), ("prim2".to_string(), -1)],
            bias: (3, 3),
        });
        node_store.set_node("comp1", composite.clone());

        // Test get_nodes
        let nodes = node_store.get_nodes(&["prim1".to_string(), "prim2".to_string(), "comp1".to_string()]);
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes.get("prim1").unwrap(), &Node::Primitive((0, 5)));
        assert_eq!(nodes.get("prim2").unwrap(), &Node::Primitive((-10, 10)));
        assert_eq!(nodes.get("comp1").unwrap(), &composite);

        // Test get_nodes with missing id
        let nodes = node_store.get_nodes(&["prim1".to_string(), "nonexistent".to_string()]);
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains_key("prim1"));
        assert!(!nodes.contains_key("nonexistent"));
    }

    #[test]
    fn test_get_all_nodes() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Empty store
        let all_nodes = node_store.get_all_nodes();
        assert_eq!(all_nodes.len(), 0);

        // Add some nodes
        node_store.set_node("a", Node::Primitive((0, 1)));
        node_store.set_node("b", Node::Primitive((0, 2)));
        node_store.set_node("c", Node::Composite(Constraint {
            coefficients: vec![("a".to_string(), 1)],
            bias: (0, 0),
        }));

        let all_nodes = node_store.get_all_nodes();
        assert_eq!(all_nodes.len(), 3);
        assert!(all_nodes.contains_key("a"));
        assert!(all_nodes.contains_key("b"));
        assert!(all_nodes.contains_key("c"));

        // Verify __outgoing__ keys are not included
        assert!(!all_nodes.iter().any(|(k, _)| k.starts_with("__outgoing__")));
    }

    #[test]
    fn test_node_exists() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        assert!(!node_store.node_exists("test"));

        node_store.set_node("test", Node::Primitive((0, 1)));
        assert!(node_store.node_exists("test"));

        node_store.delete("test");
        assert!(!node_store.node_exists("test"));
    }

    #[test]
    fn test_node_ids() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Empty store
        let ids = node_store.node_ids();
        assert_eq!(ids.len(), 0);

        // Add nodes
        node_store.set_node("node1", Node::Primitive((0, 1)));
        node_store.set_node("node2", Node::Primitive((0, 2)));
        node_store.set_node("parent", Node::Composite(Constraint {
            coefficients: vec![("node1".to_string(), 1), ("node2".to_string(), 2)],
            bias: (0, 0),
        }));

        let mut ids = node_store.node_ids();
        ids.sort();
        assert_eq!(ids, vec!["node1", "node2", "parent"]);

        // Verify __outgoing__ keys are filtered out
        assert!(!ids.iter().any(|id| id.starts_with("__outgoing__")));
    }

    #[test]
    fn test_get_parent_ids() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Create children
        node_store.set_node("child1", Node::Primitive((0, 1)));
        node_store.set_node("child2", Node::Primitive((0, 1)));
        node_store.set_node("child3", Node::Primitive((0, 1)));

        // No parents yet
        let parent_ids = node_store.get_parent_ids(&["child1".to_string()]);
        assert_eq!(parent_ids.get("child1").unwrap(), &Vec::<String>::new());

        // Create parent that references child1 and child2
        node_store.set_node("parent1", Node::Composite(Constraint {
            coefficients: vec![("child1".to_string(), 1), ("child2".to_string(), 2)],
            bias: (0, 0),
        }));

        // Create another parent that references child1 and child3
        node_store.set_node("parent2", Node::Composite(Constraint {
            coefficients: vec![("child1".to_string(), 3), ("child3".to_string(), 4)],
            bias: (0, 0),
        }));

        // Test get_parent_ids
        let parent_ids = node_store.get_parent_ids(&["child1".to_string(), "child2".to_string(), "child3".to_string()]);

        let mut child1_parents = parent_ids.get("child1").unwrap().clone();
        child1_parents.sort();
        assert_eq!(child1_parents, vec!["parent1", "parent2"]);

        assert_eq!(parent_ids.get("child2").unwrap(), &vec!["parent1"]);
        assert_eq!(parent_ids.get("child3").unwrap(), &vec!["parent2"]);

        // Test with non-existent child
        let parent_ids = node_store.get_parent_ids(&["nonexistent".to_string()]);
        assert_eq!(parent_ids.get("nonexistent").unwrap(), &Vec::<String>::new());
    }

    #[test]
    fn test_get_children_ids() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Create primitive nodes (have no children)
        node_store.set_node("prim", Node::Primitive((0, 1)));

        // Create children
        node_store.set_node("child1", Node::Primitive((0, 1)));
        node_store.set_node("child2", Node::Primitive((0, 1)));
        node_store.set_node("child3", Node::Primitive((0, 1)));

        // Create composite nodes with children
        node_store.set_node("parent1", Node::Composite(Constraint {
            coefficients: vec![("child1".to_string(), 1), ("child2".to_string(), 2)],
            bias: (0, 0),
        }));

        node_store.set_node("parent2", Node::Composite(Constraint {
            coefficients: vec![("child3".to_string(), 1)],
            bias: (0, 0),
        }));

        // Test get_children_ids
        let children_map = node_store.get_children_ids(&["prim".to_string(), "parent1".to_string(), "parent2".to_string()]);

        assert_eq!(children_map.get("prim").unwrap(), &Vec::<String>::new());
        assert_eq!(children_map.get("parent1").unwrap(), &vec!["child1", "child2"]);
        assert_eq!(children_map.get("parent2").unwrap(), &vec!["child3"]);

        // Test with non-existent node
        let children_map = node_store.get_children_ids(&["nonexistent".to_string()]);
        assert_eq!(children_map.len(), 0);
    }

    #[test]
    fn test_set_node_updates_backward_references() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        // Create child
        node_store.set_node("child", Node::Primitive((0, 1)));

        // Create parent with one reference to child
        node_store.set_node("parent", Node::Composite(Constraint {
            coefficients: vec![("child".to_string(), 1)],
            bias: (0, 0),
        }));

        let parent_ids = node_store.get_parent_ids(&["child".to_string()]);
        assert_eq!(parent_ids.get("child").unwrap(), &vec!["parent"]);

        // Update parent to add another child
        node_store.set_node("child2", Node::Primitive((0, 1)));
        node_store.set_node("parent", Node::Composite(Constraint {
            coefficients: vec![("child".to_string(), 1), ("child2".to_string(), 2)],
            bias: (0, 0),
        }));

        // Verify both children have parent in their references
        let parent_ids = node_store.get_parent_ids(&["child".to_string(), "child2".to_string()]);
        assert_eq!(parent_ids.get("child").unwrap(), &vec!["parent"]);
        assert_eq!(parent_ids.get("child2").unwrap(), &vec!["parent"]);
    }

    #[test]
    fn test_set_node_does_not_duplicate_references() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("child", Node::Primitive((0, 1)));

        // Set same parent node multiple times
        let composite = Node::Composite(Constraint {
            coefficients: vec![("child".to_string(), 1)],
            bias: (0, 0),
        });

        node_store.set_node("parent", composite.clone());
        node_store.set_node("parent", composite.clone());
        node_store.set_node("parent", composite.clone());

        // Should only have one reference
        let parent_ids = node_store.get_parent_ids(&["child".to_string()]);
        assert_eq!(parent_ids.get("child").unwrap(), &vec!["parent"]);
    }

    #[test]
    fn test_get_kv_store() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store.clone());

        // Test that we can access the underlying store
        let kv_store = node_store.get_kv_store();

        // Set a value directly through the kv store
        kv_store.set("test_key", serde_json::json!("test_value"));

        // Verify we can read it back
        let value = kv_store.get("test_key");
        assert_eq!(value, Some(serde_json::json!("test_value")));
    }

    #[test]
    fn test_store_method() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store.clone());

        // Test the store() method returns the same interface as get_kv_store()
        let kv_store = node_store.store();

        kv_store.set("key", serde_json::json!(42));
        assert_eq!(kv_store.get("key"), Some(serde_json::json!(42)));
    }
}
