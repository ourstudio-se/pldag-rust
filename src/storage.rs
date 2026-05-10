use crate::Bound;
use crate::error::{StorageError, StorageResult};
use crate::pldag::Node;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Abstract interface for key-value storage backends.
///
/// All methods are async to allow database-backed implementations. The provided
/// `InMemoryStore` resolves immediately without yielding to the runtime.
#[async_trait]
pub trait KeyValueStore: Send + Sync {
    /// Get all key-value pairs
    async fn get_all(&self) -> StorageResult<HashMap<String, Value>>;

    /// Get value for key
    async fn get(&self, key: &str) -> StorageResult<Option<Value>>;

    /// Set value for key
    async fn set(&self, key: &str, value: Value) -> StorageResult<()>;

    /// Set multiple key-value pairs
    async fn mset(&self, kv_pairs: &[(String, Value)]) -> StorageResult<()>;

    /// Check if key exists
    async fn exists(&self, key: &str) -> StorageResult<bool>;

    /// Get all keys
    async fn keys(&self) -> StorageResult<Vec<String>>;

    /// Batch get multiple keys, returns map with found key-value pairs
    async fn mget(&self, keys: &[String]) -> StorageResult<HashMap<String, Value>>;

    /// Delete a key
    async fn delete(&self, key: &str) -> StorageResult<()>;

    /// Get all key-value pairs with keys starting with the given prefix
    async fn get_prefix(&self, prefix: &str) -> StorageResult<HashMap<String, Value>>;
}

/// Abstract interface for node storage, layered on top of a `KeyValueStore`.
#[async_trait]
pub trait NodeStoreTrait: Send + Sync {
    /// Get all nodes
    async fn get_all_nodes(&self) -> StorageResult<HashMap<String, Node>>;

    /// Batch get multiple ids, returns map with only the ids that exist
    async fn get_nodes(&self, ids: &[String]) -> StorageResult<HashMap<String, Node>>;

    /// Set node for id
    async fn set_node(&self, id: &str, node: Node) -> StorageResult<()>;

    /// Bulk-set primitive nodes
    async fn set_primitives(&self, primitives: &[(&str, &Bound)]) -> StorageResult<()>;

    /// Check if node exists
    async fn node_exists(&self, id: &str) -> StorageResult<bool>;

    /// Get all node ids
    async fn node_ids(&self) -> StorageResult<Vec<String>>;

    /// Delete a node by id.
    /// NOTE: callers must ensure `get_parent_ids` won't return any deleted node IDs
    /// (this method clears outgoing references for composite nodes automatically).
    async fn delete(&self, id: &str) -> StorageResult<()>;

    /// Get all parent node ids that have an edge pointing to the given id
    async fn get_parent_ids(&self, ids: &[String]) -> StorageResult<HashMap<String, Vec<String>>>;

    /// Get all children node ids that the given id points to
    async fn get_children_ids(&self, ids: &[String]) -> StorageResult<HashMap<String, Vec<String>>>;

    /// Get a reference to the underlying KeyValueStore for custom storage needs
    fn get_kv_store(&self) -> &dyn KeyValueStore;
}

/// Default [`NodeStoreTrait`] implementation, layered on top of any [`KeyValueStore`].
///
/// `NodeStore` handles JSON (de)serialization of [`Node`] values and tracks
/// the reverse-edge "outgoing" rows used to support efficient deletions and
/// parent lookups. Pair it with [`InMemoryStore`] for tests, or with a
/// custom backend (e.g. Redis, Postgres, S3) for persistence.
pub struct NodeStore {
    data: Arc<dyn KeyValueStore>,
}

impl NodeStore {
    /// Wraps an existing [`KeyValueStore`] in a [`NodeStore`].
    pub fn new(store: Arc<dyn KeyValueStore>) -> Self {
        Self { data: store }
    }

    /// Get a reference to the underlying KeyValueStore
    pub fn store(&self) -> &dyn KeyValueStore {
        &*self.data
    }

    /// Removes `parent` from each `__outgoing__<child>` row.
    ///
    /// Used both when deleting a composite and when overwriting one whose
    /// coefficient set has shrunk — both leave behind backward-reference
    /// rows that must be rewritten so `get_parent_ids` doesn't return
    /// phantom edges.
    async fn remove_outgoing(&self, child_ids: &[String], parent: &str) -> StorageResult<()> {
        if child_ids.is_empty() {
            return Ok(());
        }
        let mut current = self.get_parent_ids(child_ids).await?;
        for child in child_ids {
            let entry = current.entry(child.clone()).or_default();
            entry.retain(|ref_id| ref_id != parent);
            let key = format!("__outgoing__{}", child);
            let serialized = serde_json::to_value(&entry).map_err(|e| {
                StorageError::Serialization {
                    key: key.clone(),
                    message: e.to_string(),
                }
            })?;
            self.data.set(&key, serialized).await?;
        }
        Ok(())
    }
}

fn to_value_for(key: &str, node: &Node) -> StorageResult<Value> {
    serde_json::to_value(node).map_err(|e| StorageError::Serialization {
        key: key.to_string(),
        message: e.to_string(),
    })
}

#[async_trait]
impl NodeStoreTrait for NodeStore {
    async fn get_all_nodes(&self) -> StorageResult<HashMap<String, Node>> {
        let raw = self.data.get_all().await?;
        let mut out = HashMap::with_capacity(raw.len());
        for (id, value) in raw {
            // Skip backward-reference rows; they are not Node-shaped.
            if id.starts_with("__outgoing__") {
                continue;
            }
            if let Ok(node) = serde_json::from_value::<Node>(value) {
                out.insert(id, node);
            }
        }
        Ok(out)
    }

    async fn get_nodes(&self, ids: &[String]) -> StorageResult<HashMap<String, Node>> {
        let raw = self
            .data
            .mget(ids)
            .await?;
        let mut out = HashMap::with_capacity(raw.len());
        for (id, value) in raw {
            let node = serde_json::from_value::<Node>(value).map_err(|e| {
                StorageError::Deserialization {
                    key: id.clone(),
                    message: e.to_string(),
                }
            })?;
            out.insert(id, node);
        }
        Ok(out)
    }

    async fn set_node(&self, id: &str, node: Node) -> StorageResult<()> {
        // Children referenced by the *new* node, used both to rewrite
        // backward-reference rows and to detect which previous children
        // (if any) have become stale.
        let new_children: HashSet<String> = match &node {
            Node::Primitive(_) => HashSet::new(),
            Node::Composite(c) => c
                .coefficients
                .iter()
                .map(|(coef_id, _)| coef_id.clone())
                .collect(),
        };

        // If we're overwriting an existing composite, drop `id` from any of
        // its previous children that are not also referenced by the new
        // node — otherwise their `__outgoing__<child>` rows keep pointing
        // at us and `get_parent_ids` returns phantom edges.
        if let Some(prev_value) = self.data.get(id).await? {
            if let Ok(Node::Composite(prev)) = serde_json::from_value::<Node>(prev_value) {
                let stale: Vec<String> = prev
                    .coefficients
                    .iter()
                    .map(|(coef_id, _)| coef_id.clone())
                    .filter(|coef_id| !new_children.contains(coef_id))
                    .collect();
                self.remove_outgoing(&stale, id).await?;
            }
        }

        match node {
            Node::Primitive(p) => {
                let value = to_value_for(id, &Node::Primitive(p))?;
                self.data.set(id, value).await?;
            }
            Node::Composite(c) => {
                let value = to_value_for(id, &Node::Composite(c.clone()))?;
                self.data.set(id, value).await?;

                // Update outgoing references for this composite node.
                // For each coefficient variable, add this id as an incoming reference.
                let coef_ids: Vec<String> = c
                    .coefficients
                    .iter()
                    .map(|(coef_id, _)| coef_id.to_string())
                    .collect();
                let mut coefficient_current_references = self.get_parent_ids(&coef_ids).await?;

                for (coef_id, current_references) in coefficient_current_references.iter_mut() {
                    if !current_references.contains(&id.to_string()) {
                        current_references.push(id.to_string());
                        let key = format!("__outgoing__{}", coef_id);
                        let serialized = serde_json::to_value(&current_references).map_err(|e| {
                            StorageError::Serialization {
                                key: key.clone(),
                                message: e.to_string(),
                            }
                        })?;
                        self.data.set(&key, serialized).await?;
                    }
                }
            }
        }
        Ok(())
    }

    async fn set_primitives(&self, primitives: &[(&str, &Bound)]) -> StorageResult<()> {
        let mut kv_pairs = Vec::with_capacity(primitives.len());
        for (id, &bound) in primitives.iter() {
            let value = to_value_for(id, &Node::Primitive(bound))?;
            kv_pairs.push((id.to_string(), value));
        }
        self.data.mset(&kv_pairs).await?;
        Ok(())
    }

    async fn node_exists(&self, id: &str) -> StorageResult<bool> {
        self.data.exists(id).await
    }

    async fn node_ids(&self) -> StorageResult<Vec<String>> {
        let all = self.data.keys().await?;
        Ok(all
            .into_iter()
            .filter(|key| !key.starts_with("__outgoing__"))
            .collect())
    }

    async fn delete(&self, id: &str) -> StorageResult<()> {
        // Also remove __outgoing__ references to this node's children.
        if let Some(node_value) = self.data.get(id).await? {
            if let Ok(Node::Composite(c)) = serde_json::from_value::<Node>(node_value) {
                let child_ids: Vec<String> = c
                    .coefficients
                    .into_iter()
                    .map(|(coef_id, _)| coef_id)
                    .collect();
                self.remove_outgoing(&child_ids, id).await?;
            }
            self.data.delete(id).await?;
        }
        Ok(())
    }

    async fn get_parent_ids(
        &self,
        ids: &[String],
    ) -> StorageResult<HashMap<String, Vec<String>>> {
        let lookup_keys: Vec<String> = ids
            .iter()
            .map(|id| format!("__outgoing__{}", id))
            .collect();
        let raw = self.data.mget(&lookup_keys).await?;

        let mut result: HashMap<String, Vec<String>> = HashMap::with_capacity(ids.len());
        for (key, refs) in raw {
            let original = key["__outgoing__".len()..].to_string();
            let parsed: Vec<String> = serde_json::from_value(refs).map_err(|e| {
                StorageError::Deserialization {
                    key: key.clone(),
                    message: e.to_string(),
                }
            })?;
            result.insert(original, parsed);
        }

        // Ensure all requested ids are present in the result with at least an empty vector.
        for id in ids {
            result.entry(id.clone()).or_default();
        }

        Ok(result)
    }

    async fn get_children_ids(
        &self,
        ids: &[String],
    ) -> StorageResult<HashMap<String, Vec<String>>> {
        let raw = self.data.mget(ids).await?;
        let mut out = HashMap::with_capacity(raw.len());
        for (id, val) in raw {
            let children = match serde_json::from_value::<Node>(val) {
                Ok(Node::Composite(c)) => c
                    .coefficients
                    .iter()
                    .map(|(child_id, _)| child_id.clone())
                    .collect(),
                _ => Vec::new(),
            };
            out.insert(id, children);
        }
        Ok(out)
    }

    fn get_kv_store(&self) -> &dyn KeyValueStore {
        &*self.data
    }
}

/// In-memory storage implementation (for testing/development).
///
/// Methods are declared `async` to fit the trait, but the bodies are synchronous
/// (`std::sync::RwLock` under the hood) so they resolve without yielding.
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

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl KeyValueStore for InMemoryStore {
    async fn get_all(&self) -> StorageResult<HashMap<String, Value>> {
        let data = self.data.read().unwrap();
        Ok(data.clone())
    }

    async fn get(&self, key: &str) -> StorageResult<Option<Value>> {
        let data = self.data.read().unwrap();
        Ok(data.get(key).cloned())
    }

    async fn set(&self, key: &str, value: Value) -> StorageResult<()> {
        let mut data = self.data.write().unwrap();
        data.insert(key.to_string(), value);
        Ok(())
    }

    async fn mset(&self, kv_pairs: &[(String, Value)]) -> StorageResult<()> {
        let mut data = self.data.write().unwrap();
        for (key, value) in kv_pairs {
            data.insert(key.to_string(), value.clone());
        }
        Ok(())
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        let data = self.data.read().unwrap();
        Ok(data.contains_key(key))
    }

    async fn keys(&self) -> StorageResult<Vec<String>> {
        let data = self.data.read().unwrap();
        Ok(data.keys().cloned().collect())
    }

    async fn mget(&self, keys: &[String]) -> StorageResult<HashMap<String, Value>> {
        let data = self.data.read().unwrap();
        let mut result = HashMap::with_capacity(keys.len());
        for key in keys {
            if let Some(value) = data.get(key) {
                result.insert(key.clone(), value.clone());
            }
        }
        Ok(result)
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        let mut data = self.data.write().unwrap();
        data.remove(key);
        Ok(())
    }

    async fn get_prefix(&self, prefix: &str) -> StorageResult<HashMap<String, Value>> {
        let data = self.data.read().unwrap();
        Ok(data
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pldag::{Constraint, Node};

    #[tokio::test]
    async fn test_delete_removes_backward_references() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("child1", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("child2", Node::Primitive((0, 1))).await.unwrap();

        let parent = Node::Composite(Constraint {
            coefficients: vec![("child1".to_string(), 2), ("child2".to_string(), 3)],
            bias: (0, 0),
        });
        node_store.set_node("parent", parent).await.unwrap();

        let parent_ids = node_store
            .get_parent_ids(&["child1".to_string(), "child2".to_string()])
            .await
            .unwrap();
        assert_eq!(parent_ids.get("child1").unwrap(), &vec!["parent".to_string()]);
        assert_eq!(parent_ids.get("child2").unwrap(), &vec!["parent".to_string()]);

        node_store.delete("parent").await.unwrap();

        assert!(!node_store.node_exists("parent").await.unwrap());

        let parent_ids_after = node_store
            .get_parent_ids(&["child1".to_string(), "child2".to_string()])
            .await
            .unwrap();
        assert_eq!(parent_ids_after.get("child1").unwrap(), &Vec::<String>::new());
        assert_eq!(parent_ids_after.get("child2").unwrap(), &Vec::<String>::new());

        assert!(node_store.node_exists("child1").await.unwrap());
        assert!(node_store.node_exists("child2").await.unwrap());
    }

    #[tokio::test]
    async fn test_delete_with_multiple_parents() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("child", Node::Primitive((0, 1))).await.unwrap();

        let parent1 = Node::Composite(Constraint {
            coefficients: vec![("child".to_string(), 1)],
            bias: (0, 0),
        });
        let parent2 = Node::Composite(Constraint {
            coefficients: vec![("child".to_string(), 2)],
            bias: (0, 0),
        });
        node_store.set_node("parent1", parent1).await.unwrap();
        node_store.set_node("parent2", parent2).await.unwrap();

        let parent_ids = node_store.get_parent_ids(&["child".to_string()]).await.unwrap();
        let mut parents = parent_ids.get("child").unwrap().clone();
        parents.sort();
        assert_eq!(parents, vec!["parent1".to_string(), "parent2".to_string()]);

        node_store.delete("parent1").await.unwrap();

        let parent_ids_after = node_store.get_parent_ids(&["child".to_string()]).await.unwrap();
        assert_eq!(
            parent_ids_after.get("child").unwrap(),
            &vec!["parent2".to_string()]
        );

        assert!(!node_store.node_exists("parent1").await.unwrap());
        assert!(node_store.node_exists("parent2").await.unwrap());
    }

    #[tokio::test]
    async fn test_delete_primitive_node() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("prim", Node::Primitive((0, 10))).await.unwrap();
        assert!(node_store.node_exists("prim").await.unwrap());

        node_store.delete("prim").await.unwrap();
        assert!(!node_store.node_exists("prim").await.unwrap());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_node() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.delete("nonexistent").await.unwrap();
    }

    #[tokio::test]
    async fn test_set_and_get_nodes() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("prim1", Node::Primitive((0, 5))).await.unwrap();
        node_store.set_node("prim2", Node::Primitive((-10, 10))).await.unwrap();

        let composite = Node::Composite(Constraint {
            coefficients: vec![("prim1".to_string(), 2), ("prim2".to_string(), -1)],
            bias: (3, 3),
        });
        node_store.set_node("comp1", composite.clone()).await.unwrap();

        let nodes = node_store
            .get_nodes(&[
                "prim1".to_string(),
                "prim2".to_string(),
                "comp1".to_string(),
            ])
            .await
            .unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes.get("prim1").unwrap(), &Node::Primitive((0, 5)));
        assert_eq!(nodes.get("prim2").unwrap(), &Node::Primitive((-10, 10)));
        assert_eq!(nodes.get("comp1").unwrap(), &composite);

        let nodes = node_store
            .get_nodes(&["prim1".to_string(), "nonexistent".to_string()])
            .await
            .unwrap();
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains_key("prim1"));
        assert!(!nodes.contains_key("nonexistent"));
    }

    #[tokio::test]
    async fn test_get_all_nodes() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        let all_nodes = node_store.get_all_nodes().await.unwrap();
        assert_eq!(all_nodes.len(), 0);

        node_store.set_node("a", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("b", Node::Primitive((0, 2))).await.unwrap();
        node_store
            .set_node(
                "c",
                Node::Composite(Constraint {
                    coefficients: vec![("a".to_string(), 1)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        let all_nodes = node_store.get_all_nodes().await.unwrap();
        assert_eq!(all_nodes.len(), 3);
        assert!(all_nodes.contains_key("a"));
        assert!(all_nodes.contains_key("b"));
        assert!(all_nodes.contains_key("c"));

        assert!(!all_nodes.iter().any(|(k, _)| k.starts_with("__outgoing__")));
    }

    #[tokio::test]
    async fn test_node_exists() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        assert!(!node_store.node_exists("test").await.unwrap());

        node_store.set_node("test", Node::Primitive((0, 1))).await.unwrap();
        assert!(node_store.node_exists("test").await.unwrap());

        node_store.delete("test").await.unwrap();
        assert!(!node_store.node_exists("test").await.unwrap());
    }

    #[tokio::test]
    async fn test_node_ids() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        let ids = node_store.node_ids().await.unwrap();
        assert_eq!(ids.len(), 0);

        node_store.set_node("node1", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("node2", Node::Primitive((0, 2))).await.unwrap();
        node_store
            .set_node(
                "parent",
                Node::Composite(Constraint {
                    coefficients: vec![("node1".to_string(), 1), ("node2".to_string(), 2)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        let mut ids = node_store.node_ids().await.unwrap();
        ids.sort();
        assert_eq!(ids, vec!["node1", "node2", "parent"]);

        assert!(!ids.iter().any(|id| id.starts_with("__outgoing__")));
    }

    #[tokio::test]
    async fn test_get_parent_ids() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("child1", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("child2", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("child3", Node::Primitive((0, 1))).await.unwrap();

        let parent_ids = node_store.get_parent_ids(&["child1".to_string()]).await.unwrap();
        assert_eq!(parent_ids.get("child1").unwrap(), &Vec::<String>::new());

        node_store
            .set_node(
                "parent1",
                Node::Composite(Constraint {
                    coefficients: vec![("child1".to_string(), 1), ("child2".to_string(), 2)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        node_store
            .set_node(
                "parent2",
                Node::Composite(Constraint {
                    coefficients: vec![("child1".to_string(), 3), ("child3".to_string(), 4)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        let parent_ids = node_store
            .get_parent_ids(&[
                "child1".to_string(),
                "child2".to_string(),
                "child3".to_string(),
            ])
            .await
            .unwrap();

        let mut child1_parents = parent_ids.get("child1").unwrap().clone();
        child1_parents.sort();
        assert_eq!(child1_parents, vec!["parent1", "parent2"]);

        assert_eq!(parent_ids.get("child2").unwrap(), &vec!["parent1"]);
        assert_eq!(parent_ids.get("child3").unwrap(), &vec!["parent2"]);

        let parent_ids = node_store.get_parent_ids(&["nonexistent".to_string()]).await.unwrap();
        assert_eq!(parent_ids.get("nonexistent").unwrap(), &Vec::<String>::new());
    }

    #[tokio::test]
    async fn test_get_children_ids() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("prim", Node::Primitive((0, 1))).await.unwrap();

        node_store.set_node("child1", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("child2", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("child3", Node::Primitive((0, 1))).await.unwrap();

        node_store
            .set_node(
                "parent1",
                Node::Composite(Constraint {
                    coefficients: vec![("child1".to_string(), 1), ("child2".to_string(), 2)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        node_store
            .set_node(
                "parent2",
                Node::Composite(Constraint {
                    coefficients: vec![("child3".to_string(), 1)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        let children_map = node_store
            .get_children_ids(&[
                "prim".to_string(),
                "parent1".to_string(),
                "parent2".to_string(),
            ])
            .await
            .unwrap();

        assert_eq!(children_map.get("prim").unwrap(), &Vec::<String>::new());
        assert_eq!(
            children_map.get("parent1").unwrap(),
            &vec!["child1", "child2"]
        );
        assert_eq!(children_map.get("parent2").unwrap(), &vec!["child3"]);

        let children_map = node_store.get_children_ids(&["nonexistent".to_string()]).await.unwrap();
        assert_eq!(children_map.len(), 0);
    }

    #[tokio::test]
    async fn test_set_node_updates_backward_references() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("child", Node::Primitive((0, 1))).await.unwrap();

        node_store
            .set_node(
                "parent",
                Node::Composite(Constraint {
                    coefficients: vec![("child".to_string(), 1)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        let parent_ids = node_store.get_parent_ids(&["child".to_string()]).await.unwrap();
        assert_eq!(parent_ids.get("child").unwrap(), &vec!["parent"]);

        node_store.set_node("child2", Node::Primitive((0, 1))).await.unwrap();
        node_store
            .set_node(
                "parent",
                Node::Composite(Constraint {
                    coefficients: vec![("child".to_string(), 1), ("child2".to_string(), 2)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        let parent_ids = node_store
            .get_parent_ids(&["child".to_string(), "child2".to_string()])
            .await
            .unwrap();
        assert_eq!(parent_ids.get("child").unwrap(), &vec!["parent"]);
        assert_eq!(parent_ids.get("child2").unwrap(), &vec!["parent"]);
    }

    #[tokio::test]
    async fn test_set_node_does_not_duplicate_references() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("child", Node::Primitive((0, 1))).await.unwrap();

        let composite = Node::Composite(Constraint {
            coefficients: vec![("child".to_string(), 1)],
            bias: (0, 0),
        });

        node_store.set_node("parent", composite.clone()).await.unwrap();
        node_store.set_node("parent", composite.clone()).await.unwrap();
        node_store.set_node("parent", composite.clone()).await.unwrap();

        let parent_ids = node_store.get_parent_ids(&["child".to_string()]).await.unwrap();
        assert_eq!(parent_ids.get("child").unwrap(), &vec!["parent"]);
    }

    #[tokio::test]
    async fn test_set_node_clears_dropped_children_when_composite_shrinks() {
        // Overwriting a composite with one whose coefficient set has shrunk
        // must remove the parent from the dropped child's __outgoing__ row.
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("a", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("b", Node::Primitive((0, 1))).await.unwrap();

        node_store
            .set_node(
                "parent",
                Node::Composite(Constraint {
                    coefficients: vec![("a".to_string(), 1), ("b".to_string(), 1)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        // Sanity: both children point at parent.
        let parents = node_store
            .get_parent_ids(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(parents.get("a").unwrap(), &vec!["parent"]);
        assert_eq!(parents.get("b").unwrap(), &vec!["parent"]);

        // Rewrite parent to drop "b".
        node_store
            .set_node(
                "parent",
                Node::Composite(Constraint {
                    coefficients: vec![("a".to_string(), 1)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        let parents = node_store
            .get_parent_ids(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(parents.get("a").unwrap(), &vec!["parent"]);
        assert!(
            parents.get("b").map(|v| v.is_empty()).unwrap_or(true),
            "b's __outgoing__ row should not still list 'parent'; got {:?}",
            parents.get("b"),
        );
    }

    #[tokio::test]
    async fn test_set_node_clears_all_children_when_composite_replaced_by_primitive() {
        // Replacing a composite with a primitive must clear *all* of the
        // previous composite's backward-reference rows.
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        node_store.set_node("a", Node::Primitive((0, 1))).await.unwrap();
        node_store.set_node("b", Node::Primitive((0, 1))).await.unwrap();

        node_store
            .set_node(
                "parent",
                Node::Composite(Constraint {
                    coefficients: vec![("a".to_string(), 1), ("b".to_string(), 1)],
                    bias: (0, 0),
                }),
            )
            .await
            .unwrap();

        // Replace the composite with a primitive at the same id.
        node_store
            .set_node("parent", Node::Primitive((0, 1)))
            .await
            .unwrap();

        let parents = node_store
            .get_parent_ids(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert!(
            parents.get("a").map(|v| v.is_empty()).unwrap_or(true),
            "a's __outgoing__ row should be empty after parent became primitive; got {:?}",
            parents.get("a"),
        );
        assert!(
            parents.get("b").map(|v| v.is_empty()).unwrap_or(true),
            "b's __outgoing__ row should be empty after parent became primitive; got {:?}",
            parents.get("b"),
        );
    }

    #[tokio::test]
    async fn test_get_kv_store() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store.clone());

        let kv_store = node_store.get_kv_store();

        kv_store.set("test_key", serde_json::json!("test_value")).await.unwrap();

        let value = kv_store.get("test_key").await.unwrap();
        assert_eq!(value, Some(serde_json::json!("test_value")));
    }

    #[tokio::test]
    async fn test_store_method() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store.clone());

        let kv_store = node_store.store();

        kv_store.set("key", serde_json::json!(42)).await.unwrap();
        assert_eq!(kv_store.get("key").await.unwrap(), Some(serde_json::json!(42)));
    }

    #[tokio::test]
    async fn test_set_primitives() {
        let store = Arc::new(InMemoryStore::new());
        let node_store = NodeStore::new(store);

        let primitives: Vec<(&str, Bound)> = vec![
            ("prim1", (0, 1)),
            ("prim2", (-5, 5)),
            ("prim3", (10, 20)),
        ];

        let primitives_ref: Vec<(&str, &Bound)> = primitives.iter().map(|(s, b)| (*s, b)).collect();
        node_store.set_primitives(&primitives_ref).await.unwrap();

        let nodes = node_store
            .get_nodes(&["prim1".to_string(), "prim2".to_string(), "prim3".to_string()])
            .await
            .unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes.get("prim1").unwrap(), &Node::Primitive((0, 1)));
        assert_eq!(nodes.get("prim2").unwrap(), &Node::Primitive((-5, 5)));
        assert_eq!(nodes.get("prim3").unwrap(), &Node::Primitive((10, 20)));

        let all_nodes = node_store.get_all_nodes().await.unwrap();
        assert_eq!(all_nodes.len(), 3);
        assert_eq!(all_nodes.get("prim1").unwrap(), &Node::Primitive((0, 1)));
    }
}
