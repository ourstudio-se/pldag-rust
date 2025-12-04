use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Abstract interface for key-value storage backends
pub trait KeyValueStore: Send + Sync {
    /// Get value for key, return None if not found
    fn get(&self, key: &str) -> Option<Value>;

    /// Set value for key
    fn set(&self, key: &str, value: Value);

    /// Check if key exists
    fn exists(&self, key: &str) -> bool;

    /// Get all keys matching pattern
    fn keys(&self, pattern: &str) -> Vec<String>;

    /// Batch get multiple keys, returns list with None for missing keys
    fn mget(&self, keys: &[String]) -> HashMap<String, Value>;

    /// Delete a key
    fn delete(&self, key: &str);
}

/// In-memory storage implementation (for testing/development)
pub struct InMemoryStore {
    data: Arc<Mutex<HashMap<String, Value>>>,
}

impl InMemoryStore {
    /// Create a new in-memory store
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn make_key(&self, key: &str) -> String {
        key.to_string()
    }
}

impl KeyValueStore for InMemoryStore {
    fn get(&self, key: &str) -> Option<Value> {
        let data = self.data.lock().unwrap();
        data.get(&self.make_key(key)).cloned()
    }

    fn set(&self, key: &str, value: Value) {
        let mut data = self.data.lock().unwrap();
        data.insert(self.make_key(key), value);
    }

    fn exists(&self, key: &str) -> bool {
        let data = self.data.lock().unwrap();
        data.contains_key(&self.make_key(key))
    }

    fn keys(&self, pattern: &str) -> Vec<String> {
        let data = self.data.lock().unwrap();
        let search_pattern = pattern.to_string();

        let mut result = Vec::new();
        for key in data.keys() {
            if glob_match::glob_match(&search_pattern, key) {
                result.push(key.to_string());
            }
        }

        result
    }

    fn mget(&self, keys: &[String]) -> HashMap<String, Value> {
        let data = self.data.lock().unwrap();
        let mut result = HashMap::with_capacity(keys.len());

        for key in keys {
            if let Some(value) = data.get(&self.make_key(key)) {
                result.insert(key.clone(), value.clone());
            }
        }

        result
    }

    fn delete(&self, key: &str) {
        let mut data = self.data.lock().unwrap();
        data.remove(&self.make_key(key));
    }
}
