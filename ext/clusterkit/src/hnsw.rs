use magnus::{
    class, exception, function, method, prelude::*, 
    Error, Float, Integer, RArray, RHash, RString, Symbol, Value, value, TryConvert, r_hash::ForEach
};
use hnsw_rs::prelude::*;
use hnsw_rs::hnswio::HnswIo;
// use ndarray::Array1; // Not used currently
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};
use std::fs::File;

// Store metadata alongside vectors
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ItemMetadata {
    label: String,
    metadata: Option<HashMap<String, String>>,
}

// Main HNSW wrapper struct
#[magnus::wrap(class = "ClusterKit::HNSW", free_immediately, size)]
pub struct HnswIndex {
    hnsw: Arc<Mutex<Hnsw<'static, f32, DistL2>>>,
    dim: usize,
    space: DistanceType,
    metadata_store: Arc<Mutex<HashMap<usize, ItemMetadata>>>,
    current_id: Arc<Mutex<usize>>,
    label_to_id: Arc<Mutex<HashMap<String, usize>>>,
    ef_search: Arc<Mutex<usize>>,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]  // These variants will be implemented in the future
enum DistanceType {
    Euclidean,
    Cosine,
    InnerProduct,
}

impl HnswIndex {
    // Initialize a new HNSW index
    pub fn new(kwargs: RHash) -> Result<Self, Error> {
        // Parse arguments
        let dim_opt: Option<Value> = kwargs.delete(Symbol::new("dim"))?;
        let dim_value = dim_opt.ok_or_else(|| Error::new(exception::arg_error(), "dim is required"))?;
        let dim: usize = TryConvert::try_convert(dim_value)
            .map_err(|_| Error::new(exception::arg_error(), "dim must be an integer"))?;
        
        // Validate dimension
        if dim == 0 {
            return Err(Error::new(exception::arg_error(), "dim must be a positive integer (got 0)"));
        }
        
        let space: String = if let Some(v) = kwargs.delete(Symbol::new("space"))? {
            // Convert Ruby symbol to string properly
            if let Ok(sym) = Symbol::try_convert(v) {
                sym.name()?.to_string()
            } else if let Ok(s) = String::try_convert(v) {
                s
            } else {
                return Err(Error::new(
                    exception::type_error(),
                    "space must be a string or symbol"
                ));
            }
        } else {
            "euclidean".to_string()
        };
        
        let max_elements: usize = if let Some(v) = kwargs.delete(Symbol::new("max_elements"))? {
            TryConvert::try_convert(v).unwrap_or(10_000)
        } else {
            10_000
        };
        
        let m: usize = if let Some(v) = kwargs.delete(Symbol::new("M"))? {
            TryConvert::try_convert(v).unwrap_or(16)
        } else {
            16
        };
        
        let ef_construction: usize = if let Some(v) = kwargs.delete(Symbol::new("ef_construction"))? {
            TryConvert::try_convert(v).unwrap_or(200)
        } else {
            200
        };
        
        let random_seed: Option<u64> = if let Some(v) = kwargs.delete(Symbol::new("random_seed"))? {
            TryConvert::try_convert(v).ok()
        } else {
            None
        };
        
        // Validate and convert space parameter
        // For now, only support Euclidean distance
        let distance_type = match space.as_str() {
            "euclidean" => DistanceType::Euclidean,
            "cosine" => {
                return Err(Error::new(
                    exception::runtime_error(),
                    "Cosine distance is not yet implemented, please use :euclidean"
                ));
            },
            "inner_product" => {
                return Err(Error::new(
                    exception::runtime_error(),
                    "Inner product distance is not yet implemented, please use :euclidean"
                ));
            },
            _ => return Err(Error::new(
                exception::arg_error(),
                format!("space must be :euclidean, :cosine, or :inner_product (got: {})", space)
            )),
        };
        
        // Create HNSW instance with Euclidean distance
        let hnsw = if let Some(seed) = random_seed {
            Hnsw::<f32, DistL2>::new_with_seed(m, max_elements, 16, ef_construction, DistL2, seed)
        } else {
            Hnsw::<f32, DistL2>::new(m, max_elements, 16, ef_construction, DistL2)
        };
        
        Ok(Self {
            hnsw: Arc::new(Mutex::new(hnsw)),
            dim,
            space: distance_type,
            metadata_store: Arc::new(Mutex::new(HashMap::new())),
            current_id: Arc::new(Mutex::new(0)),
            label_to_id: Arc::new(Mutex::new(HashMap::new())),
            ef_search: Arc::new(Mutex::new(ef_construction)),
        })
    }
    
    // Add a single item to the index
    pub fn add_item(&self, vector: RArray, kwargs: RHash) -> Result<Value, Error> {
        // Parse vector
        let vec_data = parse_vector(vector, self.dim)?;
        
        // Get or generate label
        let label: String = if let Some(v) = kwargs.delete(Symbol::new("label"))? {
            TryConvert::try_convert(v).unwrap_or_else(|_| {
                let mut id = self.current_id.lock().unwrap();
                let label = id.to_string();
                *id += 1;
                label
            })
        } else {
            let mut id = self.current_id.lock().unwrap();
            let label = id.to_string();
            *id += 1;
            label
        };
        
        // Get metadata if provided
        let metadata: Option<HashMap<String, String>> = if let Some(v) = kwargs.delete(Symbol::new("metadata"))? {
            parse_metadata(v).ok()
        } else {
            None
        };
        
        // Get internal ID for this item
        let internal_id = {
            let mut label_map = self.label_to_id.lock().unwrap();
            let mut current_id = self.current_id.lock().unwrap();
            
            if label_map.contains_key(&label) {
                return Err(Error::new(
                    exception::arg_error(),
                    format!("Label '{}' already exists in index", label)
                ));
            }
            
            let id = *current_id;
            label_map.insert(label.clone(), id);
            *current_id += 1;
            id
        };
        
        // Store metadata
        {
            let mut metadata_store = self.metadata_store.lock().unwrap();
            metadata_store.insert(internal_id, ItemMetadata {
                label: label.clone(),
                metadata,
            });
        }
        
        // Add to HNSW
        {
            let hnsw = self.hnsw.lock().unwrap();
            hnsw.insert((&vec_data, internal_id));
        }
        
        Ok(value::qnil().as_value())
    }
    
    // Add multiple items in batch
    pub fn add_batch(&self, vectors: RArray, kwargs: RHash) -> Result<Value, Error> {
        let parallel: bool = if let Some(v) = kwargs.delete(Symbol::new("parallel"))? {
            TryConvert::try_convert(v).unwrap_or(true)
        } else {
            true
        };
        
        let labels: Option<RArray> = if let Some(v) = kwargs.delete(Symbol::new("labels"))? {
            TryConvert::try_convert(v).ok()
        } else {
            None
        };
        
        // Parse all vectors
        let mut data_points: Vec<(Vec<f32>, usize)> = Vec::new();
        let mut metadata_entries: Vec<(usize, ItemMetadata)> = Vec::new();
        
        for (i, vector) in vectors.each().enumerate() {
            let vector: RArray = TryConvert::try_convert(vector?)?;
            let vec_data = parse_vector(vector, self.dim)?;
            
            // Get or generate label
            let label = if let Some(ref labels_array) = labels {
                labels_array.entry::<String>(i as isize)?
            } else {
                let mut id = self.current_id.lock().unwrap();
                let label = id.to_string();
                *id += 1;
                label
            };
            
            // Get internal ID
            let internal_id = {
                let mut label_map = self.label_to_id.lock().unwrap();
                let mut current_id = self.current_id.lock().unwrap();
                
                if label_map.contains_key(&label) {
                    return Err(Error::new(
                        exception::arg_error(),
                        format!("Label '{}' already exists in index", label)
                    ));
                }
                
                let id = *current_id;
                label_map.insert(label.clone(), id);
                *current_id += 1;
                id
            };
            
            data_points.push((vec_data, internal_id));
            metadata_entries.push((internal_id, ItemMetadata {
                label,
                metadata: None,
            }));
        }
        
        // Store metadata
        {
            let mut metadata_store = self.metadata_store.lock().unwrap();
            for (id, metadata) in metadata_entries {
                metadata_store.insert(id, metadata);
            }
        }
        
        // Insert into HNSW
        {
            let hnsw = self.hnsw.lock().unwrap();
            if parallel {
                let data_refs: Vec<(&Vec<f32>, usize)> = data_points.iter().map(|(v, id)| (v, *id)).collect();
                hnsw.parallel_insert(&data_refs);
            } else {
                for (vec, id) in data_points {
                    hnsw.insert((&vec, id));
                }
            }
        }
        
        Ok(value::qnil().as_value())
    }
    
    // Search for k nearest neighbors
    pub fn search(&self, query: RArray, kwargs: RHash) -> Result<Value, Error> {
        let k: usize = if let Some(v) = kwargs.delete(Symbol::new("k"))? {
            TryConvert::try_convert(v).unwrap_or(10)
        } else {
            10
        };
        
        let include_distances: bool = if let Some(v) = kwargs.delete(Symbol::new("include_distances"))? {
            TryConvert::try_convert(v).unwrap_or(false)
        } else {
            false
        };
        
        // Parse query vector
        let query_vec = parse_vector(query, self.dim)?;
        
        // Set search ef if provided
        if let Some(v) = kwargs.delete(Symbol::new("ef"))? {
            if let Ok(ef) = TryConvert::try_convert(v) as Result<usize, _> {
                let mut ef_search = self.ef_search.lock().unwrap();
                *ef_search = ef;
            }
        }
        
        // Perform search
        let neighbors = {
            let hnsw = self.hnsw.lock().unwrap();
            let ef_search = self.ef_search.lock().unwrap();
            hnsw.search(&query_vec, k, *ef_search)
        };
        
        // Convert results
        let metadata_store = self.metadata_store.lock().unwrap();
        
        let indices = RArray::new();
        let distances = RArray::new();
        
        for neighbor in neighbors {
            if let Some(metadata) = metadata_store.get(&neighbor.d_id) {
                indices.push(RString::new(&metadata.label))?;
                distances.push(Float::from_f64(neighbor.distance as f64))?;
            }
        }
        
        if include_distances {
            let result = RArray::new();
            result.push(indices)?;
            result.push(distances)?;
            Ok(result.as_value())
        } else {
            Ok(indices.as_value())
        }
    }
    
    // Search with metadata included
    pub fn search_with_metadata(&self, query: RArray, kwargs: RHash) -> Result<Value, Error> {
        let k: usize = if let Some(v) = kwargs.delete(Symbol::new("k"))? {
            TryConvert::try_convert(v).unwrap_or(10)
        } else {
            10
        };
        
        // Parse query vector
        let query_vec = parse_vector(query, self.dim)?;
        
        // Perform search
        let neighbors = {
            let hnsw = self.hnsw.lock().unwrap();
            let ef_search = self.ef_search.lock().unwrap();
            hnsw.search(&query_vec, k, *ef_search)
        };
        
        // Build results with metadata
        let metadata_store = self.metadata_store.lock().unwrap();
        let results = RArray::new();
        
        for neighbor in neighbors {
            if let Some(item_metadata) = metadata_store.get(&neighbor.d_id) {
                let result = RHash::new();
                result.aset(Symbol::new("label"), RString::new(&item_metadata.label))?;
                result.aset(Symbol::new("distance"), Float::from_f64(neighbor.distance as f64))?;
                
                if let Some(ref meta) = item_metadata.metadata {
                    let meta_hash = RHash::new();
                    for (key, value) in meta {
                        meta_hash.aset(RString::new(key), RString::new(value))?;
                    }
                    result.aset(Symbol::new("metadata"), meta_hash)?;
                }
                
                results.push(result)?;
            }
        }
        
        Ok(results.as_value())
    }
    
    // Get current size of the index
    pub fn size(&self) -> Result<usize, Error> {
        let metadata_store = self.metadata_store.lock().unwrap();
        Ok(metadata_store.len())
    }
    
    // Check if index is empty
    pub fn empty(&self) -> Result<bool, Error> {
        Ok(self.size()? == 0)
    }
    
    // Set the ef parameter for search
    pub fn set_ef(&self, ef: usize) -> Result<Value, Error> {
        let mut ef_search = self.ef_search.lock().unwrap();
        *ef_search = ef;
        Ok(value::qnil().as_value())
    }
    
    // Get configuration
    pub fn config(&self) -> Result<RHash, Error> {
        let config = RHash::new();
        config.aset(Symbol::new("dim"), Integer::from_i64(self.dim as i64))?;
        
        let space_str = match self.space {
            DistanceType::Euclidean => "euclidean",
            DistanceType::Cosine => "cosine",
            DistanceType::InnerProduct => "inner_product",
        };
        config.aset(Symbol::new("space"), RString::new(space_str))?;
        
        let ef_search = self.ef_search.lock().unwrap();
        config.aset(Symbol::new("ef"), Integer::from_i64(*ef_search as i64))?;
        config.aset(Symbol::new("size"), Integer::from_i64(self.size()? as i64))?;
        
        Ok(config)
    }
    
    // Get statistics about the index
    pub fn stats(&self) -> Result<RHash, Error> {
        let stats = RHash::new();
        
        stats.aset(Symbol::new("size"), Integer::from_i64(self.size()? as i64))?;
        stats.aset(Symbol::new("dim"), Integer::from_i64(self.dim as i64))?;
        
        let ef_search = self.ef_search.lock().unwrap();
        stats.aset(Symbol::new("ef_search"), Integer::from_i64(*ef_search as i64))?;
        
        // TODO: Add more statistics from HNSW structure
        
        Ok(stats)
    }
    
    // Load index from file (class method)
    pub fn load(path: RString) -> Result<Self, Error> {
        let path_str = path.to_string()?;
        
        // Load metadata first to get dimensions and space
        let metadata_path = format!("{}.metadata", path_str);
        let metadata_file = File::open(&metadata_path)
            .map_err(|e| Error::new(exception::runtime_error(), format!("Failed to open metadata file: {}", e)))?;
        
        let (
            _metadata_store,
            _label_to_id, 
            _current_id,
            _dim,
            _space_str,
        ): (
            HashMap<usize, ItemMetadata>,
            HashMap<String, usize>,
            usize,
            usize,
            String,  // Changed from &str to String for deserialization
        ) = bincode::deserialize_from(metadata_file)
            .map_err(|e| Error::new(exception::runtime_error(), format!("Failed to load metadata: {}", e)))?;
        
        // Load HNSW structure
        let hnsw_dir = format!("{}_hnsw_data", path_str);
        let hnsw_path = std::path::Path::new(&hnsw_dir);
        
        // Create HnswIo and leak it to get 'static lifetime
        // This is a memory leak, but necessary due to hnsw_rs lifetime constraints
        // The memory will never be freed until the program exits
        let hnswio = Box::new(HnswIo::new(hnsw_path, "hnsw"));
        let hnswio_static: &'static mut HnswIo = Box::leak(hnswio);
        
        // Now we can load the HNSW with 'static lifetime
        let hnsw: Hnsw<'static, f32, DistL2> = hnswio_static.load_hnsw()
            .map_err(|e| Error::new(exception::runtime_error(), format!("Failed to load HNSW index: {}", e)))?;
        
        // Use the loaded metadata
        let metadata_store = _metadata_store;
        let label_to_id = _label_to_id;
        let current_id = _current_id;
        let dim = _dim;
        let space = match _space_str.as_str() {
            "euclidean" => DistanceType::Euclidean,
            "cosine" => DistanceType::Cosine,
            "inner_product" => DistanceType::InnerProduct,
            _ => return Err(Error::new(exception::runtime_error(), "Unknown distance type in saved file")),
        };
        
        // Use default ef_construction as ef_search
        let ef_search = 200;
        
        Ok(Self {
            hnsw: Arc::new(Mutex::new(hnsw)),
            dim,
            space,
            metadata_store: Arc::new(Mutex::new(metadata_store)),
            current_id: Arc::new(Mutex::new(current_id)),
            label_to_id: Arc::new(Mutex::new(label_to_id)),
            ef_search: Arc::new(Mutex::new(ef_search)),
        })
    }
    
    // Save index to file
    pub fn save(&self, path: RString) -> Result<Value, Error> {
        let path_str = path.to_string()?;
        
        // Create directory for HNSW structure
        let hnsw_dir = format!("{}_hnsw_data", path_str);
        std::fs::create_dir_all(&hnsw_dir)
            .map_err(|e| Error::new(exception::runtime_error(), format!("Failed to create directory: {}", e)))?;
        
        // Save HNSW structure
        {
            let hnsw = self.hnsw.lock().unwrap();
            hnsw.file_dump(&std::path::Path::new(&hnsw_dir), "hnsw")
                .map_err(|e| Error::new(exception::runtime_error(), format!("Failed to save HNSW: {}", e)))?;
        }
        
        // Save metadata
        let metadata_path = format!("{}.metadata", path_str);
        {
            let metadata_store = self.metadata_store.lock().unwrap();
            let label_to_id = self.label_to_id.lock().unwrap();
            let current_id = self.current_id.lock().unwrap();
            
            let metadata_data = (
                &*metadata_store,
                &*label_to_id,
                *current_id,
                self.dim,
                match self.space {
                    DistanceType::Euclidean => "euclidean",
                    DistanceType::Cosine => "cosine",
                    DistanceType::InnerProduct => "inner_product",
                },
            );
            
            let file = File::create(&metadata_path)
                .map_err(|e| Error::new(exception::runtime_error(), format!("Failed to create metadata file: {}", e)))?;
            
            bincode::serialize_into(file, &metadata_data)
                .map_err(|e| Error::new(exception::runtime_error(), format!("Failed to save metadata: {}", e)))?;
        }
        
        Ok(value::qnil().as_value())
    }
}

// Helper function to parse a Ruby array into a Vec<f32>
fn parse_vector(array: RArray, expected_dim: usize) -> Result<Vec<f32>, Error> {
    let len = array.len();
    if len != expected_dim {
        return Err(Error::new(
            exception::arg_error(),
            format!("Vector dimension mismatch: expected {}, got {}", expected_dim, len)
        ));
    }
    
    let mut vec = Vec::with_capacity(len);
    for item in array.each() {
        let value: f64 = TryConvert::try_convert(item?)
            .map_err(|_| Error::new(exception::type_error(), "Vector elements must be numeric"))?;
        vec.push(value as f32);
    }
    
    Ok(vec)
}

// Helper function to parse metadata
fn parse_metadata(value: Value) -> Result<HashMap<String, String>, Error> {
    let hash: RHash = TryConvert::try_convert(value)
        .map_err(|_| Error::new(exception::type_error(), "Metadata must be a hash"))?;
    
    let mut metadata = HashMap::new();
    
    hash.foreach(|key: Value, value: Value| {
        // Handle both string and symbol keys
        let key_str = if let Ok(s) = String::try_convert(key) {
            s
        } else if let Ok(sym) = Symbol::try_convert(key) {
            sym.name()?.to_string()
        } else {
            return Err(Error::new(exception::type_error(), "Metadata keys must be strings or symbols"));
        };
        
        // Convert value to string
        let value_str: String = TryConvert::try_convert(value)
            .map_err(|_| Error::new(exception::type_error(), "Metadata values must be strings"))?;
        
        metadata.insert(key_str, value_str);
        Ok(ForEach::Continue)
    })?;
    
    Ok(metadata)
}

// Initialize the HNSW module
pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let class = parent.define_class("HNSW", class::object())?;
    
    class.define_singleton_method("new", function!(HnswIndex::new, 1))?;
    class.define_singleton_method("load", function!(HnswIndex::load, 1))?;
    class.define_method("add_item", method!(HnswIndex::add_item, 2))?;
    class.define_method("add_batch", method!(HnswIndex::add_batch, 2))?;
    class.define_method("search", method!(HnswIndex::search, 2))?;
    class.define_method("search_with_metadata", method!(HnswIndex::search_with_metadata, 2))?;
    class.define_method("size", method!(HnswIndex::size, 0))?;
    class.define_method("empty?", method!(HnswIndex::empty, 0))?;
    class.define_method("set_ef", method!(HnswIndex::set_ef, 1))?;
    class.define_method("config", method!(HnswIndex::config, 0))?;
    class.define_method("stats", method!(HnswIndex::stats, 0))?;
    class.define_method("save", method!(HnswIndex::save, 1))?;
    
    Ok(())
}