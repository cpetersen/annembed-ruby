<img src="/docs/assets/clusterkit-wide.png" alt="clusterkit" height="80px">

A high-performance clustering and dimensionality reduction toolkit for Ruby, powered by best-in-class Rust implementations.

## 🙏 Acknowledgments & Attribution

ClusterKit builds upon excellent work from the Rust ecosystem:

- **[annembed](https://github.com/jean-pierreBoth/annembed)** - Provides the core UMAP, t-SNE, and other dimensionality reduction algorithms. Created by Jean-Pierre Both.
- **[hdbscan](https://github.com/tom-whitehead/hdbscan)** - Provides the HDBSCAN density-based clustering implementation. A Rust port of the original HDBSCAN algorithm.

This gem would not be possible without these foundational libraries. Please consider starring their repositories if you find ClusterKit useful.

## Features

- **Dimensionality Reduction Algorithms**:
  - UMAP (Uniform Manifold Approximation and Projection) - powered by annembed
  - PCA (Principal Component Analysis)
  - SVD (Singular Value Decomposition)

- **Advanced Clustering**:
  - K-means clustering with automatic k selection via elbow method
  - HDBSCAN (Hierarchical Density-Based Spatial Clustering) for density-based clustering with noise detection
  - Silhouette scoring for cluster quality evaluation

- **High Performance**:
  - Leverages Rust's speed and parallelization
  - Efficient memory usage
  - Support for large datasets

- **Easy to Use**:
  - Simple, scikit-learn-like API
  - Consistent interface across algorithms
  - Comprehensive documentation and examples

- **Visualization Tools**:
  - Interactive HTML visualizations
  - Comparison of different algorithms
  - Built-in rake tasks for quick experimentation

## API Structure

ClusterKit organizes its functionality into clear modules:

- **`ClusterKit::Dimensionality`** - All dimensionality reduction algorithms
  - `ClusterKit::Dimensionality::UMAP` - UMAP implementation
  - `ClusterKit::Dimensionality::PCA` - PCA implementation
  - `ClusterKit::Dimensionality::SVD` - SVD implementation
- **`ClusterKit::Clustering`** - All clustering algorithms
  - `ClusterKit::Clustering::KMeans` - K-means clustering
  - `ClusterKit::Clustering::HDBSCAN` - HDBSCAN clustering
- **`ClusterKit::Utils`** - Utility functions
- **`ClusterKit::Preprocessing`** - Data preprocessing tools

All user-facing classes are in these modules. Implementation details are kept private.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'clusterkit'
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install clusterkit

### Prerequisites

- Ruby 2.7 or higher
- Rust toolchain (for building from source)

## Quick Start - Interactive Example

Copy and paste this **entire block** into IRB to try out the main features (including the srand line for reproducible results):

```ruby
require 'clusterkit'

# Generate sample high-dimensional data with structure
# This simulates real-world data like text embeddings or image features
puts "Creating sample data: 100 points in 50 dimensions with 3 clusters"

# Use a fixed seed for reproducibility in this example
# Important: Random data without structure can cause UMAP errors
srand(42)

# Create data with some inherent structure (3 clusters)
# Using better separated clusters to avoid UMAP convergence issues
data = []
3.times do |cluster|
  # Each cluster has a different center, well-separated
  center = Array.new(50) { rand * 0.1 + cluster * 2.0 }

  # Add 33 points around each center with controlled noise
  33.times do
    point = center.map { |c| c + (rand - 0.5) * 0.3 }
    data << point
  end
end

# Add one more point to make it 100
data << Array.new(50) { rand * 6.0 }  # Scale to match cluster range

# ============================================================
# 1. DIMENSIONALITY REDUCTION - Visualize high-dim data in 2D
# ============================================================

puts "\n1. DIMENSIONALITY REDUCTION:"

# UMAP - Best for preserving both local and global structure
puts "Running UMAP..."
# Note: Using n_neighbors=5 for better stability with varied data
# Lower n_neighbors helps avoid "isolated point" errors
umap = ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5)
umap_result = umap.fit_transform(data)
puts "  ✓ Reduced to #{umap_result.first.size}D: #{umap_result[0..2].map { |p| p.map { |v| v.round(3) } }}"

# PCA - Fast linear reduction, good for finding main variations
puts "Running PCA..."
pca = ClusterKit::Dimensionality::PCA.new(n_components: 2)
pca_result = pca.fit_transform(data)
puts "  ✓ Reduced to #{pca_result.first.size}D: #{pca_result[0..2].map { |p| p.map { |v| v.round(3) } }}"
puts "  ✓ Explained variance: #{(pca.explained_variance_ratio.sum * 100).round(1)}%"

# ============================================================
# 2. CLUSTERING - Find groups in your data
# ============================================================

puts "\n2. CLUSTERING:"

# K-means - When you know roughly how many clusters to expect
puts "Running K-means..."
# First, find optimal k using elbow method
elbow_scores = ClusterKit::Clustering::KMeans.elbow_method(umap_result, k_range: 2..6)
optimal_k = ClusterKit::Clustering::KMeans.detect_optimal_k(elbow_scores)
puts "  ✓ Optimal k detected: #{optimal_k}"

kmeans = ClusterKit::Clustering::KMeans.new(k: optimal_k)
kmeans_labels = kmeans.fit_predict(umap_result)
puts "  ✓ Found #{kmeans_labels.uniq.size} clusters"

# HDBSCAN - When you don't know the number of clusters and have noise
puts "Running HDBSCAN..."
hdbscan = ClusterKit::Clustering::HDBSCAN.new(min_samples: 5, min_cluster_size: 10)
hdbscan_labels = hdbscan.fit_predict(umap_result)
puts "  ✓ Found #{hdbscan.n_clusters} clusters"
puts "  ✓ Identified #{hdbscan.n_noise_points} noise points (#{(hdbscan.noise_ratio * 100).round(1)}%)"

# ============================================================
# 3. EVALUATION - How good are the clusters?
# ============================================================

puts "\n3. CLUSTER EVALUATION:"
silhouette = ClusterKit::Clustering.silhouette_score(umap_result, kmeans_labels)
puts "  K-means silhouette score: #{silhouette.round(3)} (closer to 1 is better)"

# Filter noise for HDBSCAN evaluation
non_noise = hdbscan_labels.each_with_index.select { |l, _| l != -1 }.map(&:last)
if non_noise.any?
  filtered_data = non_noise.map { |i| umap_result[i] }
  filtered_labels = non_noise.map { |i| hdbscan_labels[i] }
  hdbscan_silhouette = ClusterKit::Clustering.silhouette_score(filtered_data, filtered_labels)
  puts "  HDBSCAN silhouette score: #{hdbscan_silhouette.round(3)} (excluding noise)"
end

puts "\n✅ All done! Try visualizing with: rake clusterkit:visualize"
```

## Detailed Usage

### API Structure

ClusterKit organizes its algorithms into logical modules:

- **`ClusterKit::Dimensionality`** - Algorithms for reducing data dimensions
  - `UMAP` - Non-linear manifold learning
  - `PCA` - Principal Component Analysis
  - `SVD` - Singular Value Decomposition

- **`ClusterKit::Clustering`** - Algorithms for grouping data
  - `KMeans` - Partition-based clustering
  - `HDBSCAN` - Density-based clustering with noise detection

### Dimensionality Reduction

#### UMAP (Uniform Manifold Approximation and Projection)

**Important**: UMAP requires data with some structure. Pure random data may fail.
For testing, create data with patterns (see Quick Start example above).

```ruby
# Generate sample data with structure (not pure random)
data = []
100.times do |i|
  # Create points that cluster in groups
  group = i / 25  # 4 groups
  base = group * 2.0
  point = Array.new(50) { |j| base + rand * 0.8 + j * 0.01 }
  data << point
end

# Create UMAP instance
umap = ClusterKit::Dimensionality::UMAP.new(
  n_components: 2,      # Target dimensions (default: 2)
  n_neighbors: 5,       # Number of neighbors (default: 15, use 5 for small datasets)
  random_seed: 42,      # For reproducibility (default: nil for best performance)
  nb_grad_batch: 10,    # Gradient descent batches (default: 10, lower = faster)
  nb_sampling_by_edge: 8 # Negative samples per edge (default: 8, lower = faster)
)

# Fit and transform data
embedded = umap.fit_transform(data)

# IMPORTANT: Seed behavior
# - WITH random_seed: Fully reproducible results using serial processing (slower)
# - WITHOUT random_seed: Faster parallel processing but non-deterministic results

# Or fit once and transform multiple datasets
# Example: Split your data into training and test sets
# Note: Using structured data (not pure random) for better UMAP results
all_data = []
200.times do |i|
  # Create data with some structure - points cluster around different regions
  center = (i / 50) * 2.0  # 4 rough groups
  point = Array.new(50) { |j| center + rand * 0.5 + j * 0.01 }
  all_data << point
end

training_data = all_data[0...150]   # First 150 samples for training
test_data = all_data[150..-1]       # Last 50 samples for testing

umap.fit(training_data)
test_embedded = umap.transform(test_data)

# Save and load fitted models
umap.save_model("umap_model.bin")                  # Save the fitted model
loaded_umap = ClusterKit::Dimensionality::UMAP.load_model("umap_model.bin")  # Load it later
new_data_embedded = loaded_umap.transform(new_data)  # Use loaded model for new data

# Save and load transformed data (useful for caching results)
ClusterKit::Dimensionality::UMAP.save_data(embedded, "embeddings.json")
cached_embeddings = ClusterKit::Dimensionality::UMAP.load_data("embeddings.json")

# Note: The library automatically adjusts n_neighbors if it's too large for your dataset
```

#### PCA (Principal Component Analysis)

```ruby
pca = ClusterKit::Dimensionality::PCA.new(n_components: 2)
transformed = pca.fit_transform(data)

# Access explained variance
puts "Explained variance ratio: #{pca.explained_variance_ratio}"
puts "Cumulative explained variance: #{pca.cumulative_explained_variance_ratio}"

# Inverse transform to reconstruct original data
reconstructed = pca.inverse_transform(transformed)
```


#### SVD (Singular Value Decomposition)

```ruby
# Direct SVD decomposition using the class interface
svd = ClusterKit::Dimensionality::SVD.new(n_components: 10, n_iter: 5)
u, s, vt = svd.fit_transform(data)

# U: left singular vectors (documents in LSA)
# S: singular values (importance of each component)
# V^T: right singular vectors (terms in LSA)

puts "Shape of U: #{u.size}x#{u.first.size}"
puts "Singular values: #{s[0..4].map { |v| v.round(2) }}"
puts "Shape of V^T: #{vt.size}x#{vt.first.size}"

# For dimensionality reduction, use U * S
reduced = u.map.with_index do |row, i|
  row.map.with_index { |val, j| val * s[j] }
end

# Or use the convenience method
u, s, vt = ClusterKit.svd(data, 10, n_iter: 5)
```


### Clustering

#### K-means with Automatic K Selection

```ruby
# Find optimal number of clusters
elbow_scores = ClusterKit::Clustering::KMeans.elbow_method(data, k_range: 2..10)
optimal_k = ClusterKit::Clustering::KMeans.detect_optimal_k(elbow_scores)

# Cluster with optimal k
kmeans = ClusterKit::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
labels = kmeans.fit_predict(data)

# Access cluster centers
centers = kmeans.cluster_centers
```

#### HDBSCAN (Density-Based Clustering)

```ruby
# HDBSCAN automatically determines the number of clusters
# and can identify noise points
hdbscan = ClusterKit::Clustering::HDBSCAN.new(
  min_samples: 5,        # Minimum samples in neighborhood
  min_cluster_size: 10,  # Minimum cluster size
  metric: 'euclidean'    # Distance metric
)

labels = hdbscan.fit_predict(data)

# Noise points are labeled as -1
puts "Clusters found: #{hdbscan.n_clusters}"
puts "Noise points: #{hdbscan.n_noise_points} (#{(hdbscan.noise_ratio * 100).round(1)}%)"

# Access additional HDBSCAN information
probabilities = hdbscan.probabilities      # Cluster membership probabilities
outlier_scores = hdbscan.outlier_scores   # Outlier scores for each point
```

### HNSW - Fast Nearest Neighbor Search

ClusterKit includes HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search, useful for building recommendation systems, similarity search, and as a building block for other algorithms.

```ruby
# Create an HNSW index for fast similarity search
index = ClusterKit::HNSW.new(
  dim: 128,               # Vector dimensions
  space: :euclidean,      # Distance metric (currently only euclidean)
  m: 16,                  # Graph connectivity (higher = better recall, more memory)
  ef_construction: 200,   # Build quality (higher = better graph, slower build)
  max_elements: 10000,    # Maximum number of elements
  random_seed: 42         # Optional: for reproducibility
)

# Add items to the index
index.add_item([0.1, 0.2, ...], label: 'item1', metadata: { category: 'A' })
index.add_batch(vectors, labels: labels)

# Search for nearest neighbors
results = index.search(query_vector, k: 10)
# Returns array of labels

# Search with full details
results = index.search_with_metadata(query_vector, k: 10)
# Returns array of hashes with :label, :distance, :metadata

# Adjust search quality/speed trade-off
index.set_ef(100)  # Higher ef = better recall, slower search

# Save and load indices
index.save('/path/to/index')
loaded_index = ClusterKit::HNSW.load('/path/to/index')
```

#### When to Use HNSW

HNSW is ideal for:
- **Recommendation Systems**: Find similar items/users quickly
- **Semantic Search**: Find documents with similar embeddings
- **Duplicate Detection**: Identify near-duplicate content
- **Clustering Support**: As a fast neighbor graph for HDBSCAN
- **Real-time Applications**: When you need sub-millisecond search times

#### Configuration Guidelines

```ruby
# High recall (>0.95) - Best quality, slower
ClusterKit::HNSW.new(
  dim: dim,
  m: 32,
  ef_construction: 400
).tap { |idx| idx.set_ef(100) }

# Balanced (>0.90 recall) - Good quality, fast
ClusterKit::HNSW.new(
  dim: dim,
  m: 16,
  ef_construction: 200
).tap { |idx| idx.set_ef(50) }

# Speed optimized (>0.85 recall) - Fastest, acceptable quality
ClusterKit::HNSW.new(
  dim: dim,
  m: 8,
  ef_construction: 100
).tap { |idx| idx.set_ef(20) }
```

#### Important Notes

1. **Memory Usage**: HNSW keeps the entire index in memory. Estimate: `(num_items * (dim * 4 + m * 16))` bytes
2. **Distance Metrics**: Currently only Euclidean distance is fully supported
3. **Loading Behavior**: Due to Rust lifetime constraints, loading an index creates a small memory leak (the index metadata persists until program exit). This is typically negligible for most applications.
4. **Build Time**: Index construction is O(N * log(N)). For large datasets (>1M items), consider building offline

#### Example: Semantic Search System

```ruby
# Build a simple semantic search system
documents = load_documents()
embeddings = generate_embeddings(documents)  # Use red-candle or similar

# Build search index
search_index = ClusterKit::HNSW.new(
  dim: embeddings.first.size,
  m: 16,
  ef_construction: 200,
  max_elements: documents.size
)

# Add all documents
documents.each_with_index do |doc, i|
  search_index.add_item(
    embeddings[i],
    label: i,
    metadata: { title: doc[:title], url: doc[:url] }
  )
end

# Search function
def search(query, index, k: 10)
  query_embedding = generate_embedding(query)
  results = index.search_with_metadata(query_embedding, k: k)

  results.map do |result|
    {
      title: result[:metadata]['title'],
      url: result[:metadata]['url'],
      similarity: 1.0 - result[:distance]  # Convert distance to similarity
    }
  end
end

# Save for later use
search_index.save('document_index')
```

### Visualization

ClusterKit includes a built-in visualization tool:

```bash
# Generate interactive visualization
rake clusterkit:visualize

# With options
rake clusterkit:visualize[output.html,iris,both]  # filename, dataset, clustering method

# Dataset options: clusters, swiss, iris
# Clustering options: kmeans, hdbscan, both
```

This creates an interactive HTML file with:
- Side-by-side comparison of dimensionality reduction methods
- Clustering results visualization
- Performance metrics
- Interactive Plotly.js charts

## Choosing the Right Algorithm

### Dimensionality Reduction

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **UMAP** | General purpose, preserving both local and global structure | Fast, scalable, supports transform() | Requires tuning parameters |
| **PCA** | Linear relationships, feature extraction | Very fast, interpretable, deterministic | Only captures linear relationships |
| **SVD** | Text analysis (LSA), recommendation systems | Memory efficient, good for sparse data | Only linear relationships |

### Clustering

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **K-means** | Spherical clusters, known cluster count | Fast, simple, deterministic with seed | Requires knowing k, assumes spherical clusters |
| **HDBSCAN** | Unknown cluster count, irregular shapes, noise | Finds clusters automatically, handles noise | More complex parameters, slower than k-means |

### Recommended Combinations

- **Document Clustering**: UMAP (20D) → HDBSCAN
- **Image Clustering**: PCA (50D) → K-means
- **Customer Segmentation**: UMAP (10D) → K-means with elbow method
- **Anomaly Detection**: UMAP (5D) → HDBSCAN (outliers are noise points)
- **Visualization**: UMAP (2D) or PCA (2D) → visual inspection

## Advanced Examples

### Document Clustering Pipeline

```ruby
# Typical NLP workflow: embed → reduce → cluster
documents = ["text1", "text2", ...]  # Your documents

# Step 1: Get embeddings (use your favorite embedding model)
# embeddings = get_embeddings(documents)  # e.g., from red-candle

# Step 2: Reduce dimensions for better clustering
umap = ClusterKit::Dimensionality::UMAP.new(n_components: 20, n_neighbors: 10)
reduced_embeddings = umap.fit_transform(embeddings)

# Step 3: Find clusters
hdbscan = ClusterKit::Clustering::HDBSCAN.new(
  min_samples: 5,
  min_cluster_size: 10
)
clusters = hdbscan.fit_predict(reduced_embeddings)

# Step 4: Analyze results
clusters.each_with_index do |cluster_id, doc_idx|
  next if cluster_id == -1  # Skip noise
  puts "Document '#{documents[doc_idx]}' belongs to cluster #{cluster_id}"
end
```

### Model Persistence

```ruby
# Save trained model
umap.save("model.bin")

# Load trained model
loaded_umap = ClusterKit::Dimensionality::UMAP.load("model.bin")
result = loaded_umap.transform(new_data)
```

## Performance Tips

1. **Large Datasets**: Use sampling for initial parameter tuning
2. **HDBSCAN**: Reduce to 10-50 dimensions with UMAP first for better results
3. **Memory**: Process in batches for very large datasets
4. **Speed**: Compile with optimizations: `RUSTFLAGS="-C target-cpu=native" bundle install`

### UMAP Reproducibility vs Performance

ClusterKit's UMAP implementation offers two modes:

| Mode | Usage | Performance | Reproducibility |
|------|-------|-------------|-----------------|
| **Fast (default)** | `UMAP.new()` | Parallel processing, ~25-35% faster | Non-deterministic |
| **Reproducible** | `UMAP.new(random_seed: 42)` | Serial processing | Fully deterministic |

**When to use each mode:**
- **Production/Analysis**: Use default (no seed) for best performance when exact reproducibility isn't critical
- **Research/Testing**: Use a seed when you need reproducible results for comparisons or debugging
- **CI/Testing**: Always use a seed to ensure consistent test results

**Note**: The `transform` method is always deterministic once a model is fitted, regardless of seed usage during training.


## Troubleshooting

### UMAP "isolated point" or "graph not connected" errors

This error occurs when UMAP cannot find enough neighbors for some points. Solutions:

1. **Reduce n_neighbors**: Use a smaller value (e.g., 5 instead of 15)
   ```ruby
   umap = ClusterKit::Dimensionality::UMAP.new(n_neighbors: 5)
   ```

2. **Add structure to your data**: Completely random data may not work well
   ```ruby
   # Bad: Pure random data with no structure
   data = Array.new(100) { Array.new(50) { rand } }

   # Good: Data with clusters or patterns (see Quick Start example)
   # Create clusters with centers and add points around them
   ```

3. **Ensure sufficient data points**: UMAP needs at least n_neighbors + 1 points

4. **Use consistent data generation**: For examples/testing, use a fixed seed
   ```ruby
   srand(42)  # Ensures reproducible data generation
   ```

Note: Real-world embeddings (from text, images, etc.) typically have inherent structure and work better than random data.

### Memory issues with large datasets

- Process in batches for datasets > 100k points
- Use PCA to reduce dimensions before UMAP

### Installation issues

- Ensure Rust is installed: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- For M1/M2 Macs, ensure you have the latest Xcode command line tools
- Clear the build cache if needed: `bundle exec rake clean`

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests.

To install this gem onto your local machine, run `bundle exec rake install`.

## Testing

```bash
# Run all tests
bundle exec rspec

# Run specific test file
bundle exec rspec spec/clusterkit/clustering_spec.rb

# Run with coverage
COVERAGE=true bundle exec rspec
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/cpetersen/clusterkit.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Citation

If you use ClusterKit in your research, please cite:

```
@software{clusterkit,
  author = {Chris Petersen},
  title = {ClusterKit: High-Performance Clustering and Dimensionality Reduction for Ruby},
  year = {2024},
  url = {https://github.com/cpetersen/clusterkit}
}
```

And please also cite the underlying libraries:
- [annembed](https://github.com/jean-pierreBoth/annembed) for dimensionality reduction algorithms
- [hdbscan](https://github.com/petabi/hdbscan) for HDBSCAN clustering
