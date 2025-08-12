# annembed-ruby

High-performance dimensionality reduction for Ruby, powered by the [annembed](https://github.com/jean-pierreBoth/annembed) Rust crate.

## Features

- **Multiple algorithms**: UMAP, t-SNE, LargeVis, and Diffusion Maps for dimensionality reduction
- **Linear methods**: PCA and SVD for fast linear dimensionality reduction
- **Clustering**: K-means clustering with automatic k selection via elbow method
- **High performance**: Leverages Rust's speed and parallelization
- **Easy to use**: Simple, scikit-learn-like API
- **Model persistence**: Save and load trained models
- **Flexible**: Supports various configuration options

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'annembed-ruby'
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install annembed-ruby

### Prerequisites

- Ruby 2.7 or higher
- Rust toolchain (for building from source)

## Quick Start - Interactive Example

Copy and paste this into IRB to try out the main features:

```ruby
require 'annembed'

# Generate sample high-dimensional data
# Imagine this is text embeddings, image features, or any high-dim data
puts "Creating sample data: 100 points in 50 dimensions"
data = Array.new(100) { Array.new(50) { rand } }

# ============================================================
# 1. UMAP - State-of-the-art non-linear dimensionality reduction
# ============================================================
puts "\n1. UMAP - Reducing to 2D for visualization"
embedder = AnnEmbed::Embedder.new(
  method: :umap,
  n_components: 2,    # Reduce to 2D
  n_neighbors: 15      # Balance local/global structure
)

# Fit and transform the data
umap_result = embedder.fit_transform(data)
puts "   Shape: #{umap_result.size} points × #{umap_result.first.size} dimensions"
puts "   First point: [#{umap_result.first.map { |v| v.round(3) }.join(', ')}]"

# Save the trained model
embedder.save("umap_model.bin")
puts "   Model saved to umap_model.bin"

# ============================================================
# 2. t-SNE - Popular for visualization, especially clusters
# ============================================================
puts "\n2. t-SNE - Alternative visualization method"
tsne = AnnEmbed::Embedder.new(
  method: :tsne,
  n_components: 2,
  perplexity: 30.0    # Balances local/global structure
)

tsne_result = tsne.fit_transform(data)
puts "   Shape: #{tsne_result.size} points × #{tsne_result.first.size} dimensions"
puts "   First point: [#{tsne_result.first.map { |v| v.round(3) }.join(', ')}]"

# ============================================================
# 3. SVD - Fast linear dimensionality reduction
# ============================================================
puts "\n3. SVD - Linear dimensionality reduction (like PCA)"
# Reduce to top 10 components
u, s, vt = AnnEmbed.svd(data, 10, n_iter: 2)
puts "   U shape: #{u.size}×#{u.first.size} (transformed data)"
puts "   S values: [#{s[0..2].map { |v| v.round(2) }.join(', ')}, ...]"
puts "   V^T shape: #{vt.size}×#{vt.first.size} (components)"

# The reduced data is in U
svd_result = u
puts "   First point: [#{svd_result.first[0..2].map { |v| v.round(3) }.join(', ')}, ...]"

# ============================================================
# 4. Transform new data with a trained model
# ============================================================
puts "\n4. Transforming new data with saved UMAP model"
# Load the saved model
loaded = AnnEmbed::Embedder.load("umap_model.bin")

# New data (5 new points)
new_data = Array.new(5) { Array.new(50) { rand } }
new_embedding = loaded.transform(new_data)
puts "   New data shape: #{new_embedding.size}×#{new_embedding.first.size}"
puts "   First new point: [#{new_embedding.first.map { |v| v.round(3) }.join(', ')}]"

# ============================================================
# 5. Comparison - Which method to use?
# ============================================================
puts "\n5. Quick comparison:"
puts "   UMAP: Best for preserving both local and global structure"
puts "   t-SNE: Great for visualizing clusters, but slower"
puts "   SVD: Fastest, linear, good for denoising or pre-processing"

# ============================================================
# 6. Practical tip: Reduce dimensions for faster similarity search
# ============================================================
puts "\n6. Example: Speeding up similarity search"
# Original: 100 points × 50 dimensions = 5000 numbers to store
# After UMAP: 100 points × 2 dimensions = 200 numbers to store
# That's 25× less storage and faster distance calculations!

puts "\nStorage comparison:"
puts "   Original: #{data.size * data.first.size} floats"
puts "   After UMAP: #{umap_result.size * umap_result.first.size} floats"
puts "   Reduction: #{((1 - (umap_result.first.size.to_f / data.first.size)) * 100).round(1)}%"

puts "\n✅ Done! You've just reduced 50D data to 2D using three different methods!"
```

## Quick Start - Simplified API

For convenience, you can also use the simplified API:

```ruby
require 'annembed'

# Generate sample data
data = Array.new(100) { Array.new(50) { rand } }

# One-line dimensionality reduction
umap_2d = AnnEmbed.umap(data, n_components: 2)
tsne_2d = AnnEmbed.tsne(data, n_components: 2)
u, s, vt = AnnEmbed.svd(data, 10)  # Top 10 components

# Results are ready to use!
puts "UMAP result: #{umap_2d.first}"
puts "t-SNE result: #{tsne_2d.first}"
puts "SVD result: #{u.first}"
```

## API Reference

### AnnEmbed::Embedder

The universal class for all dimensionality reduction algorithms.

```ruby
# Create an embedder with any supported method
embedder = AnnEmbed::Embedder.new(
  method: :umap,       # :umap, :tsne, :largevis, or :diffusion
  n_components: 2,     # Target dimensions
  **options           # Method-specific options
)

# Methods work the same for all algorithms
result = embedder.fit_transform(data)
embedder.save("model.bin")
loaded = AnnEmbed::Embedder.load("model.bin")
```

### AnnEmbed::SVD

Randomized Singular Value Decomposition for fast linear dimensionality reduction.

```ruby
# Perform SVD
u, s, vt = AnnEmbed.svd(matrix, k, n_iter: 2)

# Parameters:
#   matrix: 2D array of data
#   k: Number of components to keep
#   n_iter: Number of iterations for randomized algorithm (default: 2)

# Returns:
#   u: Left singular vectors (transformed data)
#   s: Singular values (importance of each component)
#   vt: Right singular vectors transposed (components)

# Example: Reduce 100×50 matrix to 100×10
data = Array.new(100) { Array.new(50) { rand } }
u, s, vt = AnnEmbed.svd(data, 10)
reduced_data = u  # This is your reduced 100×10 data
```

### AnnEmbed::PCA

Principal Component Analysis for linear dimensionality reduction.

```ruby
# Simple PCA
pca = AnnEmbed::PCA.new(n_components: 2)
transformed = pca.fit_transform(data)

# Check explained variance
puts "Explained variance ratio: #{pca.explained_variance_ratio}"
puts "Cumulative variance: #{pca.cumulative_explained_variance_ratio}"

# Inverse transform to reconstruct data
reconstructed = pca.inverse_transform(transformed)

# Module-level convenience method
transformed = AnnEmbed.pca(data, n_components: 2)
```

### AnnEmbed::Clustering

K-means clustering for grouping similar data points.

```ruby
# Simple clustering
kmeans = AnnEmbed::Clustering::KMeans.new(k: 3)
labels = kmeans.fit_predict(data)

# Advanced usage with all options
kmeans = AnnEmbed::Clustering::KMeans.new(
  k: 5,              # Number of clusters
  max_iter: 300,     # Maximum iterations
  random_seed: 42    # For reproducibility
)

# Fit the model
kmeans.fit(data)

# Get cluster assignments
labels = kmeans.labels

# Get cluster centers
centers = kmeans.cluster_centers

# Get inertia (sum of squared distances to nearest centroid)
inertia = kmeans.inertia

# Predict clusters for new data
new_labels = kmeans.predict(new_data)

# Find optimal k using elbow method
results = AnnEmbed::Clustering.elbow_method(data, k_range: 2..10)
# Returns hash: {2 => inertia_k2, 3 => inertia_k3, ...}

# Detect optimal k from elbow results
optimal_k = AnnEmbed::Clustering.detect_optimal_k(results)
# Returns the k value at the "elbow" of the curve

# Or do it all automatically
optimal_k, labels, centroids, inertia = AnnEmbed::Clustering.optimal_kmeans(data, k_range: 2..10)
# Automatically finds optimal k and performs clustering

# Calculate clustering quality with silhouette score
score = AnnEmbed::Clustering.silhouette_score(data, labels)
# Returns value between -1 (poor) and 1 (excellent)
```

#### Complete Clustering Workflow

```ruby
require 'annembed'

# 1. Load or generate high-dimensional data
data = load_your_data()  # e.g., text embeddings, image features

# 2. Reduce dimensions for better clustering
umap = AnnEmbed::Embedder.new(method: :umap, n_components: 2)
reduced_data = umap.fit_transform(data)

# 3. Find optimal number of clusters automatically
optimal_k, labels, centroids, inertia = AnnEmbed::Clustering.optimal_kmeans(
  reduced_data, 
  k_range: 2..10
)
puts "Found #{optimal_k} clusters with inertia: #{inertia.round(2)}"

# Or manually with more control:
# elbow_results = AnnEmbed::Clustering.elbow_method(reduced_data, k_range: 2..10)
# optimal_k = AnnEmbed::Clustering.detect_optimal_k(elbow_results)
# kmeans = AnnEmbed::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
# labels = kmeans.fit_predict(reduced_data)

# 5. Evaluate clustering quality
silhouette = AnnEmbed::Clustering.silhouette_score(reduced_data, labels)
puts "Silhouette score: #{silhouette.round(3)}"

# 6. Use clusters for downstream tasks
labels.each_with_index do |cluster_id, point_idx|
  puts "Point #{point_idx} belongs to cluster #{cluster_id}"
end
```

### AnnEmbed::UMAP

The main class for UMAP dimensionality reduction.

#### Initialization

```ruby
umap = AnnEmbed::UMAP.new(
  n_components: 2,    # Target number of dimensions (default: 2)
  n_neighbors: 15,    # Number of neighbors for manifold approximation (default: 15)
  random_seed: 42     # Random seed for reproducibility (optional)
)
```

#### Methods

##### `#fit(data)`
Train the model on the provided data.

```ruby
data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
umap.fit(data)  # Returns self for method chaining
```

##### `#transform(data)`
Transform new data using the fitted model.

```ruby
new_data = [[7.0, 8.0], [9.0, 10.0]]
reduced = umap.transform(new_data)  # Returns 2D array of reduced dimensions
```

##### `#fit_transform(data)`
Fit the model and transform the data in one step.

```ruby
embedding = umap.fit_transform(data)  # Returns transformed data
```

##### `#fitted?`
Check if the model has been fitted.

```ruby
if umap.fitted?
  puts "Model is ready to transform data"
end
```

##### `#save(path)`
Save the fitted model to a file.

```ruby
umap.save("my_model.bin")
```

##### `.load(path)`
Load a previously saved model.

```ruby
umap = AnnEmbed::UMAP.load("my_model.bin")
```

#### Data Export/Import Utilities

For caching transformed results:

```ruby
# Export transformed data to JSON
AnnEmbed::UMAP.export_data(embedding, "embeddings.json")

# Import previously exported data
cached_embedding = AnnEmbed::UMAP.import_data("embeddings.json")
```

## Practical Example: Reducing Text Embeddings

A common use case for UMAP is reducing high-dimensional embeddings (e.g., from OpenAI, Cohere, etc.) to lower dimensions for storage and visualization.

### Step 1: Training the Model

```ruby
require 'annembed'
require 'json'

# Load your training embeddings (e.g., 768-dim vectors from a language model)
training_embeddings = load_embeddings_from_database(limit: 10000)

# Create and configure UMAP
umap = AnnEmbed::UMAP.new(
  n_components: 50,     # Reduce to 50 dimensions
  n_neighbors: 30,      # Higher for more global structure
  random_seed: 42       # For reproducibility
)

# Train the model
puts "Training UMAP on #{training_embeddings.length} embeddings..."
reduced_embeddings = umap.fit_transform(training_embeddings)

# Save the trained model
umap.save("models/text_umap_768_to_50.bin")
puts "Model saved!"

# Optionally cache the reduced embeddings
AnnEmbed::UMAP.export_data(reduced_embeddings, "cache/reduced_embeddings.json")
```

### Step 2: Production Use - Single Embedding

```ruby
require 'annembed'

# Load the pre-trained model once (e.g., at application startup)
UMAP_MODEL = AnnEmbed::UMAP.load("models/text_umap_768_to_50.bin")

# Function to reduce a single embedding
def reduce_embedding(high_dim_embedding)
  # Input: 768-dimensional array
  # Output: 50-dimensional array
  UMAP_MODEL.transform([high_dim_embedding]).first
end

# Example usage
document = "Your text content here..."
high_dim_embedding = generate_embedding(document)  # Returns 768-dim vector
low_dim_embedding = reduce_embedding(high_dim_embedding)

# Store in database
save_to_database(
  document_id: 123,
  embedding: low_dim_embedding,
  original_embedding: high_dim_embedding  # Optionally keep original
)
```

### Step 3: Batch Processing

```ruby
# For better performance when processing multiple embeddings
def reduce_embeddings_batch(high_dim_embeddings)
  UMAP_MODEL.transform(high_dim_embeddings)
end

# Example: Process a batch of documents
documents = fetch_new_documents(limit: 100)
high_dim_embeddings = documents.map { |doc| generate_embedding(doc.text) }

# Reduce all at once (much faster than one-by-one)
low_dim_embeddings = reduce_embeddings_batch(high_dim_embeddings)

# Bulk insert to database
documents.zip(low_dim_embeddings).each do |doc, embedding|
  save_to_database(document_id: doc.id, embedding: embedding)
end
```

### Step 4: Caching and Recovery

```ruby
# Cache transformed results for recovery
embeddings = umap.fit_transform(training_data)
AnnEmbed::UMAP.export_data(embeddings, "backup/embeddings_#{Date.today}.json")

# Later, if needed
cached_embeddings = AnnEmbed::UMAP.import_data("backup/embeddings_2024-01-15.json")
```

## Real-World Example: Search System with Reduced Dimensions

```ruby
class EmbeddingService
  def initialize(model_path)
    @umap = AnnEmbed::UMAP.load(model_path)
    @embedder = TextEmbedder.new  # Your text embedding service
  end
  
  def process_document(text)
    # Generate high-dimensional embedding
    full_embedding = @embedder.embed(text)
    
    # Reduce dimensions for efficient storage
    reduced_embedding = @umap.transform([full_embedding]).first
    
    {
      full: full_embedding,      # 768 dimensions
      reduced: reduced_embedding  # 50 dimensions
    }
  end
  
  def search(query, documents)
    # Get query embedding
    query_full = @embedder.embed(query)
    query_reduced = @umap.transform([query_full]).first
    
    # Fast approximate search using reduced dimensions
    candidates = documents.sort_by do |doc|
      cosine_distance(query_reduced, doc[:reduced_embedding])
    end.first(20)
    
    # Rerank using full embeddings for precision
    candidates.sort_by do |doc|
      cosine_distance(query_full, doc[:full_embedding])
    end.first(5)
  end
end
```

## Performance Considerations

1. **Training Data Size**: UMAP works best with at least 100 samples
2. **Memory Usage**: Training on 10,000 768-dim vectors requires ~60MB RAM
3. **Training Time**: Expect 30-60 seconds for 10,000 vectors
4. **Transform Speed**: ~1ms per embedding, faster in batches
5. **Model Size**: Saved models are typically 5-20MB

## Best Practices

1. **Representative Training Data**: Use a diverse sample that represents your data distribution
2. **Validation**: Always validate that reduced embeddings maintain semantic relationships
3. **Batch Processing**: Transform multiple embeddings at once for better performance
4. **Model Versioning**: Keep track of model versions and training dates
5. **Error Handling**: Always check `fitted?` before calling `transform`

```ruby
# Good practice
if umap.fitted?
  result = umap.transform(data)
else
  raise "Model not fitted!"
end

# Batch processing
results = umap.transform(batch_of_100_items)  # Fast

# vs individual processing
results = items.map { |item| umap.transform([item]).first }  # Slower
```

## Working with Different Data Formats

```ruby
# Regular Ruby arrays
data = [[1.0, 2.0], [3.0, 4.0]]
embedding = umap.fit_transform(data)

# From CSV
require 'csv'
data = CSV.read('data.csv').map { |row| row.map(&:to_f) }
embedding = umap.fit_transform(data)

# From JSON
require 'json'
data = JSON.parse(File.read('data.json'))
embedding = umap.fit_transform(data)

# Export results
AnnEmbed::UMAP.export_data(embedding, 'results.json')
```

## Error Handling

The UMAP implementation provides clear error messages:

```ruby
# Empty data
umap.fit([])  # => ArgumentError: Input cannot be empty

# Inconsistent dimensions
umap.fit([[1, 2], [3, 4, 5]])  # => ArgumentError: All rows must have the same length

# Non-numeric data
umap.fit([["a", "b"]])  # => ArgumentError: Element at position [0, 0] is not numeric

# Unfitted model
umap = AnnEmbed::UMAP.new
umap.transform([[1, 2]])  # => RuntimeError: Model must be fitted before transform
```

## Troubleshooting

### "Model must be fitted before transform"
Make sure to call `fit` or `fit_transform` before calling `transform`:

```ruby
umap = AnnEmbed::UMAP.new
umap.fit(training_data)  # Required!
result = umap.transform(new_data)
```

### "assertion failed: (*f).abs() <= box_size"
This internal assertion can occur with data that has extreme values. Normalize your data:

```ruby
# Normalize to [0, 1] range
def normalize(data)
  data.map do |row|
    min = row.min
    max = row.max
    range = max - min
    row.map { |v| range.zero? ? 0.5 : (v - min) / range }
  end
end

normalized_data = normalize(your_data)
embedding = umap.fit_transform(normalized_data)
```

### Poor quality embeddings
- Increase `n_neighbors` for more global structure
- Decrease `n_neighbors` for more local structure
- Ensure you have enough training data (at least 100 samples)
- Check that your input data is meaningful (not random noise)

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests.

To install this gem onto your local machine, run `bundle exec rake install`.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yourusername/annembed-ruby.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments

This gem wraps the excellent [annembed](https://github.com/jean-pierreBoth/annembed) Rust crate by Jean-Pierre Both.