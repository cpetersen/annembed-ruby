# frozen_string_literal: true

module ClusterKit
  # HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search
  #
  # @example Basic usage
  #   index = ClusterKit::HNSW.new(dim: 128, space: :euclidean)
  #   index.add_batch(vectors, labels: labels)
  #   neighbors = index.search(query_vector, k: 10)
  #
  # @example With metadata
  #   index = ClusterKit::HNSW.new(dim: 768, space: :cosine)
  #   index.add_item(vector, label: "doc_1", metadata: { title: "Introduction", date: "2024-01-01" })
  #   results = index.search_with_metadata(query, k: 5)
  #   # => [{ label: "doc_1", distance: 0.23, metadata: { title: "...", date: "..." } }, ...]
  class HNSW
    # Note: The actual HNSW class is defined in Rust (ext/clusterkit/src/hnsw.rs)
    # This Ruby file adds additional convenience methods and documentation.
    # The Rust implementation provides these core methods:
    #   - new(kwargs) - constructor
    #   - add_item(vector, kwargs) - add single item
    #   - add_batch(vectors, kwargs) - add multiple items
    #   - search(query, kwargs) - search for neighbors
    #   - search_with_metadata(query, kwargs) - search with metadata
    #   - size() - get number of items
    #   - config() - get configuration
    #   - stats() - get statistics
    #   - set_ef(ef) - set search quality parameter
    #   - save(path) - save to file
    
    # Initialize is actually handled by the Rust code
    # This documentation is for reference
    #
    # @param dim [Integer] Dimension of vectors (required)
    # @param space [Symbol] Distance metric: :euclidean, :cosine, or :inner_product (default: :euclidean)
    # @param max_elements [Integer] Maximum number of elements (default: 10_000)
    # @param m [Integer] Number of bi-directional links (default: 16)
    # @param ef_construction [Integer] Size of dynamic candidate list (default: 200)
    # @param random_seed [Integer, nil] Random seed for reproducible builds (default: nil)
    # @param dynamic_list [Boolean] Allow index to grow dynamically (not yet implemented)
    
    # Fit the index with training data (alias for add_batch)
    #
    # @param data [Array<Array>, Numo::NArray] Training vectors
    # @param labels [Array, nil] Optional labels for vectors
    # @return [self]
    def fit(data, labels: nil)
      add_batch(data, labels: labels)
      self
    end
    
    # Fit and return transformed data (for compatibility with sklearn-like interface)
    #
    # @param data [Array<Array>, Numo::NArray] Training vectors
    # @return [self]
    def fit_transform(data)
      fit(data)
      self
    end
    
    # Add a vector using the << operator
    #
    # @param vector [Array, Numo::NArray] Vector to add
    # @return [self]
    def <<(vector)
      add_item(vector, {})
      self
    end
    
    # Alias for search that always includes distances
    #
    # @param query [Array, Numo::NArray] Query vector
    # @param k [Integer] Number of neighbors
    # @param ef [Integer, nil] Search parameter (higher = better quality, slower)
    # @return [Array<Array>] Array of [indices, distances]
    def knn_query(query, k: 10, ef: nil)
      search(query, k: k, ef: ef, include_distances: true)
    end
    
    # Batch search for multiple queries
    #
    # @param queries [Array<Array>, Numo::NArray] Multiple query vectors
    # @param k [Integer] Number of neighbors per query
    # @param parallel [Boolean] Process queries in parallel
    # @return [Array<Array>] Results for each query
    def batch_search(queries, k: 10, parallel: true)
      queries = ensure_array(queries)
      
      if parallel && queries.size > 1
        require 'parallel'
        Parallel.map(queries) { |query| search(query, k: k) }
      else
        queries.map { |query| search(query, k: k) }
      end
    rescue LoadError
      # Parallel gem not available, fall back to sequential
      queries.map { |query| search(query, k: k) }
    end
    
    # Range search - find all points within a given radius
    #
    # @param query [Array, Numo::NArray] Query vector
    # @param radius [Float] Search radius
    # @param limit [Integer, nil] Maximum number of results
    # @return [Array<Hash>] Results within radius
    def range_search(query, radius:, limit: nil)
      # Get a large number of candidates
      k = limit || size
      k = [k, size].min
      
      results = search_with_metadata(query, k: k)
      
      # Filter by radius
      results.select { |r| r[:distance] <= radius }
             .take(limit || results.size)
    end
    
    # Check if index is empty
    # @return [Boolean]
    def empty?
      size == 0
    end
    
    # Clear all elements from the index
    #
    # @return [self]
    def clear!
      # Would need to recreate the index
      raise NotImplementedError, "Clear not yet implemented"
    end
    
    # Check if a label exists in the index
    #
    # @param label [String, Integer] Label to check
    # @return [Boolean]
    def include?(label)
      # This would need to be implemented in Rust
      # For now, return false
      false
    end
    
    # Get recall rate for a test set
    #
    # @param test_queries [Array<Array>] Query vectors
    # @param ground_truth [Array<Array>] True nearest neighbors for each query
    # @param k [Integer] Number of neighbors to evaluate
    # @return [Float] Recall rate (0.0 to 1.0)
    def recall(test_queries, ground_truth, k: 10)
      test_queries = ensure_array(test_queries)
      
      require 'set'
      total_correct = 0
      total_possible = 0
      
      test_queries.each_with_index do |query, i|
        predicted = Set.new(search(query, k: k))
        actual = Set.new(ground_truth[i].take(k))
        
        total_correct += (predicted & actual).size
        total_possible += [k, actual.size].min
      end
      
      total_possible > 0 ? total_correct.to_f / total_possible : 0.0
    end
    
    # Load an index from file
    #
    # @param path [String] File path to load from
    # @return [HNSW] New HNSW instance loaded from file
    def self.load(path)
      # This would need to be implemented in Rust
      raise NotImplementedError, "Loading from file not yet implemented"
    end
    
    # Create an index from embeddings produced by UMAP or other dimensionality reduction
    #
    # @param embeddings [Array<Array>, Numo::NArray] Embedding vectors
    # @param kwargs [Hash] Additional options for HNSW initialization
    # @return [HNSW] New HNSW instance
    def self.from_embedding(embeddings, **kwargs)
      embeddings = ensure_array(embeddings)
      
      dim = embeddings.first.size
      index = new(dim: dim, **kwargs)
      index.fit(embeddings)
      index
    end
    
    # Builder pattern for creating HNSW indices
    class Builder
      def initialize
        @config = {}
      end
      
      def space(type)
        @config[:space] = type
        self
      end
      
      def dimensions(dim)
        @config[:dim] = dim
        self
      end
      
      def max_elements(n)
        @config[:max_elements] = n
        self
      end
      
      def m_parameter(m)
        @config[:m] = m
        self
      end
      
      def ef_construction(ef)
        @config[:ef_construction] = ef
        self
      end
      
      def seed(seed)
        @config[:random_seed] = seed
        self
      end
      
      def build
        HNSW.new(**@config)
      end
    end
    
    private
    
    # Ensure input is a proper array format
    def ensure_array(data)
      case data
      when Array
        data
      else
        data.respond_to?(:to_a) ? data.to_a : raise(ArgumentError, "Data must be convertible to Array")
      end
    end
    
    # Class method to make it available to class methods
    def self.ensure_array(data)
      case data
      when Array
        data
      else
        data.respond_to?(:to_a) ? data.to_a : raise(ArgumentError, "Data must be convertible to Array")
      end
    end
  end
end