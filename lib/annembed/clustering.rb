# frozen_string_literal: true

require_relative 'annembed_ruby'

module AnnEmbed
  # Module for clustering algorithms
  module Clustering
    # K-means clustering algorithm
    class KMeans
      attr_reader :k, :max_iter, :centroids, :labels, :inertia

      # Initialize K-means clusterer
      # @param k [Integer] Number of clusters
      # @param max_iter [Integer] Maximum iterations (default: 300)
      # @param random_seed [Integer] Random seed for reproducibility (optional)
      def initialize(k:, max_iter: 300, random_seed: nil)
        raise ArgumentError, "k must be positive" unless k > 0
        @k = k
        @max_iter = max_iter
        @random_seed = random_seed
        @fitted = false
      end

      # Fit the K-means model
      # @param data [Array] 2D array of data points
      # @return [self] Returns self for method chaining
      def fit(data)
        validate_data(data)
        
        # Set random seed if provided
        srand(@random_seed) if @random_seed
        
        # Call Rust implementation
        @labels, @centroids, @inertia = Clustering.kmeans_rust(data, @k, @max_iter)
        @fitted = true
        
        self
      end

      # Predict cluster labels for new data
      # @param data [Array] 2D array of data points
      # @return [Array] Cluster labels
      def predict(data)
        raise RuntimeError, "Model must be fitted before predict" unless fitted?
        validate_data(data)
        
        Clustering.kmeans_predict_rust(data, @centroids)
      end

      # Fit the model and return labels
      # @param data [Array] 2D array of data points
      # @return [Array] Cluster labels
      def fit_predict(data)
        fit(data)
        @labels
      end

      # Check if model has been fitted
      # @return [Boolean] True if fitted
      def fitted?
        @fitted
      end

      # Get cluster centers
      # @return [Array] 2D array of cluster centers
      def cluster_centers
        @centroids
      end

      # Get the sum of squared distances of samples to their closest cluster center
      # @return [Float] Inertia value
      def inertia
        @inertia
      end

      private

      def validate_data(data)
        raise ArgumentError, "Data must be an array" unless data.is_a?(Array)
        raise ArgumentError, "Data cannot be empty" if data.empty?
        raise ArgumentError, "Data must be 2D array" unless data.first.is_a?(Array)
        
        # Check all rows have same length
        row_length = data.first.length
        unless data.all? { |row| row.is_a?(Array) && row.length == row_length }
          raise ArgumentError, "All rows must have the same length"
        end
        
        # Check all values are numeric
        data.each_with_index do |row, i|
          row.each_with_index do |val, j|
            unless val.is_a?(Numeric)
              raise ArgumentError, "Element at position [#{i}, #{j}] is not numeric"
            end
          end
        end
      end
    end

    # Convenience methods at module level
    class << self
      # Perform K-means clustering
      # @param data [Array] 2D array of data points
      # @param k [Integer] Number of clusters
      # @param max_iter [Integer] Maximum iterations
      # @return [Array<Array, Array, Float>] labels, centroids, inertia
      def kmeans(data, k, max_iter: 300)
        kmeans_rust(data, k, max_iter)
      end

      # Find optimal number of clusters using elbow method
      # @param data [Array] 2D array of data points
      # @param k_range [Range] Range of k values to try
      # @param max_iter [Integer] Maximum iterations per k
      # @return [Hash] Mapping of k to inertia values
      def elbow_method(data, k_range: 2..10, max_iter: 300)
        results = {}
        
        k_range.each do |k|
          _, _, inertia = kmeans(data, k, max_iter: max_iter)
          results[k] = inertia
        end
        
        results
      end

      # Detect optimal k from elbow method results
      # @param elbow_results [Hash] Mapping of k to inertia values (from elbow_method)
      # @param fallback_k [Integer] Default k to return if detection fails (default: 3)
      # @return [Integer] Optimal number of clusters
      def detect_optimal_k(elbow_results, fallback_k: 3)
        return fallback_k if elbow_results.nil? || elbow_results.empty?
        
        k_values = elbow_results.keys.sort
        return k_values.first if k_values.size == 1
        
        # Find the k with the largest drop in inertia
        max_drop = 0
        optimal_k = k_values.first
        
        k_values.each_cons(2) do |k1, k2|
          drop = elbow_results[k1] - elbow_results[k2]
          if drop > max_drop
            max_drop = drop
            optimal_k = k2  # Use k after the drop
          end
        end
        
        optimal_k
      end

      # Find optimal k and perform K-means clustering
      # @param data [Array] 2D array of data points
      # @param k_range [Range] Range of k values to try (default: 2..10)
      # @param max_iter [Integer] Maximum iterations (default: 300)
      # @return [Array<Integer, Array, Array, Float>] optimal_k, labels, centroids, inertia
      def optimal_kmeans(data, k_range: 2..10, max_iter: 300)
        elbow_results = elbow_method(data, k_range: k_range, max_iter: max_iter)
        optimal_k = detect_optimal_k(elbow_results)
        
        labels, centroids, inertia = kmeans(data, optimal_k, max_iter: max_iter)
        [optimal_k, labels, centroids, inertia]
      end

      # Simple silhouette score calculation
      # @param data [Array] 2D array of data points
      # @param labels [Array] Cluster labels
      # @return [Float] Mean silhouette coefficient
      def silhouette_score(data, labels)
        n_samples = data.size
        unique_labels = labels.uniq
        
        return 0.0 if unique_labels.size == 1
        
        silhouette_values = []
        
        data.each_with_index do |point, i|
          cluster_label = labels[i]
          
          # Calculate mean intra-cluster distance
          same_cluster_indices = labels.each_index.select { |j| labels[j] == cluster_label && j != i }
          if same_cluster_indices.empty?
            silhouette_values << 0.0
            next
          end
          
          a = same_cluster_indices.sum { |j| euclidean_distance(point, data[j]) } / same_cluster_indices.size.to_f
          
          # Calculate mean nearest-cluster distance
          b = Float::INFINITY
          unique_labels.each do |other_label|
            next if other_label == cluster_label
            
            other_cluster_indices = labels.each_index.select { |j| labels[j] == other_label }
            next if other_cluster_indices.empty?
            
            mean_dist = other_cluster_indices.sum { |j| euclidean_distance(point, data[j]) } / other_cluster_indices.size.to_f
            b = mean_dist if mean_dist < b
          end
          
          # Calculate silhouette value for this point
          if a == 0.0 && b == 0.0
            s = 0.0  # When all points are identical
          else
            s = (b - a) / [a, b].max
          end
          silhouette_values << s
        end
        
        silhouette_values.sum / silhouette_values.size.to_f
      end

      private

      def euclidean_distance(a, b)
        Math.sqrt(a.zip(b).sum { |x, y| (x - y) ** 2 })
      end
    end
  end
end