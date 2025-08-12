# frozen_string_literal: true

require_relative 'annembed_ruby'
require_relative 'svd'

module AnnEmbed
  # Principal Component Analysis using SVD
  # PCA is a linear dimensionality reduction technique that finds
  # the directions of maximum variance in the data
  class PCA
    attr_reader :n_components, :components, :explained_variance, :explained_variance_ratio, :mean

    # Initialize PCA
    # @param n_components [Integer] Number of principal components to keep
    def initialize(n_components: 2)
      @n_components = n_components
      @fitted = false
    end

    # Fit the PCA model
    # @param data [Array] 2D array of data points (n_samples × n_features)
    # @return [self] Returns self for method chaining
    def fit(data)
      validate_data(data)
      
      # Center the data (subtract mean from each feature)
      @mean = calculate_mean(data)
      centered_data = center_data(data, @mean)
      
      # Perform SVD on centered data
      # U contains the transformed data, S contains singular values, VT contains components
      u, s, vt = AnnEmbed.svd(centered_data, @n_components, n_iter: 5)
      
      # Store the principal components (eigenvectors)
      @components = vt  # Shape: (n_components, n_features)
      
      # Store singular values for consistency
      @singular_values = s
      
      # Calculate explained variance (eigenvalues)
      n_samples = data.size.to_f
      @explained_variance = s.map { |val| (val ** 2) / (n_samples - 1) }
      
      # Calculate explained variance ratio
      total_variance = calculate_total_variance(centered_data, n_samples)
      @explained_variance_ratio = @explained_variance.map { |var| var / total_variance }
      
      @fitted = true
      self
    end

    # Transform data using the fitted PCA model
    # @param data [Array] 2D array of data points
    # @return [Array] Transformed data in principal component space
    def transform(data)
      raise RuntimeError, "Model must be fitted before transform" unless fitted?
      validate_data(data)
      
      # Center the data using the stored mean
      centered_data = center_data(data, @mean)
      
      # Project onto principal components
      # Result = centered_data × components.T
      project_data(centered_data, @components)
    end

    # Fit the model and transform the data in one step
    # @param data [Array] 2D array of data points
    # @return [Array] Transformed data  
    def fit_transform(data)
      validate_data(data)
      
      # Center the data (subtract mean from each feature)
      @mean = calculate_mean(data)
      centered_data = center_data(data, @mean)
      
      # Perform SVD on centered data
      u, s, vt = AnnEmbed.svd(centered_data, @n_components, n_iter: 5)
      
      # Store the principal components (eigenvectors)
      @components = vt
      
      # Store singular values for later use
      @singular_values = s
      
      # Calculate explained variance (eigenvalues)
      n_samples = data.size.to_f
      @explained_variance = s.map { |val| (val ** 2) / (n_samples - 1) }
      
      # Calculate explained variance ratio
      total_variance = calculate_total_variance(centered_data, n_samples)
      @explained_variance_ratio = @explained_variance.map { |var| var / total_variance }
      
      @fitted = true
      
      # For PCA, the transformed data is U * S
      # Scale U by singular values
      transformed = []
      u.each do |row|
        scaled_row = row.each_with_index.map { |val, i| val * s[i] }
        transformed << scaled_row
      end
      transformed
    end

    # Inverse transform - reconstruct data from principal components
    # @param data [Array] Transformed data in PC space
    # @return [Array] Reconstructed data in original space
    def inverse_transform(data)
      raise RuntimeError, "Model must be fitted before inverse_transform" unless fitted?
      
      # Reconstruct: data × components + mean
      reconstructed = []
      data.each do |sample|
        reconstructed_sample = Array.new(@mean.size, 0.0)
        
        sample.each_with_index do |value, i|
          @components[i].each_with_index do |comp_val, j|
            reconstructed_sample[j] += value * comp_val
          end
        end
        
        # Add back the mean
        reconstructed_sample = reconstructed_sample.zip(@mean).map { |r, m| r + m }
        reconstructed << reconstructed_sample
      end
      
      reconstructed
    end

    # Get the amount of variance explained by each component
    # @return [Array] Explained variance for each component
    def explained_variance
      raise RuntimeError, "Model must be fitted first" unless fitted?
      @explained_variance
    end

    # Get the percentage of variance explained by each component
    # @return [Array] Explained variance ratio for each component
    def explained_variance_ratio
      raise RuntimeError, "Model must be fitted first" unless fitted?
      @explained_variance_ratio
    end

    # Get cumulative explained variance ratio
    # @return [Array] Cumulative sum of explained variance ratios
    def cumulative_explained_variance_ratio
      raise RuntimeError, "Model must be fitted first" unless fitted?
      
      cumsum = []
      sum = 0.0
      @explained_variance_ratio.each do |ratio|
        sum += ratio
        cumsum << sum
      end
      cumsum
    end

    # Check if model has been fitted
    # @return [Boolean] True if fitted
    def fitted?
      @fitted
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
      
      # Check we have enough samples for n_components
      if data.size < @n_components
        raise ArgumentError, "n_components (#{@n_components}) cannot be larger than n_samples (#{data.size})"
      end
      
      if data.first.size < @n_components
        raise ArgumentError, "n_components (#{@n_components}) cannot be larger than n_features (#{data.first.size})"
      end
    end

    def calculate_mean(data)
      n_features = data.first.size
      mean = Array.new(n_features, 0.0)
      
      data.each do |row|
        row.each_with_index do |val, i|
          mean[i] += val
        end
      end
      
      mean.map { |sum| sum / data.size.to_f }
    end

    def center_data(data, mean)
      data.map do |row|
        row.zip(mean).map { |val, m| val - m }
      end
    end

    def calculate_total_variance(centered_data, n_samples)
      total_var = 0.0
      
      centered_data.each do |row|
        row.each do |val|
          total_var += val ** 2
        end
      end
      
      total_var / (n_samples - 1)
    end

    def project_data(centered_data, components)
      # Matrix multiplication: centered_data × components.T
      transformed = []
      
      centered_data.each do |sample|
        projected = Array.new(@n_components, 0.0)
        
        components.each_with_index do |component, i|
          dot_product = 0.0
          sample.each_with_index do |val, j|
            dot_product += val * component[j]
          end
          projected[i] = dot_product
        end
        
        transformed << projected
      end
      
      transformed
    end
  end

  # Module-level convenience method
  # @param data [Array] 2D array of data points
  # @param n_components [Integer] Number of components
  # @return [Array] Transformed data
  def self.pca(data, n_components: 2)
    pca = PCA.new(n_components: n_components)
    pca.fit_transform(data)
  end
end