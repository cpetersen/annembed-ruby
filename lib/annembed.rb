# frozen_string_literal: true

require_relative "annembed/version"
require_relative "annembed/annembed_ruby"
require_relative "annembed/configuration"

# Main module for annembed-ruby gem
# Provides high-performance dimensionality reduction algorithms
module AnnEmbed
  class Error < StandardError; end

  # Available embedding methods
  METHODS = %i[umap tsne largevis diffusion].freeze

  # Core error classes
  class DimensionError < Error; end
  class ConvergenceError < Error; end
  class InvalidParameterError < Error; end

  # Autoload classes for better performance
  autoload :UMAP, "annembed/umap"
  autoload :Embedder, "annembed/embedder"
  autoload :Config, "annembed/config"
  autoload :Utils, "annembed/utils"
  autoload :Preprocessing, "annembed/preprocessing"
  autoload :Silence, "annembed/silence"
  
  # SVD, PCA and Clustering need special handling - require them after the extension is loaded
  require_relative "annembed/svd"
  require_relative "annembed/pca"
  require_relative "annembed/clustering"

  class << self
    # Quick UMAP embedding
    # @param data [Array] Input data (or Numo::NArray if available)
    # @param n_components [Integer] Number of dimensions in output
    # @return [Array] Embedded data (or Numo::NArray if Numo is loaded)
    def umap(data, n_components: 2, **options)
      embedder = Embedder.new(method: :umap, n_components: n_components, **options)
      embedder.fit_transform(data)
    end

    # Quick t-SNE embedding
    # @param data [Array, Numo::NArray] Input data
    # @param n_components [Integer] Number of dimensions in output
    # @return [Numo::NArray] Embedded data
    def tsne(data, n_components: 2, **options)
      embedder = Embedder.new(method: :tsne, n_components: n_components, **options)
      embedder.fit_transform(data)
    end

    # Estimate intrinsic dimension of data
    # @param data [Array, Numo::NArray] Input data
    # @param k [Integer] Number of neighbors to consider
    # @return [Float] Estimated intrinsic dimension
    def estimate_dimension(data, k: 10)
      Utils.estimate_intrinsic_dimension(data, k_neighbors: k)
    end

    # Perform randomized SVD
    # @param matrix [Array, Numo::NArray] Input matrix
    # @param k [Integer] Number of components
    # @return [Array<Numo::NArray>] U, S, V matrices
    def svd(matrix, k, n_iter: 2)
      SVD.randomized_svd(matrix, k, n_iter: n_iter)
    end
  end
end