# frozen_string_literal: true

require_relative 'annembed_ruby'

module AnnEmbed
  # The SVD module is defined by the Rust extension.
  # Here we only add Ruby convenience methods to the existing module.
  module SVD
    # Don't use class << self here as it can interfere with the Rust methods
    # Instead, define methods directly on the module
    
    # Perform randomized SVD
    # @param matrix [Array, Numo::NArray] Input matrix
    # @param k [Integer] Number of components
    # @param n_iter [Integer] Number of iterations
    # @return [Array<Numo::NArray>] U, S, V matrices
    def self.randomized_svd(matrix, k, n_iter: 2)
      raise ArgumentError, "Unsupported matrix type: #{matrix.class}" unless matrix.is_a?(Array)
      
      randomized_svd_rust(matrix, k, n_iter)
    end

    # Perform truncated SVD
    # @param matrix [Array, Numo::NArray] Input matrix
    # @param k [Integer] Number of components
    # @return [Array<Numo::NArray>] U, S, V matrices
    def self.truncated_svd(matrix, k)
      randomized_svd(matrix, k, n_iter: 2)
    end
  end
end