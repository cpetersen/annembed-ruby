#!/usr/bin/env ruby

require 'bundler/setup'
require 'annembed'

puts "PCA Example - Dimensionality Reduction and Variance Analysis"
puts "=" * 60

# Generate sample data with clear structure
# High variance in first 2 dimensions, low variance in others
def generate_structured_data(n_samples: 100, n_features: 20)
  data = []
  
  n_samples.times do
    point = []
    
    # First dimension: high variance (range ~10)
    point << rand * 10
    
    # Second dimension: medium variance (range ~5)
    point << rand * 5
    
    # Third dimension: some variance (range ~2)
    point << rand * 2
    
    # Remaining dimensions: very low variance (noise)
    (n_features - 3).times do
      point << rand * 0.1
    end
    
    data << point
  end
  
  data
end

# Generate data
data = generate_structured_data(n_samples: 100, n_features: 20)
puts "\nGenerated #{data.size} samples with #{data.first.size} features"

# Perform PCA with different numbers of components
[2, 3, 5, 10].each do |n_components|
  puts "\n" + "-" * 40
  puts "PCA with #{n_components} components:"
  
  pca = AnnEmbed::PCA.new(n_components: n_components)
  transformed = pca.fit_transform(data)
  
  puts "  Transformed shape: #{transformed.size} x #{transformed.first.size}"
  
  # Show explained variance for each component
  puts "  Explained variance ratio:"
  pca.explained_variance_ratio.each_with_index do |ratio, i|
    puts "    PC#{i+1}: #{(ratio * 100).round(2)}%"
  end
  
  # Show cumulative explained variance
  cumulative = pca.cumulative_explained_variance_ratio[-1]
  puts "  Total variance explained: #{(cumulative * 100).round(2)}%"
end

# Demonstrate reconstruction
puts "\n" + "=" * 60
puts "Reconstruction Example:"
puts "-" * 40

# Use 2 components (should capture most variance)
pca_2 = AnnEmbed::PCA.new(n_components: 2)
compressed = pca_2.fit_transform(data)
reconstructed = pca_2.inverse_transform(compressed)

# Calculate reconstruction error
sample_idx = 0
original = data[sample_idx]
recon = reconstructed[sample_idx]

puts "\nOriginal data point (first 5 features):"
puts "  #{original[0..4].map { |v| v.round(3) }.join(', ')}"

puts "\nReconstructed from 2 components (first 5 features):"
puts "  #{recon[0..4].map { |v| v.round(3) }.join(', ')}"

# Calculate mean squared error
mse = original.zip(recon).map { |o, r| (o - r) ** 2 }.sum / original.size
puts "\nReconstruction MSE: #{mse.round(4)}"

# Demonstrate data compression ratio
original_size = data.size * data.first.size
compressed_size = compressed.size * compressed.first.size
compression_ratio = (1 - compressed_size.to_f / original_size) * 100

puts "\nData Compression:"
puts "  Original size: #{original_size} values"
puts "  Compressed size: #{compressed_size} values"
puts "  Compression ratio: #{compression_ratio.round(1)}%"
puts "  Variance retained: #{(pca_2.cumulative_explained_variance_ratio[-1] * 100).round(1)}%"

# Compare with SVD
puts "\n" + "=" * 60
puts "PCA vs SVD Comparison:"
puts "-" * 40

# PCA (with mean centering)
pca = AnnEmbed::PCA.new(n_components: 2)
pca_result = pca.fit_transform(data)

# SVD (without mean centering)
u, s, vt = AnnEmbed.svd(data, 2)
svd_result = u

puts "PCA result (first point): #{pca_result[0].map { |v| v.round(3) }}"
puts "SVD result (first point): #{svd_result[0].map { |v| v.round(3) }}"
puts "\nNote: PCA centers the data (subtracts mean), SVD does not."
puts "This makes PCA better for finding principal components of variation."