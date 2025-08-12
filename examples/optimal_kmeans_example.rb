#!/usr/bin/env ruby

require 'bundler/setup'
require 'annembed'

# Generate sample data with 3 natural clusters
def generate_sample_data
  data = []
  
  # Cluster 1: centered around [0, 0]
  30.times do
    data << [rand * 0.5 - 0.25, rand * 0.5 - 0.25]
  end
  
  # Cluster 2: centered around [3, 3]
  30.times do
    data << [3 + rand * 0.5 - 0.25, 3 + rand * 0.5 - 0.25]
  end
  
  # Cluster 3: centered around [1.5, -2]
  30.times do
    data << [1.5 + rand * 0.5 - 0.25, -2 + rand * 0.5 - 0.25]
  end
  
  data
end

puts "AnnEmbed Optimal K-means Clustering Example"
puts "=" * 50

# Generate data
data = generate_sample_data
puts "\nGenerated #{data.size} data points with 3 natural clusters"

# Method 1: Manual elbow method and detection
puts "\nMethod 1: Manual elbow method"
puts "-" * 30

elbow_results = AnnEmbed::Clustering.elbow_method(data, k_range: 2..8)
puts "Elbow method results:"
elbow_results.sort.each do |k, inertia|
  puts "  k=#{k}: inertia=#{inertia.round(2)}"
end

optimal_k = AnnEmbed::Clustering.detect_optimal_k(elbow_results)
puts "\nDetected optimal k: #{optimal_k}"

# Perform K-means with optimal k
labels, centroids, inertia = AnnEmbed::Clustering.kmeans(data, optimal_k)
puts "Final inertia: #{inertia.round(2)}"
puts "Cluster sizes: #{labels.tally.sort.to_h}"

# Method 2: Using optimal_kmeans (all-in-one)
puts "\nMethod 2: Using optimal_kmeans (automatic)"
puts "-" * 30

optimal_k, labels, centroids, inertia = AnnEmbed::Clustering.optimal_kmeans(data, k_range: 2..8)
puts "Automatically detected k: #{optimal_k}"
puts "Final inertia: #{inertia.round(2)}"
puts "Cluster sizes: #{labels.tally.sort.to_h}"

# Method 3: Using KMeans class with detected k
puts "\nMethod 3: Using KMeans class"
puts "-" * 30

# First detect optimal k
elbow_results = AnnEmbed::Clustering.elbow_method(data, k_range: 2..8)
optimal_k = AnnEmbed::Clustering.detect_optimal_k(elbow_results)

# Create KMeans instance with optimal k
kmeans = AnnEmbed::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
labels = kmeans.fit_predict(data)

puts "K-means with k=#{optimal_k}:"
puts "Inertia: #{kmeans.inertia.round(2)}"
puts "Cluster sizes: #{labels.tally.sort.to_h}"

# Show cluster centers
puts "\nCluster centers:"
kmeans.cluster_centers.each_with_index do |center, i|
  puts "  Cluster #{i}: [#{center[0].round(2)}, #{center[1].round(2)}]"
end

# Calculate silhouette score to validate clustering quality
silhouette = AnnEmbed::Clustering.silhouette_score(data, labels)
puts "\nSilhouette score: #{silhouette.round(3)}"
puts "(Higher is better, range is -1 to 1)"

# Custom fallback example
puts "\nCustom fallback example:"
puts "-" * 30
empty_results = {}
default_k = AnnEmbed::Clustering.detect_optimal_k(empty_results)
custom_k = AnnEmbed::Clustering.detect_optimal_k(empty_results, fallback_k: 5)
puts "Default fallback k: #{default_k}"
puts "Custom fallback k: #{custom_k}"