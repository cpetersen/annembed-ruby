# frozen_string_literal: true

require 'spec_helper'
require 'annembed/clustering'

RSpec.describe AnnEmbed::Clustering do
  describe AnnEmbed::Clustering::KMeans do
    let(:simple_data) {
      # Create 3 clear clusters
      cluster1 = 5.times.map { [rand + 0, rand + 0] }
      cluster2 = 5.times.map { [rand + 5, rand + 5] }
      cluster3 = 5.times.map { [rand + 10, rand + 10] }
      cluster1 + cluster2 + cluster3
    }
    
    describe '#initialize' do
      it 'creates a new KMeans instance' do
        kmeans = described_class.new(k: 3)
        expect(kmeans).to be_a(described_class)
        expect(kmeans.k).to eq(3)
        expect(kmeans.max_iter).to eq(300)
      end
      
      it 'accepts custom parameters' do
        kmeans = described_class.new(k: 5, max_iter: 100, random_seed: 42)
        expect(kmeans.k).to eq(5)
        expect(kmeans.max_iter).to eq(100)
      end
    end
    
    describe '#fit' do
      let(:kmeans) { described_class.new(k: 3, random_seed: 42) }
      
      it 'fits the model to data' do
        result = kmeans.fit(simple_data)
        expect(result).to eq(kmeans)
        expect(kmeans).to be_fitted
      end
      
      it 'sets labels and centroids' do
        kmeans.fit(simple_data)
        expect(kmeans.labels).to be_a(Array)
        expect(kmeans.labels.size).to eq(simple_data.size)
        expect(kmeans.centroids).to be_a(Array)
        expect(kmeans.centroids.size).to eq(3)
      end
      
      it 'calculates inertia' do
        kmeans.fit(simple_data)
        expect(kmeans.inertia).to be_a(Float)
        expect(kmeans.inertia).to be > 0
      end
      
      it 'raises error for empty data' do
        expect {
          kmeans.fit([])
        }.to raise_error(ArgumentError, /Data cannot be empty/)
      end
      
      it 'raises error for non-array data' do
        expect {
          kmeans.fit("not an array")
        }.to raise_error(ArgumentError, /Data must be an array/)
      end
      
      it 'raises error for 1D array' do
        expect {
          kmeans.fit([1, 2, 3])
        }.to raise_error(ArgumentError, /Data must be 2D array/)
      end
      
      it 'raises error for inconsistent row lengths' do
        bad_data = [[1, 2], [3, 4, 5]]
        expect {
          kmeans.fit(bad_data)
        }.to raise_error(ArgumentError, /All rows must have the same length/)
      end
      
      it 'raises error for non-numeric data' do
        bad_data = [[1, "two"], [3, 4]]
        expect {
          kmeans.fit(bad_data)
        }.to raise_error(ArgumentError, /is not numeric/)
      end
    end
    
    describe '#predict' do
      let(:kmeans) { described_class.new(k: 3, random_seed: 42) }
      let(:new_data) { [[0.5, 0.5], [5.5, 5.5], [10.5, 10.5]] }
      
      it 'predicts cluster labels for new data' do
        kmeans.fit(simple_data)
        labels = kmeans.predict(new_data)
        expect(labels).to be_a(Array)
        expect(labels.size).to eq(new_data.size)
        expect(labels.uniq.size).to be <= 3
      end
      
      it 'raises error if not fitted' do
        expect {
          kmeans.predict(new_data)
        }.to raise_error(RuntimeError, /Model must be fitted before predict/)
      end
    end
    
    describe '#fit_predict' do
      let(:kmeans) { described_class.new(k: 3, random_seed: 42) }
      
      it 'fits and returns labels in one step' do
        labels = kmeans.fit_predict(simple_data)
        expect(labels).to be_a(Array)
        expect(labels.size).to eq(simple_data.size)
        expect(kmeans).to be_fitted
      end
    end
    
    describe '#fitted?' do
      let(:kmeans) { described_class.new(k: 3) }
      
      it 'returns false before fitting' do
        expect(kmeans).not_to be_fitted
      end
      
      it 'returns true after fitting' do
        kmeans.fit(simple_data)
        expect(kmeans).to be_fitted
      end
    end
    
    describe '#cluster_centers' do
      let(:kmeans) { described_class.new(k: 3, random_seed: 42) }
      
      it 'returns cluster centers after fitting' do
        kmeans.fit(simple_data)
        centers = kmeans.cluster_centers
        expect(centers).to be_a(Array)
        expect(centers.size).to eq(3)
        expect(centers.first).to be_a(Array)
        expect(centers.first.size).to eq(2)
      end
    end
  end
  
  describe 'module methods' do
    let(:data) {
      # Create 2 clear clusters
      cluster1 = 10.times.map { [rand * 2, rand * 2] }
      cluster2 = 10.times.map { [rand * 2 + 5, rand * 2 + 5] }
      cluster1 + cluster2
    }
    
    describe '.kmeans' do
      it 'performs k-means clustering' do
        labels, centroids, inertia = described_class.kmeans(data, 2)
        expect(labels).to be_a(Array)
        expect(labels.size).to eq(data.size)
        expect(centroids).to be_a(Array)
        expect(centroids.size).to eq(2)
        expect(inertia).to be_a(Float)
      end
    end
    
    describe '.elbow_method' do
      it 'calculates inertia for different k values' do
        results = described_class.elbow_method(data, k_range: 2..4)
        expect(results).to be_a(Hash)
        expect(results.keys).to eq([2, 3, 4])
        expect(results.values).to all(be_a(Float))
        
        # Inertia should generally decrease as k increases
        expect(results[2]).to be >= results[3]
        expect(results[3]).to be >= results[4]
      end
    end
    
    describe '.silhouette_score' do
      it 'calculates silhouette score' do
        labels, _, _ = described_class.kmeans(data, 2)
        score = described_class.silhouette_score(data, labels)
        expect(score).to be_a(Float)
        expect(score).to be_between(-1, 1)
      end
      
      it 'returns 0 for single cluster' do
        labels = [0] * data.size
        score = described_class.silhouette_score(data, labels)
        expect(score).to eq(0.0)
      end
    end
  end
  
  describe 'integration with embeddings' do
    it 'works with UMAP reduced data' do
      # Generate high-dimensional data
      high_dim_data = 30.times.map { 10.times.map { rand } }
      
      # Reduce dimensions
      umap = AnnEmbed::Embedder.new(method: :umap, n_components: 2)
      reduced = umap.fit_transform(high_dim_data)
      
      # Cluster reduced data
      kmeans = AnnEmbed::Clustering::KMeans.new(k: 3)
      labels = kmeans.fit_predict(reduced)
      
      expect(labels).to be_a(Array)
      expect(labels.size).to eq(30)
      expect(labels.uniq.size).to be <= 3
    end
    
    it 'works with t-SNE reduced data' do
      # Generate high-dimensional data
      high_dim_data = 30.times.map { 10.times.map { rand } }
      
      # Reduce dimensions
      tsne = AnnEmbed::Embedder.new(method: :tsne, n_components: 2)
      reduced = tsne.fit_transform(high_dim_data)
      
      # Cluster reduced data
      kmeans = AnnEmbed::Clustering::KMeans.new(k: 3)
      labels = kmeans.fit_predict(reduced)
      
      expect(labels).to be_a(Array)
      expect(labels.size).to eq(30)
      expect(labels.uniq.size).to be <= 3
    end
    
    it 'works with SVD reduced data' do
      # Generate high-dimensional data
      high_dim_data = 30.times.map { 10.times.map { rand } }
      
      # Reduce dimensions with SVD
      u, s, vt = AnnEmbed.svd(high_dim_data, 2)
      
      # Cluster reduced data
      kmeans = AnnEmbed::Clustering::KMeans.new(k: 3)
      labels = kmeans.fit_predict(u)
      
      expect(labels).to be_a(Array)
      expect(labels.size).to eq(30)
      expect(labels.uniq.size).to be <= 3
    end
  end
  
  describe 'edge cases' do
    it 'handles k=1 (single cluster)' do
      data = 10.times.map { [rand, rand] }
      kmeans = AnnEmbed::Clustering::KMeans.new(k: 1)
      labels = kmeans.fit_predict(data)
      
      expect(labels.uniq.size).to eq(1)
      expect(labels).to all(eq(0))
    end
    
    it 'handles k equal to number of points' do
      data = 5.times.map { [rand * 10, rand * 10] }
      kmeans = AnnEmbed::Clustering::KMeans.new(k: 5)
      labels = kmeans.fit_predict(data)
      
      expect(labels.uniq.size).to be <= 5
    end
    
    it 'handles identical points' do
      data = 10.times.map { [1.0, 2.0] }  # All points are the same
      kmeans = AnnEmbed::Clustering::KMeans.new(k: 3)
      labels = kmeans.fit_predict(data)
      
      expect(labels).to be_a(Array)
      expect(labels.size).to eq(10)
      # All identical points might end up in any cluster
    end
  end
end