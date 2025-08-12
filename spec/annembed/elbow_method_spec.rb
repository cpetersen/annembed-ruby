require 'spec_helper'

RSpec.describe AnnEmbed::Clustering do
  describe '.elbow_method' do
    context 'with clear clusters' do
      let(:data) do
        # Generate 3 clear clusters
        result = []
        3.times do |cluster_id|
          center = Array.new(10) { (rand - 0.5) * 0.3 + cluster_id * 0.3 }
          20.times do
            point = center.map { |c| c + (rand - 0.5) * 0.05 }
            result << point
          end
        end
        result
      end

      it 'returns inertia values for each k' do
        results = described_class.elbow_method(data, k_range: 2..5)
        
        expect(results).to be_a(Hash)
        expect(results.keys).to contain_exactly(2, 3, 4, 5)
        
        results.each do |k, inertia|
          expect(inertia).to be_a(Float)
          expect(inertia).to be >= 0
        end
      end

      it 'shows decreasing inertia as k increases' do
        results = described_class.elbow_method(data, k_range: 2..5)
        
        sorted_results = results.sort.map(&:last)
        expect(sorted_results).to eq(sorted_results.sort.reverse)
      end

      it 'shows a clear elbow at k=3' do
        results = described_class.elbow_method(data, k_range: 2..6)
        
        # Calculate drops between consecutive k values
        drops = {}
        results.sort.each_cons(2) do |(k1, v1), (k2, v2)|
          drops[k2] = v1 - v2
        end
        
        # The biggest drop should be at k=3
        max_drop_k = drops.max_by { |_, v| v }&.first
        expect(max_drop_k).to eq(3)
      end
    end

    context 'with single cluster' do
      let(:data) do
        # Generate one cluster
        center = Array.new(10) { rand }
        Array.new(30) do
          center.map { |c| c + (rand - 0.5) * 0.1 }
        end
      end

      it 'shows minimal improvement after k=2' do
        results = described_class.elbow_method(data, k_range: 2..4)
        
        # Inertia should decrease as k increases
        expect(results[2]).to be > results[3] if results[2] && results[3]
        expect(results[3]).to be > results[4] if results[3] && results[4]
        
        # For a single cluster, improvements should be small
        drop_2_to_3 = results[2] - results[3] if results[2] && results[3]
        drop_3_to_4 = results[3] - results[4] if results[3] && results[4]
        
        # Drops should be relatively small for single cluster data
        if drop_2_to_3 && drop_3_to_4
          # Both drops should be small (not testing relative size as it's not guaranteed)
          expect(drop_2_to_3).to be < results[2] * 0.5  # Less than 50% drop
          expect(drop_3_to_4).to be < results[3] * 0.5  # Less than 50% drop
        end
      end
    end

    context 'with identical points' do
      let(:data) do
        # All points are identical
        point = Array.new(5) { rand }
        Array.new(20) { point.dup }
      end

      it 'returns zero or near-zero inertia' do
        results = described_class.elbow_method(data, k_range: 2..4)
        
        results.each do |k, inertia|
          expect(inertia).to be < 0.0001
        end
      end
    end

    context 'parameter validation' do
      let(:data) { [[1, 2], [3, 4], [5, 6]] }

      it 'accepts a range for k_range' do
        expect { described_class.elbow_method(data, k_range: 2..3) }.not_to raise_error
      end

      it 'accepts an array for k_range' do
        expect { described_class.elbow_method(data, k_range: [2, 3]) }.not_to raise_error
      end

      it 'raises error for invalid k values' do
        # k=0 should cause an error
        expect { described_class.elbow_method(data, k_range: [0, 2]) }.to raise_error
        # Negative k should cause an error  
        expect { described_class.elbow_method(data, k_range: [-1, 2]) }.to raise_error
      end

      it 'raises error when k exceeds number of samples' do
        expect { described_class.elbow_method(data, k_range: 2..10) }.to raise_error(ArgumentError)
      end
    end

    context 'with custom parameters' do
      let(:data) do
        Array.new(50) { Array.new(10) { rand } }
      end

      it 'respects max_iter parameter' do
        results_low_iter = described_class.elbow_method(data, k_range: 2..3, max_iter: 1)
        results_high_iter = described_class.elbow_method(data, k_range: 2..3, max_iter: 100)
        
        # Both should complete without error
        expect(results_low_iter.keys).to contain_exactly(2, 3)
        expect(results_high_iter.keys).to contain_exactly(2, 3)
      end

      it 'produces consistent results for the same data' do
        # Run elbow method multiple times
        results1 = described_class.elbow_method(data, k_range: 2..4)
        results2 = described_class.elbow_method(data, k_range: 2..4)
        
        # Results should have the same keys
        expect(results1.keys).to eq(results2.keys)
        
        # Results might vary slightly due to random initialization,
        # but should be in the same ballpark
        results1.each do |k, inertia|
          expect(inertia).to be_within(inertia * 0.3).of(results2[k])
        end
      end
    end
  end

  describe '.detect_optimal_k' do
    it 'detects clear elbow at k=3' do
      results = {2 => 1000.0, 3 => 100.0, 4 => 90.0, 5 => 85.0}
      expect(described_class.detect_optimal_k(results)).to eq(3)
    end

    it 'detects elbow at k=4' do
      results = {2 => 500.0, 3 => 400.0, 4 => 100.0, 5 => 95.0}
      expect(described_class.detect_optimal_k(results)).to eq(4)
    end

    it 'handles minimal drops' do
      results = {2 => 100.0, 3 => 95.0, 4 => 93.0, 5 => 92.0}
      # With small drops, it picks the first biggest drop (k=3)
      expect(described_class.detect_optimal_k(results)).to eq(3)
    end

    it 'handles gradual decline without clear elbow' do
      results = {2 => 100.0, 3 => 80.0, 4 => 60.0, 5 => 40.0}
      # Should pick k=3 as it has the first equal drop
      expect(described_class.detect_optimal_k(results)).to eq(3)
    end

    it 'handles empty results with default fallback' do
      expect(described_class.detect_optimal_k({})).to eq(3)  # Default fallback
    end
    
    it 'handles empty results with custom fallback' do
      expect(described_class.detect_optimal_k({}, fallback_k: 5)).to eq(5)
    end

    it 'handles nil input' do
      expect(described_class.detect_optimal_k(nil)).to eq(3)
    end

    it 'handles single k value' do
      results = {3 => 100.0}
      expect(described_class.detect_optimal_k(results)).to eq(3)
    end

    it 'handles unsorted keys' do
      results = {5 => 85.0, 2 => 1000.0, 4 => 90.0, 3 => 100.0}
      expect(described_class.detect_optimal_k(results)).to eq(3)
    end
  end

  describe '.optimal_kmeans' do
    let(:data) do
      # Generate 3 clear clusters  
      result = []
      3.times do |cluster_id|
        center = Array.new(10) { (rand - 0.5) * 0.3 + cluster_id * 0.3 }
        20.times do
          point = center.map { |c| c + (rand - 0.5) * 0.05 }
          result << point
        end
      end
      result
    end

    it 'returns optimal k, labels, centroids, and inertia' do
      optimal_k, labels, centroids, inertia = described_class.optimal_kmeans(data, k_range: 2..5)
      
      expect(optimal_k).to be_a(Integer)
      expect(optimal_k).to be_between(2, 5)
      
      expect(labels).to be_a(Array)
      expect(labels.size).to eq(data.size)
      expect(labels.uniq.size).to eq(optimal_k)
      
      expect(centroids).to be_a(Array)
      expect(centroids.size).to eq(optimal_k)
      expect(centroids.first.size).to eq(data.first.size)
      
      expect(inertia).to be_a(Float)
      expect(inertia).to be >= 0
    end

    it 'finds approximately 3 clusters for 3-cluster data' do
      optimal_k, _, _, _ = described_class.optimal_kmeans(data, k_range: 2..6)
      # Should detect 3 clusters (or close to it)
      expect(optimal_k).to be_between(2, 4)
    end
  end
end