require 'spec_helper'

RSpec.describe AnnEmbed::PCA do
  describe '#initialize' do
    it 'creates a PCA instance with default parameters' do
      pca = described_class.new
      expect(pca.n_components).to eq(2)
      expect(pca).not_to be_fitted
    end

    it 'accepts custom n_components' do
      pca = described_class.new(n_components: 5)
      expect(pca.n_components).to eq(5)
    end
  end

  describe '#fit' do
    let(:data) do
      # Create data with clear principal components
      # 3 features, first two have high variance, third has low variance
      Array.new(50) do
        [rand * 10, rand * 10, rand * 0.1]
      end
    end

    it 'fits the model to data' do
      pca = described_class.new(n_components: 2)
      result = pca.fit(data)
      
      expect(result).to eq(pca)  # Returns self
      expect(pca).to be_fitted
    end

    it 'calculates mean correctly' do
      pca = described_class.new(n_components: 2)
      pca.fit(data)
      
      expect(pca.mean).to be_a(Array)
      expect(pca.mean.size).to eq(3)  # 3 features
      
      # Mean should be around [5, 5, 0.05]
      expect(pca.mean[0]).to be_within(1).of(5)
      expect(pca.mean[1]).to be_within(1).of(5)
      expect(pca.mean[2]).to be_within(0.1).of(0.05)
    end

    it 'extracts principal components' do
      pca = described_class.new(n_components: 2)
      pca.fit(data)
      
      expect(pca.components).to be_a(Array)
      expect(pca.components.size).to eq(2)  # 2 components
      expect(pca.components.first.size).to eq(3)  # 3 features
    end

    it 'calculates explained variance' do
      pca = described_class.new(n_components: 2)
      pca.fit(data)
      
      expect(pca.explained_variance).to be_a(Array)
      expect(pca.explained_variance.size).to eq(2)
      
      # Variance should be positive
      pca.explained_variance.each do |var|
        expect(var).to be > 0
      end
      
      # First component should explain more variance than second
      expect(pca.explained_variance[0]).to be > pca.explained_variance[1]
    end

    it 'calculates explained variance ratio' do
      pca = described_class.new(n_components: 2)
      pca.fit(data)
      
      expect(pca.explained_variance_ratio).to be_a(Array)
      expect(pca.explained_variance_ratio.size).to eq(2)
      
      # Ratios should be between 0 and 1
      pca.explained_variance_ratio.each do |ratio|
        expect(ratio).to be_between(0, 1)
      end
      
      # Sum should be <= 1 (since we're only keeping 2 of 3 components)
      sum = pca.explained_variance_ratio.sum
      expect(sum).to be <= 1.0
    end

    it 'raises error with invalid data' do
      expect { described_class.new.fit(nil) }.to raise_error(ArgumentError)
      expect { described_class.new.fit([]) }.to raise_error(ArgumentError)
      expect { described_class.new.fit([1, 2, 3]) }.to raise_error(ArgumentError)
    end

    it 'raises error when n_components > n_samples' do
      small_data = [[1, 2], [3, 4]]
      pca = described_class.new(n_components: 3)
      expect { pca.fit(small_data) }.to raise_error(ArgumentError, /n_components.*cannot be larger than n_samples/)
    end

    it 'raises error when n_components > n_features' do
      data = [[1, 2], [3, 4], [5, 6]]
      pca = described_class.new(n_components: 3)
      expect { pca.fit(data) }.to raise_error(ArgumentError, /n_components.*cannot be larger than n_features/)
    end
  end

  describe '#transform' do
    let(:data) do
      Array.new(30) { [rand * 10, rand * 10, rand * 0.1] }
    end

    let(:fitted_pca) do
      pca = described_class.new(n_components: 2)
      pca.fit(data)
      pca
    end

    it 'transforms data to principal component space' do
      transformed = fitted_pca.transform(data)
      
      expect(transformed).to be_a(Array)
      expect(transformed.size).to eq(data.size)
      expect(transformed.first.size).to eq(2)  # 2 components
    end

    it 'produces centered transformed data' do
      transformed = fitted_pca.transform(data)
      
      # Mean of transformed data should be close to 0
      mean_pc1 = transformed.map { |t| t[0] }.sum / transformed.size.to_f
      mean_pc2 = transformed.map { |t| t[1] }.sum / transformed.size.to_f
      
      expect(mean_pc1.abs).to be < 1.0
      expect(mean_pc2.abs).to be < 1.0
    end

    it 'raises error if not fitted' do
      pca = described_class.new
      expect { pca.transform(data) }.to raise_error(RuntimeError, /must be fitted/)
    end
  end

  describe '#fit_transform' do
    let(:data) do
      Array.new(30) { [rand * 10, rand * 10, rand * 0.1] }
    end

    it 'fits and transforms in one step' do
      pca = described_class.new(n_components: 2)
      transformed = pca.fit_transform(data)
      
      expect(pca).to be_fitted
      expect(transformed).to be_a(Array)
      expect(transformed.size).to eq(data.size)
      expect(transformed.first.size).to eq(2)
    end

    it 'produces same result as fit then transform' do
      pca1 = described_class.new(n_components: 2)
      result1 = pca1.fit_transform(data)
      
      pca2 = described_class.new(n_components: 2)
      pca2.fit(data)
      result2 = pca2.transform(data)
      
      # Results should be very similar (might have small numerical differences)
      # Note: Signs might be flipped for eigenvectors, so we check absolute values
      result1.zip(result2).each do |r1, r2|
        r1.zip(r2).each do |v1, v2|
          # Check if values are close (allowing for sign flip)
          diff1 = (v1 - v2).abs
          diff2 = (v1 + v2).abs
          expect([diff1, diff2].min).to be < 0.1
        end
      end
    end
  end

  describe '#inverse_transform' do
    let(:data) do
      # Create data with known structure
      Array.new(20) do
        [rand * 10, rand * 10, rand * 0.1]
      end
    end

    it 'reconstructs data from principal components' do
      pca = described_class.new(n_components: 2)
      transformed = pca.fit_transform(data)
      reconstructed = pca.inverse_transform(transformed)
      
      expect(reconstructed).to be_a(Array)
      expect(reconstructed.size).to eq(data.size)
      expect(reconstructed.first.size).to eq(3)  # Original features
    end

    it 'produces approximate reconstruction' do
      pca = described_class.new(n_components: 2)
      transformed = pca.fit_transform(data)
      reconstructed = pca.inverse_transform(transformed)
      
      # Calculate reconstruction error
      total_error = 0.0
      data.zip(reconstructed).each do |original, recon|
        original.zip(recon).each do |o, r|
          total_error += (o - r) ** 2
        end
      end
      avg_error = Math.sqrt(total_error / (data.size * data.first.size))
      
      # Average reconstruction error should be reasonable
      # (We lose information by using only 2 of 3 components)
      expect(avg_error).to be < 3.0
    end

    it 'raises error if not fitted' do
      pca = described_class.new
      expect { pca.inverse_transform([[1, 2]]) }.to raise_error(RuntimeError, /must be fitted/)
    end
  end

  describe '#cumulative_explained_variance_ratio' do
    let(:data) do
      Array.new(30) { [rand * 10, rand * 10, rand * 0.1] }
    end

    it 'returns cumulative sum of explained variance ratios' do
      pca = described_class.new(n_components: 2)
      pca.fit(data)
      
      cumsum = pca.cumulative_explained_variance_ratio
      
      expect(cumsum).to be_a(Array)
      expect(cumsum.size).to eq(2)
      
      # Should be increasing
      expect(cumsum[1]).to be > cumsum[0]
      
      # First value should equal first explained variance ratio
      expect(cumsum[0]).to eq(pca.explained_variance_ratio[0])
      
      # Last value should equal sum of all ratios
      expect(cumsum[-1]).to be_within(0.0001).of(pca.explained_variance_ratio.sum)
    end

    it 'raises error if not fitted' do
      pca = described_class.new
      expect { pca.cumulative_explained_variance_ratio }.to raise_error(RuntimeError, /must be fitted/)
    end
  end

  describe 'preserving variance structure' do
    it 'identifies high variance directions' do
      # Create data where variance is much higher in first dimension
      data = Array.new(50) do
        [rand * 100, rand * 10, rand * 1]
      end
      
      pca = described_class.new(n_components: 2)
      pca.fit(data)
      
      # First PC should explain most variance
      expect(pca.explained_variance_ratio[0]).to be > 0.8
      
      # First component should align mostly with first feature
      first_component = pca.components[0]
      expect(first_component[0].abs).to be > first_component[1].abs
      expect(first_component[0].abs).to be > first_component[2].abs
    end
  end

  describe 'module-level convenience method' do
    let(:data) do
      Array.new(20) { [rand * 10, rand * 10, rand * 0.1] }
    end

    it 'performs PCA with default parameters' do
      result = AnnEmbed.pca(data)
      
      expect(result).to be_a(Array)
      expect(result.size).to eq(data.size)
      expect(result.first.size).to eq(2)  # Default 2 components
    end

    it 'accepts n_components parameter' do
      result = AnnEmbed.pca(data, n_components: 3)
      
      expect(result.first.size).to eq(3)
    end
  end

  describe 'with known dataset' do
    it 'correctly reduces dimensions of iris-like data' do
      # Create iris-like data with 4 features
      data = []
      
      # Class 1: small values
      20.times do
        data << [5.0 + rand * 0.5, 3.4 + rand * 0.4, 1.5 + rand * 0.3, 0.2 + rand * 0.1]
      end
      
      # Class 2: medium values
      20.times do
        data << [5.9 + rand * 0.5, 2.8 + rand * 0.4, 4.3 + rand * 0.3, 1.3 + rand * 0.1]
      end
      
      # Class 3: large values
      20.times do
        data << [6.5 + rand * 0.5, 3.0 + rand * 0.4, 5.5 + rand * 0.3, 2.0 + rand * 0.1]
      end
      
      pca = described_class.new(n_components: 2)
      transformed = pca.fit_transform(data)
      
      # Should preserve most variance in 2D
      cumsum = pca.cumulative_explained_variance_ratio
      expect(cumsum[-1]).to be > 0.9  # Should explain >90% variance with 2 PCs
      
      # Classes should be separable in PC space
      # (We can't test exact positions but we can test that variance is preserved)
      pc1_values = transformed.map { |t| t[0] }
      pc1_variance = calculate_variance(pc1_values)
      expect(pc1_variance).to be > 0  # Should have variance along PC1
    end
  end

  # Helper method for tests
  def calculate_variance(values)
    mean = values.sum / values.size.to_f
    sum_sq_diff = values.map { |v| (v - mean) ** 2 }.sum
    sum_sq_diff / (values.size - 1)
  end
end