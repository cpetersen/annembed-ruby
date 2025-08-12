namespace :annembed do
  desc "Generate interactive visualization comparing dimensionality reduction and clustering methods"
  task :visualize, [:output_file, :dataset] do |t, args|
    require 'bundler/setup'
    require 'annembed'
    require 'json'
    
    output_file = args[:output_file] || 'annembed_visualization.html'
    dataset_type = args[:dataset] || 'clusters'
    
    puts "Generating visualization with dataset: #{dataset_type}"
    
    # Generate dataset based on type
    data, true_labels, dataset_name = case dataset_type
    when 'swiss'
      generate_swiss_roll
    when 'iris'
      generate_iris_like_data
    else
      generate_clustered_data
    end
    
    puts "Generated #{data.size} points in #{data.first.size} dimensions"
    
    # Reduce dimensions
    print "Running UMAP..."
    umap = AnnEmbed::Embedder.new(method: :umap, n_components: 2, n_neighbors: 15, random_seed: 42)
    umap_data = umap.fit_transform(data)
    puts " done"
    
    print "Running PCA..."
    pca = AnnEmbed::PCA.new(n_components: 2)
    pca_data = pca.fit_transform(data)
    variance_explained = pca.cumulative_explained_variance_ratio[-1]
    puts " done (explained variance: #{(variance_explained * 100).round(1)}%)"
    
    print "Running SVD..."
    u, s, vt = AnnEmbed.svd(data, 2, n_iter: 5)
    svd_data = u
    puts " done"
    
    # Perform K-means clustering on reduced data
    print "Clustering with K-means..."
    
    # Find optimal k using elbow method
    elbow_results = AnnEmbed::Clustering.elbow_method(umap_data, k_range: 2..6)
    
    # Use library method to detect optimal k
    optimal_k = AnnEmbed::Clustering.detect_optimal_k(elbow_results)
    
    puts "Elbow method results:"
    elbow_results.sort.each do |k, inertia|
      puts "  k=#{k}: #{inertia.round(2)}"
    end
    puts "Detected optimal k: #{optimal_k}"
    
    kmeans_umap = AnnEmbed::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
    kmeans_labels_umap = kmeans_umap.fit_predict(umap_data)
    
    kmeans_pca = AnnEmbed::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
    kmeans_labels_pca = kmeans_pca.fit_predict(pca_data)
    
    kmeans_svd = AnnEmbed::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
    kmeans_labels_svd = kmeans_svd.fit_predict(svd_data)
    puts " done"
    
    # Calculate metrics
    silhouette_umap = AnnEmbed::Clustering.silhouette_score(umap_data, kmeans_labels_umap)
    silhouette_pca = AnnEmbed::Clustering.silhouette_score(pca_data, kmeans_labels_pca)
    silhouette_svd = AnnEmbed::Clustering.silhouette_score(svd_data, kmeans_labels_svd)
    
    # Generate HTML
    html = generate_visualization_html(
      data: data,
      umap_data: umap_data,
      pca_data: pca_data,
      svd_data: svd_data,
      true_labels: true_labels,
      kmeans_labels_umap: kmeans_labels_umap,
      kmeans_labels_pca: kmeans_labels_pca,
      kmeans_labels_svd: kmeans_labels_svd,
      dataset_name: dataset_name,
      metrics: {
        silhouette_umap: silhouette_umap,
        silhouette_pca: silhouette_pca,
        silhouette_svd: silhouette_svd,
        optimal_k: optimal_k,
        elbow_results: elbow_results,
        pca_variance_explained: variance_explained
      }
    )
    
    File.write(output_file, html)
    puts "\nVisualization saved to: #{output_file}"
    puts "Open in browser: open #{output_file}"
  end
  
  def generate_clustered_data(n_points_per_cluster: 50, n_features: 50, n_clusters: 3)
    data = []
    labels = []
    
    n_clusters.times do |cluster_id|
      # Keep values smaller and normalized to avoid UMAP issues
      center = Array.new(n_features) { (rand - 0.5) * 0.3 + cluster_id * 0.3 }
      
      n_points_per_cluster.times do
        point = center.map { |c| c + (rand - 0.5) * 0.1 }
        data << point
        labels << cluster_id
      end
    end
    
    [data, labels, "Gaussian Clusters"]
  end
  
  def generate_swiss_roll(n_points: 150)
    data = []
    labels = []
    
    n_points.times do |i|
      t = 0.5 * Math::PI * (1 + 2 * i.to_f / n_points)
      height = rand
      
      x = t * Math.cos(t) * 0.1
      y = height * 0.1
      z = t * Math.sin(t) * 0.1
      
      point = [x, y, z]
      
      # Add correlated features
      10.times do |j|
        point << x * Math.sin(j) + y * Math.cos(j) + (rand - 0.5) * 0.01
      end
      
      # Add random features
      37.times do
        point << rand * 0.01
      end
      
      data << point
      labels << (t / (3 * Math::PI) * 3).to_i
    end
    
    [data, labels, "Swiss Roll"]
  end
  
  def generate_iris_like_data
    data = []
    labels = []
    
    species_params = [
      { sepal_length: 0.5, sepal_width: 0.34, petal_length: 0.15, petal_width: 0.02 },
      { sepal_length: 0.59, sepal_width: 0.28, petal_length: 0.43, petal_width: 0.13 },
      { sepal_length: 0.65, sepal_width: 0.30, petal_length: 0.55, petal_width: 0.20 }
    ]
    
    species_params.each_with_index do |params, species_id|
      50.times do
        features = [
          params[:sepal_length] + (rand - 0.5) * 0.08,
          params[:sepal_width] + (rand - 0.5) * 0.06,
          params[:petal_length] + (rand - 0.5) * 0.08,
          params[:petal_width] + (rand - 0.5) * 0.04
        ]
        
        # Expand to 50 dimensions
        expanded = features.dup
        
        features.each do |f1|
          features.each do |f2|
            expanded << f1 * f2 * 0.01
          end
        end
        
        features.each_with_index do |f, i|
          expanded << Math.sin(f) * 0.01 * (i + 1)
          expanded << Math.cos(f) * 0.01 * (i + 1)
        end
        
        while expanded.length < 50
          expanded << rand * 0.01
        end
        
        data << expanded[0...50]
        labels << species_id
      end
    end
    
    [data, labels, "Iris-like Dataset"]
  end
  
  def generate_visualization_html(data:, umap_data:, pca_data:, svd_data:, true_labels:, 
                                   kmeans_labels_umap:, kmeans_labels_pca:, kmeans_labels_svd:, 
                                   dataset_name:, metrics:)
    <<~HTML
      <!DOCTYPE html>
      <html>
      <head>
          <title>AnnEmbed Visualization - #{dataset_name}</title>
          <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
          <style>
              body {
                  font-family: Arial, sans-serif;
                  margin: 20px;
                  background: #f5f5f5;
              }
              h1 {
                  color: #333;
                  text-align: center;
              }
              .container {
                  display: grid;
                  grid-template-columns: repeat(2, 1fr);
                  gap: 20px;
                  max-width: 1400px;
                  margin: 0 auto;
              }
              .plot {
                  background: white;
                  border-radius: 8px;
                  padding: 10px;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
              }
              .stats {
                  background: white;
                  border-radius: 8px;
                  padding: 20px;
                  margin: 20px auto;
                  max-width: 1400px;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
              }
              table {
                  width: 100%;
                  border-collapse: collapse;
              }
              th, td {
                  padding: 10px;
                  text-align: left;
                  border-bottom: 1px solid #ddd;
              }
              th {
                  background-color: #f8f8f8;
                  font-weight: bold;
              }
              .metrics {
                  display: grid;
                  grid-template-columns: repeat(4, 1fr);
                  gap: 20px;
                  margin-top: 20px;
              }
              .metric-card {
                  background: #f8f8f8;
                  padding: 15px;
                  border-radius: 5px;
                  text-align: center;
              }
              .metric-value {
                  font-size: 24px;
                  font-weight: bold;
                  color: #333;
              }
              .metric-label {
                  color: #666;
                  margin-top: 5px;
              }
          </style>
      </head>
      <body>
          <h1>Dimensionality Reduction & Clustering Analysis</h1>
          <h2 style="text-align: center; color: #666;">Dataset: #{dataset_name}</h2>
          
          <div class="stats">
              <h2>Dataset Information</h2>
              <table>
                  <tr>
                      <th>Property</th>
                      <th>Value</th>
                  </tr>
                  <tr>
                      <td>Original Dimensions</td>
                      <td>#{data.first.size}</td>
                  </tr>
                  <tr>
                      <td>Number of Points</td>
                      <td>#{data.size}</td>
                  </tr>
                  <tr>
                      <td>True Number of Clusters</td>
                      <td>#{true_labels.uniq.size}</td>
                  </tr>
                  <tr>
                      <td>K-means Detected Clusters</td>
                      <td>#{metrics[:optimal_k]}</td>
                  </tr>
              </table>
              
              <div class="metrics">
                  <div class="metric-card">
                      <div class="metric-value">#{metrics[:silhouette_umap].round(3)}</div>
                      <div class="metric-label">UMAP + K-means<br>Silhouette Score</div>
                  </div>
                  <div class="metric-card">
                      <div class="metric-value">#{metrics[:silhouette_pca].round(3)}</div>
                      <div class="metric-label">PCA + K-means<br>Silhouette Score</div>
                  </div>
                  <div class="metric-card">
                      <div class="metric-value">#{metrics[:silhouette_svd].round(3)}</div>
                      <div class="metric-label">SVD + K-means<br>Silhouette Score</div>
                  </div>
                  <div class="metric-card">
                      <div class="metric-value">#{(metrics[:pca_variance_explained] * 100).round(1)}%</div>
                      <div class="metric-label">PCA Variance<br>Explained</div>
                  </div>
              </div>
          </div>
          
          <div class="container">
              <div class="plot" id="umap-true"></div>
              <div class="plot" id="umap-kmeans"></div>
              <div class="plot" id="pca-true"></div>
              <div class="plot" id="pca-kmeans"></div>
              <div class="plot" id="svd-true"></div>
              <div class="plot" id="svd-kmeans"></div>
          </div>
          
          <div class="stats">
              <h2>Elbow Method Results</h2>
              <div id="elbow-plot" style="height: 400px;"></div>
          </div>
          
          <script>
              // Color schemes
              const colorscale = [
                  [0, 'rgb(255, 0, 0)'],
                  [0.5, 'rgb(0, 255, 0)'],
                  [1, 'rgb(0, 0, 255)']
              ];
              
              // UMAP with true labels
              Plotly.newPlot('umap-true', [{
                  x: #{umap_data.map { |p| p[0] }.to_json},
                  y: #{umap_data.map { |p| p[1] }.to_json},
                  mode: 'markers',
                  marker: {
                      color: #{true_labels.to_json},
                      size: 8,
                      colorscale: colorscale,
                      showscale: false
                  },
                  type: 'scatter'
              }], {
                  title: 'UMAP - True Labels',
                  xaxis: { title: 'UMAP 1' },
                  yaxis: { title: 'UMAP 2' },
                  height: 400
              });
              
              // UMAP with K-means labels
              Plotly.newPlot('umap-kmeans', [{
                  x: #{umap_data.map { |p| p[0] }.to_json},
                  y: #{umap_data.map { |p| p[1] }.to_json},
                  mode: 'markers',
                  marker: {
                      color: #{kmeans_labels_umap.to_json},
                      size: 8,
                      colorscale: colorscale,
                      showscale: false
                  },
                  type: 'scatter'
              }], {
                  title: 'UMAP - K-means Clusters',
                  xaxis: { title: 'UMAP 1' },
                  yaxis: { title: 'UMAP 2' },
                  height: 400
              });
              
              // PCA with true labels
              Plotly.newPlot('pca-true', [{
                  x: #{pca_data.map { |p| p[0] }.to_json},
                  y: #{pca_data.map { |p| p[1] }.to_json},
                  mode: 'markers',
                  marker: {
                      color: #{true_labels.to_json},
                      size: 8,
                      colorscale: colorscale,
                      showscale: false
                  },
                  type: 'scatter'
              }], {
                  title: 'PCA - True Labels',
                  xaxis: { title: 'PC 1' },
                  yaxis: { title: 'PC 2' },
                  height: 400
              });
              
              // PCA with K-means labels
              Plotly.newPlot('pca-kmeans', [{
                  x: #{pca_data.map { |p| p[0] }.to_json},
                  y: #{pca_data.map { |p| p[1] }.to_json},
                  mode: 'markers',
                  marker: {
                      color: #{kmeans_labels_pca.to_json},
                      size: 8,
                      colorscale: colorscale,
                      showscale: false
                  },
                  type: 'scatter'
              }], {
                  title: 'PCA - K-means Clusters',
                  xaxis: { title: 'PC 1' },
                  yaxis: { title: 'PC 2' },
                  height: 400
              });
              
              // SVD with true labels
              Plotly.newPlot('svd-true', [{
                  x: #{svd_data.map { |p| p[0] }.to_json},
                  y: #{svd_data.map { |p| p[1] }.to_json},
                  mode: 'markers',
                  marker: {
                      color: #{true_labels.to_json},
                      size: 8,
                      colorscale: colorscale,
                      showscale: false
                  },
                  type: 'scatter'
              }], {
                  title: 'SVD - True Labels',
                  xaxis: { title: 'Component 1' },
                  yaxis: { title: 'Component 2' },
                  height: 400
              });
              
              // SVD with K-means labels
              Plotly.newPlot('svd-kmeans', [{
                  x: #{svd_data.map { |p| p[0] }.to_json},
                  y: #{svd_data.map { |p| p[1] }.to_json},
                  mode: 'markers',
                  marker: {
                      color: #{kmeans_labels_svd.to_json},
                      size: 8,
                      colorscale: colorscale,
                      showscale: false
                  },
                  type: 'scatter'
              }], {
                  title: 'SVD - K-means Clusters',
                  xaxis: { title: 'Component 1' },
                  yaxis: { title: 'Component 2' },
                  height: 400
              });
              
              // Elbow plot
              const elbowData = #{metrics[:elbow_results].to_a.sort.to_h.to_json};
              Plotly.newPlot('elbow-plot', [{
                  x: Object.keys(elbowData),
                  y: Object.values(elbowData),
                  mode: 'lines+markers',
                  marker: { size: 10 },
                  line: { width: 2 }
              }], {
                  title: 'Elbow Method - Optimal K Selection',
                  xaxis: { title: 'Number of Clusters (k)' },
                  yaxis: { title: 'Inertia' },
                  height: 400
              });
          </script>
      </body>
      </html>
    HTML
  end
end