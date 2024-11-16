from flask import Flask, request, render_template, send_from_directory
from ImageProcessing.Kmeans import get_dominant_color
from utils.rhs_color_mapper import rgb_to_hex, find_closest_colors_with_ucl
from utils.ManageImage import save_result
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Route to serve images from the /image directory
@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory('image', filename)

# Flask route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to process the form and display results
@app.route('/process', methods=['POST'])
def process_image():
    # Get form data
    image_path = request.form['image_path']
    mask_path = request.form.get('mask_path')
    num_colors = int(request.form.get('num_colors', 5))

    # Get dominant colors
    segmented_image, dominant_colors, best_k, best_score, len_info, ks_and_score = get_dominant_color(f"Services/image/{image_path}", max_k=num_colors, mask_path=f"Services/image/{mask_path}")
    save_result(segmented_image, f"Services/image/final_{image_path}")

    # Generate and save the graph for ks_and_score
    ks, scores = zip(*ks_and_score)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker='o', label='K vs Silhouette Score')
    plt.scatter(best_k, best_score, color='red', label=f'Max Value (K={best_k}, Score={best_score:.3f})', zorder=5)
    plt.title('KMeans Clustering: K vs Silhouette Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid(True)
    graph_filename = f"graph_{image_path}.png"
    graph_path = os.path.join('Services/image', graph_filename)
    plt.savefig(graph_path)
    plt.close()  # Ensure the plot is closed to release resources

    # Generate HTML output
    result_html = f"<h3>Top closest colors for {image_path} using K-Means with {best_k} clusters, the silhouette {best_score}:</h3>"
    result_html += f"<img src='/image/{image_path}' style='max-width: 300px; max-height: 300px;'>"
    result_html += f"<img src='/image/final_{image_path}' style='max-width: 300px; max-height: 300px;'>"
    result_html += f"<img src='/image/{graph_filename}' style='max-width: 400px; max-height: 400px;'><br>"  # Embed the graph here
    result_html += f"<p>Total non-black pixel feeding KMeans {len_info[1]} </p>"

    for color_number, color in enumerate(dominant_colors):
        # rgb_values = tuple(color.astype(int))
        r, g, b = map(int, color)
        # hex_color = rgb_to_hex(rgb_values)
        rgb_hex = rgb_to_hex(r, g, b)
        # closest_colors = find_closest_colors_with_ucl(rgb_values)
        top_5_closest_in_rgb = find_closest_colors_with_ucl(rgb_hex)

        # Display primary color box and details
        result_html += (
            f"<div style='background-color:{rgb_hex}; width:50px; height:50px; display:inline-block;'></div> "
            f"{color_number+1}. RGB: {r},{g},{b}. Percentage: {(len_info[0][color_number][0]/len_info[1])*100:.2f}%. Total pixel: {len_info[0][color_number][0]}<br>"
            "<ul>"
        )
        
        for color in top_5_closest_in_rgb:
            label, rgb, distance, ucl_name = color
            closest_hex = rgb_to_hex(rgb[0], rgb[1], rgb[2])
            result_html += (
                f"<li>"
                f"<div style='background-color:{closest_hex}; width:25px; height:25px; display:inline-block;'></div> "
                f"RHS#: {label}, RGB: {rgb}, Distance: {distance:.2f}, RHS Color Name: {ucl_name}"
                f"</li>"
            )

        result_html += "</ul>"

    return result_html

if __name__ == "__main__":
    app.run(debug=True)
