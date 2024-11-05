from flask import Flask, request, render_template, send_from_directory
from ImageProcessing.Kmeans import get_dominant_color
from utils.rhs_color_mapper import rgb_to_hex, find_closest_colors_with_ucl

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
    dominant_colors = get_dominant_color(f"Services/image/{image_path}", num_colors=num_colors, mask_path=f"Services/image/{mask_path}")

    # Generate HTML output for each color
    result_html = f"<h3>Top closest colors for {image_path} using K-Means with {num_colors} clusters:</h3>"
    # Shows image from image path
    result_html += f"<img src='/image/{image_path}' style='max-width: 300px; max-height: 300px;'><br><br>"

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
            f"{color_number+1}. RGB: {r},{g},{b}<br>"
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
