from flask import Flask, render_template, request, send_file
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    # Open the uploaded image and apply image enhancement techniques
    image = Image.open(request.files['image'])
    enhanced_image = image.enhance()
    
    # Save the enhanced image to a temporary file
    enhanced_image_file = "enhanced_image.jpg"
    enhanced_image.save(enhanced_image_file)
    
    # Render the enhanced image template with the original and enhanced images
    return render_template('enhanced_image.html', original_image=image, enhanced_image=enhanced_image)

@app.route('/download')
def download():
    # Send the enhanced image file to the user
    return send_file("enhanced_image.jpg", as_attachment=True)

if __name__ == '__main__':
    app.run()
