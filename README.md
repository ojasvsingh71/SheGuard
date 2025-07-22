# SheGuard

SheGuard is an AI-powered deepfake detection web application designed to identify manipulated media and combat misinformation. Using advanced machine learning models, SheGuard classifies uploaded images as real or deepfake to promote digital safety and awareness.

## Features
- Image classification as **Real** or **Deepfake**
- User-friendly web interface
- AI-powered detection using TensorFlow and Google Vision API
- Secure and scalable backend

## Tech Stack
### Frontend:
- React.js
- Tailwind CSS

### Backend:
- Flask (Python)
- Google Vision API
- TensorFlow
- OpenCV

### Deployment:
- GitHub
- Cloud Hosting (e.g., Heroku, AWS, or Render)

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Node.js and npm
- Virtual environment (venv)

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/ojasvsingh71/SheGuard.git
cd SheGuard/sheguard-backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # For MacOS/Linux
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask application
python app.py
```

### Frontend Setup
```bash
# Navigate back to project root (if coming from backend setup)
cd ..

# Navigate to frontend directory
cd sheguard

# Install dependencies
npm install

# Start the frontend
npm start
```

## API Endpoints
| Method | Endpoint         | Description          |
|--------|-----------------|----------------------|
| POST   | `/upload`       | Upload an image     |
| GET    | `/result`       | Get classification result |

## How It Works
1. User uploads an image.
2. The backend processes the image using the AI model.
3. The model predicts whether the image is **Real** or **Deepfake**.
4. The result is displayed to the user.

## Contribution Guidelines
1. Fork the repository.
2. Create a new branch (`feature-branch` or `bugfix-branch`).
3. Commit changes and push to GitHub.
4. Open a pull request for review.

## License
This project is licensed under the MIT License.

## Contact
For any questions or collaborations, reach out to:
- **GitHub**: [ojasvsingh71](https://github.com/ojasvsingh71)
- **Email**: ojasvsingh0@gmail.com

