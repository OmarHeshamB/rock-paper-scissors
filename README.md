# Rock Paper Scissors AI Game

![visitors](https://visitor-badge.glitch.me/badge?page_id=OBADR01.rock-paper-scissors)

A computer vision-based Rock Paper Scissors game where you play against an AI using your webcam. The AI predicts your hand gestures in real-time using a trained machine learning model.

## Features
- Real-time hand gesture recognition using TensorFlow/Keras
- Webcam integration with mirror-like display
- Interactive gameplay against computer AI
- Data collection and model training scripts
- Easy-to-use command-line interface

## Demo Video
[Watch the demo](https://youtu.be/0uSA3xyXlwM)

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy

## Installation & Setup

1. Clone the repository:
```sh
git clone https://github.com/OBADR01/rock-paper-scissors.git
cd rock-paper-scissors
```

2. Create and activate a virtual environment (recommended):
```sh
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

3. Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

### 1. Gather Training Images
Collect images for each gesture (rock, paper, scissors, and none):
```sh
python gather_images.py rock 200
python gather_images.py paper 200
python gather_images.py scissors 200
python gather_images.py none 200
```

### 2. Train the Model
Train the neural network on your collected data:
```sh
python train.py
```

### 3. Test the Model (Optional)
Test the trained model on sample images:
```sh
python test.py path/to/test_image.jpg
```

### 4. Play the Game
Start the interactive game:
```sh
python play.py
```

## How to Play
- Run `python play.py`
- Position your hand in the white box on the left
- Make rock, paper, or scissors gestures
- The AI will predict your move and play against you
- Press 'q' to quit

## Project Structure
```
rock-paper-scissors/
├── gather_images.py    # Data collection script
├── train.py           # Model training script
├── test.py            # Model testing script
├── play.py            # Main game script
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
├── images/            # Icon images for gestures
├── image_data/        # Collected training images
└── rock-paper-scissors-model.h5  # Trained model
```

## Contributing
Feel free to fork this repository and submit pull requests with improvements!

## License
This project is open source and available under the [MIT License](LICENSE).
