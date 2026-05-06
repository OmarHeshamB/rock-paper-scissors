#from keras.models import load_model
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize frame FIRST to ensure consistent dimensions
    frame = cv2.resize(frame, (1280, 720))
    # Flip horizontally so the webcam behaves like a mirror
    frame = cv2.flip(frame, 1)

    # Define box positions and size
    user_box_topleft = (100, 160)
    user_box_bottomright = (500, 560)
    comp_box_topleft = (780, 160)
    comp_box_bottomright = (1180, 560)
    box_size = (user_box_bottomright[0] - user_box_topleft[0], user_box_bottomright[1] - user_box_topleft[1])

    # rectangle for user to play
    cv2.rectangle(frame, user_box_topleft, user_box_bottomright, (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, comp_box_topleft, comp_box_bottomright, (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[user_box_topleft[1]:user_box_bottomright[1], user_box_topleft[0]:user_box_bottomright[0]]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread("images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, box_size)
        # Place icon exactly in the computer's box
        frame[comp_box_topleft[1]:comp_box_bottomright[1], comp_box_topleft[0]:comp_box_bottomright[0]] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
