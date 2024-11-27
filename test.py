import torch
import pygame
import numpy as np
from model import Linear_QNet
from game import Snake
from settings import *


def test_snake_ai(model_path='model/model.pth'):
    # Load the trained model
    model = Linear_QNet(len(Snake().get_state()), HIDDEN_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize the game
    game = Snake()

    # Run the game loop for demonstration
    running = True
    while running:
        # Get the current state of the snake
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float)

        # Predict action based on the model
        with torch.no_grad():
            prediction = model(state_tensor)
            action = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[action] = 1

        # Execute the action in the game
        reward, done, score = game.play_step(final_move)

        # Check if the game is over
        if done:
            game.reset()
            print(f'Game Over. Score: {score}')

        # Quit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()


if __name__ == "__main__":
    test_snake_ai()
