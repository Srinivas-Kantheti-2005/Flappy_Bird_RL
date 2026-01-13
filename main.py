import game
import numpy as np

if __name__ == "__main__":
    print("=" * 50)
    print("FlapDuel RL Project - DQN Agent")
    print("Initializing Game Environment...")
    
    g = game.Game("dqn_agent", "cpu")

    hyperparameter = {
        "lr_start": 5e-4,
        "lr_end": 1e-5,
        "batch_size": 64,
        "gamma": 0.95,
        "eps_start": 1.0,
        "eps_end": 0.05
    }

    print("\nStarting Training...")
    g.train_agent(draw=True, episodes=100, batches=50, hyperparameter=hyperparameter)
    print("Training Complete.\n")

    print("Testing Best Model...")
    final_score = g.main(draw=True)
    print("Evaluation Complete.")

    print("=" * 50)
    print(f"Your Score: {final_score}")
    print("=" * 50)
