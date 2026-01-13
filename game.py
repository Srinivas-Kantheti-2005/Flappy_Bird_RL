from config import *
import objects
import pygame
from pygame.locals import *
import random
import sys
import time
import numpy as np
import torch
import agents.user_agent, agents.random_agent, agents.dqn_agent
import matplotlib.pyplot as plt
from collections import deque
import os
import pandas as pd
from scipy import stats

AGENTS = ["user_agent", "random_agent", "dqn_agent"]

# Load assets
bird_image = pygame.image.load('assets/bluebird-upflap.png')
bird_image = pygame.transform.scale(bird_image, (BIRD_WIDTH, BIRD_HEIGHT))
pipe_image = pygame.image.load('assets/pipe-green.png')
pipe_image = pygame.transform.scale(pipe_image, (PIPE_WIDHT, PIPE_HEIGHT))
ground_image = pygame.image.load('assets/base.png')
ground_image = pygame.transform.scale(ground_image, (GROUND_WIDHT, GROUND_HEIGHT))
BACKGROUND = pygame.image.load('assets/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))


class Game:
    def __init__(self, agent_name, device):
        if agent_name not in AGENTS:
            sys.exit("Agent not defined")
        if device not in ["cpu", "cuda"]:
            sys.exit("Computing device not available")

        if agent_name == "user_agent":
            self.agent = agents.user_agent.User_agent()
        elif agent_name == "random_agent":
            self.agent = agents.random_agent.Random_agent()
        elif agent_name == "dqn_agent":
            self.agent = agents.dqn_agent.DQN_agent(device)

        self.device = device
        self.bird = None
        self.ground = None
        self.pipes = None
        self.score = None
        self.turn = None
        self.train = False

        self.metrics = {
            'episode_scores': [],
            'moving_avg_scores': [],
            'losses': [],
            'epsilon_values': []
        }
        self.score_window = deque(maxlen=100)

    def init_game(self):
        self.bird = objects.Bird(bird_image)
        self.ground = objects.Ground(ground_image, 0)
        self.pipes = []
        self.score = 0
        self.turn = 0

        for i in range(3):
            xpos = PIPE_DISTANCE * i + PIPE_DISTANCE
            ysize = random.randint(200, 300)
            self.pipes.append(objects.Pipe(pipe_image, False, xpos, ysize))
            self.pipes.append(objects.Pipe(pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))

    def pipe_handling(self):
        if self.pipes[0].pos[0] <= -100:
            del self.pipes[0:2]
            xpos = PIPE_DISTANCE * 3 - 100
            ysize = random.randint(150, 350)
            self.pipes.append(objects.Pipe(pipe_image, False, xpos, ysize))
            self.pipes.append(objects.Pipe(pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))

    def collision(self):
        if self.bird.pos[1] < 0 or self.bird.pos[1] > SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT:
            return True
        if self.pipes[0].pos[0] - self.bird.pos[2] < self.bird.pos[0] < self.pipes[0].pos[0] + self.pipes[0].pos[2]:
            if self.pipes[0].pos[1] < self.bird.pos[1] + self.bird.pos[3] or self.pipes[0].pos[1] - PIPE_GAP > self.bird.pos[1]:
                return True
        return False

    def score_update(self):
        if self.bird.pos[0] == self.pipes[0].pos[0]:
            self.score += 1

    def game_state(self):
        state = []
        for pipe in self.pipes:
            if self.bird.pos[0] < pipe.pos[0] + pipe.pos[2]:
                state.append((-self.bird.pos[0] + pipe.pos[2] + pipe.pos[0]) / PIPE_DISTANCE)
                state.append((pipe.pos[1] - PIPE_GAP / 2 - self.bird.pos[1] - self.bird.pos[3] / 2) / SCREEN_HEIGHT * 2)
                break
        state.append(self.bird.speed / SPEED)
        return state

    def reward(self):
        reward = 0.1
        state = self.game_state()
        vertical_dist = abs(state[1])
        bird_speed = state[2]

        if vertical_dist < 0.2:
            reward += 0.5
        elif vertical_dist < 0.4:
            reward += 0.2

        if abs(bird_speed) > 0.8:
            reward -= 0.3

        if self.collision():
            reward = -10
        elif self.bird.pos[0] == self.pipes[0].pos[0]:
            reward += 5

        if self.score > 50:
            reward *= 1.2

        screen_center = SCREEN_HEIGHT / 2
        distance_from_center = abs(self.bird.pos[1] - screen_center) / screen_center
        if distance_from_center < 0.3:
            reward += 0.2

        return round(reward, 4)

    def main(self, draw):
        if draw:
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
            pygame.display.set_icon(bird_image)
            pygame.display.set_caption('Flappy Bird')
            clock = pygame.time.Clock()
            font = pygame.font.SysFont(None, 36)

        active_episode = True
        self.init_game()

        while active_episode:
            if draw:
                clock.tick(30)
                screen.blit(BACKGROUND, (0, 0))
                for event in pygame.event.get():
                    if event.type == QUIT:
                        active_episode = False

            state = self.game_state()
            action = self.agent.act(state, self.train)

            if action == 1:
                self.bird.bump()
            if action == -1:
                active_episode = False

            self.bird.update()
            for pipe in self.pipes:
                pipe.update()

            self.score_update()
            self.pipe_handling()

            if self.collision():
                active_episode = False

            if self.train:
                next_state = self.game_state()
                reward = self.reward()
                self.agent.buffer.add(state, next_state, reward, action)

            self.turn += 1

            if draw:
                self.bird.draw(screen)
                self.ground.draw(screen)
                for pipe in self.pipes:
                    pipe.draw(screen)

                # Display score
                score_surface = font.render(f"Score: {self.score}", True, (255, 255, 255))
                shadow = font.render(f"Score: {self.score}", True, (0, 0, 0))
                screen.blit(shadow, (12, 12))
                screen.blit(score_surface, (10, 10))

                # Display target score
                target_surface = font.render("Target: 100", True, (255, 255, 0))
                screen.blit(target_surface, (SCREEN_WIDHT - 150, 10))
                pygame.display.update()

            if self.score >= 100:
                if draw:
                    win_font = pygame.font.SysFont(None, 48)
                    win_text = win_font.render("🎉 You Win! 🎉", True, (0, 255, 0))
                    screen.blit(win_text, (SCREEN_WIDHT // 2 - 100, SCREEN_HEIGHT // 2 - 20))
                    pygame.display.update()
                    time.sleep(2)
                active_episode = False

        if draw:
            pygame.display.quit()
            pygame.quit()

        return self.score

    def train_agent(self, draw, episodes, batches, hyperparameter):
        convergence = 0
        loss = 0
        mean_score = []
        time_start = time.time()

        if not isinstance(self.agent, agents.dqn_agent.DQN_agent):
            sys.exit("Agent is not trainable")

        self.train = True

        for episode in range(1, episodes + 1):
            eps = hyperparameter["eps_end"] + (hyperparameter["eps_start"] - hyperparameter["eps_end"]) * np.exp(-1. * episode / episodes * 10)
            lr = hyperparameter["lr_end"] + (hyperparameter["lr_start"] - hyperparameter["lr_end"]) * np.exp(-1. * episode / episodes * 10)
            self.agent.epsilon = eps
            self.agent.lr = lr
            self.agent.batch_size = hyperparameter["batch_size"]
            self.agent.gamma = hyperparameter["gamma"]

            _ = self.main(draw)

            for _ in range(batches):
                loss += self.agent.train()

            self.train = False
            test_score = self.main(False)
            mean_score.append(test_score)

            if test_score == 100:
                convergence += 1
            else:
                convergence = 0

            self.metrics['episode_scores'].append(test_score)
            self.score_window.append(test_score)
            self.metrics['moving_avg_scores'].append(np.mean(self.score_window))
            self.metrics['epsilon_values'].append(eps)
            avg_loss = loss / batches if batches > 0 else 0
            self.metrics['losses'].append(avg_loss)

            if convergence == 2:
                print("Agent performed faultless")
                break

            self.agent.save("metrics/best_model.pth")
            self.agent.save(f"metrics/model_ep{episode}.pth")

            if episode % 10 == 0:
                print(f"Episode [{episode}/{episodes}] - Loss: {round(avg_loss, 6)} - MeanTestScore: {round(np.mean(mean_score))}")
                mean_score = []

            loss = 0
            self.train = True

        self.train = False
        self._plot_training_metrics()
        print("Training finished after {} episodes".format(episode))

    def _plot_training_metrics(self):
        if not os.path.exists('metrics'):
            os.makedirs('metrics')

        episodes = range(len(self.metrics['episode_scores']))
        fig = plt.figure(figsize=(15, 12))

        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(episodes, self.metrics['episode_scores'], 'b-', alpha=0.3, label='Episode Score')
        ax1.plot(episodes, self.metrics['moving_avg_scores'], 'r-', linewidth=2, label='Moving Avg (100)')
        if len(self.metrics['episode_scores']) > 0:
            std = pd.Series(self.metrics['episode_scores']).rolling(window=20).std().ffill()
            avg = np.array(self.metrics['moving_avg_scores'])
            ax1.fill_between(episodes, avg - std, avg + std, alpha=0.2, color='r')
        ax1.set_title('Training Performance')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.legend()

        ax2 = plt.subplot(2, 2, 2)
        losses = np.array(self.metrics['losses'])
        smoothed = pd.Series(losses).rolling(window=10, min_periods=1).mean()
        ax2.plot(losses, 'g-', alpha=0.3, label='Raw Loss')
        ax2.plot(smoothed, 'g-', linewidth=2, label='Smoothed Loss')
        ax2.set_yscale('log')
        ax2.set_title('Loss Curve')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(self.metrics['epsilon_values'], color='purple', linewidth=2)
        ax3.set_title('Epsilon Decay')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')

        ax4 = plt.subplot(2, 2, 4)
        scores = np.array(self.metrics['episode_scores'])
        ax4.hist(scores, bins=30, color='skyblue', alpha=0.7, density=True)
        kde = stats.gaussian_kde(scores)
        x_vals = np.linspace(min(scores), max(scores), 100)
        ax4.plot(x_vals, kde(x_vals), 'r-', linewidth=2)
        ax4.set_title('Score Distribution')
        ax4.set_xlabel('Score')
        ax4.set_ylabel('Density')

        plt.tight_layout()
        plt.savefig('metrics/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig('metrics/training_metrics.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        np.savez('metrics/training_data.npz',
                 scores=self.metrics['episode_scores'],
                 moving_avg=self.metrics['moving_avg_scores'],
                 losses=self.metrics['losses'],
                 epsilon=self.metrics['epsilon_values'])

        stats_summary = {
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'final_loss': self.metrics['losses'][-1] if self.metrics['losses'] else None,
            'convergence_episode': np.argmax(np.array(self.metrics['moving_avg_scores']) > 95)
            if any(np.array(self.metrics['moving_avg_scores']) > 95) else None
        }

        with open('metrics/training_statistics.txt', 'w') as f:
            for key, value in stats_summary.items():
                f.write(f"{key}: {value}\n")
