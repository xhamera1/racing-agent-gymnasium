# Project Outline: Autonomous Driving Agent

## Overview
The main objective of this project is to train an artificial intelligence agent capable of autonomously driving a vehicle on a race track using Reinforcement Learning in continuous spaces. 

## Core Technologies
**Environment:** We will use `CarRacing-v2` from the **Gymnasium** library. It perfectly fits the continuous requirements by providing continuous observation spaces (visual pixel data) and continuous action spaces (smooth control over steering, acceleration, and braking).
* **Algorithm:** We selected **Soft Actor-Critic (SAC)** from the **stable-baselines3** library. SAC is a modern, state-of-the-art algorithm designed specifically for continuous control tasks, known for its stability and efficiency.

## Project Scope
Our workflow consists of three primary stages:
1. **Training & Tuning:** We will teach the agent to navigate the track, experimenting with various hyperparameter configurations to optimize its learning process and overall performance.
2. **Vision System Customization:** Since the agent relies on visual input, we will implement and test different structures of Convolutional Neural Networks (CNNs) that serve as the agent's "eyes".
3. **Deterministic Evaluation:** Once trained, we will select the best-performing model. We will then simulate its driving with random exploration completely turned off, allowing us to observe and evaluate its pure, optimal driving skills.