from cube import Cube
from constants import *
from utility import *

import random
import numpy as np
import copy
import matplotlib.pyplot as plt

class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4))

        self.lr = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.01
        
        self.total_reward = []
        self.histogram = []

    def get_optimal_policy(self, state):
        optimal_policy = np.argmax(self.q_table[state])
        # print("1", self.color, state, optimal_policy)
        return optimal_policy

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        sample = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] = (1 - self.lr) * self.q_table[state][action] + self.lr * sample

    def move(self, snack, other_snake):
        self.pre_head = copy.deepcopy(self.head)
        
        cur_state = self.create_state(snack, other_snake)
        
        action = self.make_action(cur_state)

        if action == 0:  # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1:  # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2:  # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3:  # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]


        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)
    
        
        new_state = self.create_state(snack, other_snake)
        # print("2", new_state)
        return cur_state, new_state, action

    def check_board(self, location, loc_x, loc_y, other_snake):
        res = []
        for i in range(3):
            if location[0] + loc_x[i] < 1 or location[0] + loc_x[i] >= ROWS - 1 or location[1] + loc_y[i] < 1 or location[1] + loc_y[i] >= ROWS - 1:
                res.append(0) # out of board
            elif (location[0] + loc_x[i], location[1] + loc_y[i]) in list(map(lambda z: z.pos, self.body)):
                res.append(0) # hit to it's own body
            elif (location[0] + loc_x[i], location[1] + loc_y[i]) in list(map(lambda z: z.pos, other_snake.body)):
                res.append(0) # hit to the other snake's body
            elif (location[0] + loc_x[i], location[1] + loc_y[i]) == other_snake.head.pos:
                res.append(0) # hit to the other snake's head
            else:
                res.append(1)
        
        return res
        

    def create_state(self, snack, other_snake):
        location = [self.head.pos[0], self.head.pos[1]]
        res = []
        loc_x = [-1, 0, 1]
        
        loc_y = [1, 1, 1]
        res.extend(self.check_board(location, loc_x, loc_y, other_snake))

        loc_y = [0, 0, 0]
        res.extend(self.check_board(location, loc_x, loc_y, other_snake))
                
        loc_y = [-1, -1, -1]
        res.extend(self.check_board(location, loc_x, loc_y, other_snake))

        snack_pos = self.where_is_snack(snack)
        
        return tuple(res + [snack_pos])

    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def calculate_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def check_for_getting_close(self, snack):
        dist1 = self.calculate_distance(snack.pos, self.pre_head.pos)
        dist2 = self.calculate_distance(snack.pos, self.head.pos)
        return GOT_CLOSER_REWARD if dist2 - dist1 < 0 else -GOT_CLOSER_REWARD * 3

    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        
        reward += self.check_for_getting_close(snack)
        
        if self.check_out_of_board():
            reward += LOSE_REWARD # Punish the snake for getting out of the board
            win_other = True
            reset(self, other_snake, win_other)
        
        if self.head.pos == snack.pos:
            self.addCube() # Reward the snake for eating
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += EAT_REWARD
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward += LOSE_REWARD # Punish the snake for hitting itself
            win_other = True
            reset(self, other_snake, win_other)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                reward += LOSE_REWARD # Punish the snake for hitting the other snake
                win_other = True
                reset(self, other_snake, win_other)
            else:
                if len(self.body) > len(other_snake.body):
                    reward += EAT_REWARD # Reward the snake for hitting the head of the other snake and being longer
                    win_self = True
                    win_other = False
                    reset(self, other_snake, win_other)
                elif len(self.body) == len(other_snake.body):
                    reward += DO_NOTHING # No winner
                else:
                    reward += LOSE_REWARD # Punish the snake for hitting the head of the other snake and being shorter
                    win_other = True
                    reset(self, other_snake, win_other)
                    
        self.total_reward.append(reward)
        
        return snack, reward, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

        self.histogram.append(np.mean(self.total_reward))
        self.total_reward.clear()
        print(len(self.histogram))
        if len(self.histogram) >= 100:
            # self.epsilon *= 0.95
            plt.plot(self.histogram)
            plt.savefig('img3')

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)

    def where_is_snack(self, snack):
        if abs(snack.pos[0] - self.head.pos[0]) > abs(snack.pos[1] - self.head.pos[1]):
            if snack.pos[0] <= self.head.pos[0]:
                return 0
            if snack.pos[0] > self.head.pos[0]:
                return 1
        else:
            if snack.pos[1] <= self.head.pos[1]:
                return 2
            if snack.pos[1] > self.head.pos[1]:
                return 3