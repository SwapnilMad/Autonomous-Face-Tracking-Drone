import time
import cv2
import numpy as np
from aigym import box, seeding


class SimulatorEnv:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.width = 500
        self.height = 500
        self.state = None
        self.maxspeed = 60
        print('initialized')
        self.observation_space = box.Box(0, 1, shape=(3,), dtype=np.float32)
        self.action_space = box.Box(-1, +1, (3,), dtype=np.float32)

    def move(self, action):
        (lr, fb, ud) = action
        prev_x, prev_y, prev_w = self.state
        prev_x = prev_x * self.width
        prev_y = prev_y * self.width
        prev_w = prev_w * self.width
        print('prev', (prev_x, prev_y, prev_w))

        curr_x = prev_x - 0.5*self.maxspeed * lr
        curr_y = prev_y + 0.5*self.maxspeed * ud
        curr_w = prev_w + 0.5*self.maxspeed * fb

        difference = abs(prev_w-curr_w)
        if prev_w > curr_w:
            curr_x = curr_x + 0.5 * difference
            curr_y = curr_y + 0.5 * difference
        elif prev_w < curr_w:
            curr_x = curr_x - 0.5 * difference
            curr_y = curr_y - 0.5 * difference
        print('next', (curr_x, curr_y, curr_w))

    def step(self, action):
        reward = 0
        action = np.clip(action, -1, +1).astype(np.float32)
        prev_x, prev_y, prev_w = self.state
        prev_x = prev_x * self.width
        prev_y = prev_y * self.width
        prev_w = prev_w * self.width
        cv2.waitKey(1)
        prev_rem_x = self.width - (prev_x + prev_w)
        prev_diff_x = abs(prev_x - prev_rem_x)

        prev_rem_y = self.width - (prev_y + prev_w)
        prev_diff_y = abs(prev_y-prev_rem_y)

        done = False
        #print('action', action)

        (lr, fb, ud) = action
        curr_x = prev_x - self.maxspeed * lr
        curr_y = prev_y + self.maxspeed * ud
        curr_w = prev_w + 1.5 * self.maxspeed * fb

        difference = abs(prev_w - curr_w)
        if prev_w > curr_w:
            curr_x = curr_x + 0.5 * difference
            curr_y = curr_y + 0.5 * difference
        else:
            curr_x = curr_x - 0.5 * difference
            curr_y = curr_y - 0.5 * difference

        cv2.waitKey(1)
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (self.width, self.height))

        if curr_x < 0 or curr_x + curr_w > self.width or curr_y < 0 or curr_y+curr_w > self.width or curr_w < 50:
            done = True
            reward = reward - 10.0
        else:
            #print((curr_x, curr_y), (curr_x + curr_w, curr_y + curr_w))
            cv2.rectangle(frame,(int(curr_x), int(curr_y)), (int(curr_x)+int(curr_w), int(curr_y)+int(curr_w)), (0, 255, 0), 3)
            self.state = (curr_x / self.width, curr_y / self.width, curr_w / self.width)

        cv2.imshow('Frame', frame)

        if not done:
            rem = self.width - (curr_x + curr_w)
            diff_x = abs(curr_x - rem)

            remy = self.width - (curr_y + curr_w)
            diff_y = abs(curr_y - remy)

            if (abs(curr_w - int(self.width / 5)) <= 10) and (diff_y < 30) and (diff_x < 30):
                #print('perfect')
                reward = reward + 10.0
            else:
                #print('diff', diff)
                if diff_x > 30 and diff_x < prev_diff_x:
                    reward = reward + (0.003 * (self.width-diff_x))
                elif diff_y > 30 and diff_y < prev_diff_y:
                    reward = reward + (0.001 * (self.width - diff_y))
                elif curr_w - int(self.width / 5) > 10:
                    if curr_w < prev_w:
                        reward = reward + 0.2
                    else:
                        reward = reward - 0.2

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        cv2.waitKey(1)
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (self.width, self.height))
        cv2.imshow('Frame', frame)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        data = np.random.randint(1, self.width-1, 3)
        ret, frame = self.cap.read()
        while data[0] + data[2] > self.width-1 or data[1] + data[2] > self.width-1:
            data = np.random.randint(1, self.width-1, 3)
        frame = cv2.resize(frame, (self.width, self.width))
        cv2.rectangle(frame, (data[0], data[1]), (data[0] + data[2], data[1] + data[2]), (0, 255, 255), 3)
        self.state = (data[0] / self.width, data[1] / self.width, data[2] / self.width)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        return np.array(self.state)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

#s = SimulatorEnv()
#for i in range(5):
    #s.reset()
    #s.move([0.5, 0, 0]) #right
    #s.move([0, -0.5, 0]) #forward
    #s.move([0, 0, 0.5]) #up
