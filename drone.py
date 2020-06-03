from djitellopy import Tello
import time
import cv2
import numpy as np
from aigym import box, seeding


class Drone:
    def __init__(self):
        self.me = Tello()
        self.me.connect()
        self.me.front_back_velocity = 0
        self.me.left_right_velocity = 0
        self.me.up_down_velocity = 0
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.me.yaw_velocity = 0
        self.me.speed = 0
        print(self.me.get_battery())
        self.me.streamoff()
        self.me.streamon()
        self.me.takeoff()
        # self.me.move_up(50)

        self.width = 500
        self.height = 500
        self.hei_hf = int(self.height / 9)
        self.state = None
        self.maxspeed = 40
        cv2.waitKey(1)
        print('initialized')
        self.steps_beyond_done = None
        self.observation_space = box.Box(0, 1, shape=(3,), dtype=np.float32)
        self.action_space = box.Box(-1, +1, (3,), dtype=np.float32)

    def step(self, action):
        c1 = 0
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
        prev_diff_y = abs(prev_y - prev_rem_y)

        curr_w = 0
        curr_y = 0
        curr_x = 0
        done = False
        print('action', action)
        self.me.send_rc_control(int(action[0] * self.maxspeed), int(action[1] * self.maxspeed),
                                int(action[2] * self.maxspeed), 0)
        frame_read = self.me.get_frame_read()
        img = frame_read.frame
        img = cv2.resize(img, (self.width, self.height))
        faces = self.face_cascade.detectMultiScale(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
            nframe = img[y:y + h, x:x + w]
            eyes = self.eyes_cascade.detectMultiScale(nframe)

            for eye in eyes:
                c1 = c1 + 1

            if (c1 > 0):
                print('state', (x, y, w, h))
                curr_w = w
                curr_x = x
                curr_y = y
                self.state = (x / self.width, y / self.width, w / self.width)
            else:
                done = True
                reward = reward - 10.0

        if not done:
            rem = self.width - (curr_x + curr_w)
            diff_x = abs(curr_x - rem)

            remy = self.width - (curr_y + curr_w)
            diff_y = abs(curr_y - remy)

            if (abs(curr_w - int(self.width / 5)) <= 10) and (diff_y < 30) and (diff_x < 30):
                # acceptable range width - (115, 135); y = 50,115;
                print('perfect')
                reward = reward + 10.0
            else:
                # print('diff', diff)
                if diff_x > 30 and diff_x < prev_diff_x:
                    reward = reward + (0.003 * (self.width - diff_x))
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
        frame_read = self.me.get_frame_read()
        img = frame_read.frame
        img = cv2.resize(img, (self.width, self.height))
        cv2.imshow('Frame', img)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.me.send_rc_control(0, 0, 0, 0)
        c = 0
        # self.me.land()
        # time.sleep(0.2)
        cv2.waitKey(1)

        frame_read = self.me.get_frame_read()
        img = frame_read.frame
        img = cv2.resize(img, (self.width, self.height))
        print('reset')
        self.steps_beyond_done = 0
        faces = self.face_cascade.detectMultiScale(img)
        print('faces', len(faces))

        flag = True
        if (len(faces) > 0):
            nframe = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
            eyes = self.eyes_cascade.detectMultiScale(nframe)
            cv2.rectangle(img, (faces[0][0], faces[0][1]), (faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]),
                          (0, 255, 255), 3)

            for eye in eyes:
                c = c + 1
            if (c > 0):
                flag = False
            c = 0
        cv2.imshow('Frame', img)

        while flag:
            self.me.send_rc_control(0, 0, 0, 30)
            frame_read = self.me.get_frame_read()
            img = frame_read.frame
            img = cv2.resize(img, (self.width, self.height))
            cv2.imshow('Frame', img)
            faces = self.face_cascade.detectMultiScale(img)
            print('faces', faces)
            if (len(faces) > 0):
                (x1, y1, w1, h1) = faces[0]
                nframe = img[y1:y1 + h1, x1:x1 + w1]
                cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 3)

                eyes = self.eyes_cascade.detectMultiScale(nframe)

                for eye in eyes:
                    c = c + 1
                if (c > 0):
                    flag = False
                    self.me.send_rc_control(0, 0, 0, 0)
                    self.state = (x1 / self.width, y1 / self.width, w1 / self.width)
                c = 0
            cv2.waitKey(1)

        return np.array(self.state)

    def close(self):
        self.me.land()
        cv2.destroyAllWindows()
