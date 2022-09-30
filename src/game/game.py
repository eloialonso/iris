from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import gym
import numpy as np
import pygame
from PIL import Image

from envs import WorldModelEnv
from game.keymap import get_keymap_and_action_names
from utils import make_video


class Game:
    def __init__(self, env: Union[gym.Env, WorldModelEnv], keymap_name: str, size: Tuple[int, int], fps: int, verbose: bool, record_mode: bool) -> None:
        self.env = env
        self.height, self.width = size
        self.fps = fps
        self.verbose = verbose
        self.record_mode = record_mode
        self.keymap, self.action_names = get_keymap_and_action_names(keymap_name)

        self.record_dir = Path('media') / 'recordings'

        print('Actions:')
        for key, idx in self.keymap.items():
            print(f'{pygame.key.name(key)}: {self.action_names[idx]}')

    def run(self) -> None:
        pygame.init()

        header_height = 100 if self.verbose else 0
        font_size = 24
        screen = pygame.display.set_mode((self.width, self.height + header_height))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, font_size)
        header_rect = pygame.Rect(0, 0, self.width, header_height)

        def clear_header():
            pygame.draw.rect(screen, pygame.Color('black'), header_rect)
            pygame.draw.rect(screen, pygame.Color('white'), header_rect, 1)

        def draw_text(text, idx_line, idx_column=0):
            pos = (5 + idx_column * int(self.width // 4), 5 + idx_line * font_size)
            assert (0 <= pos[0] <= self.width) and (0 <= pos[1] <= header_height)
            screen.blit(font.render(text, True, pygame.Color('white')), pos)

        def draw_game(image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                assert isinstance(image, Image.Image)
            pygame_image = np.array(image.resize((self.width, self.height), resample=Image.NEAREST)).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(pygame_image)
            screen.blit(surface, (0, header_height))

        if isinstance(self.env, gym.Env):
            _, info = self.env.reset(return_info=True)
            img = info['rgb']
        else:
            self.env.reset()
            img = self.env.render()
        
        draw_game(img)

        clear_header()
        pygame.display.flip()

        episode_buffer = []
        segment_buffer = []
        recording = False

        do_reset, do_wait = False, False
        should_stop = False
        
        while not should_stop:

            action = 0  # noop
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_stop = True
                if event.type == pygame.KEYDOWN and event.key in self.keymap.keys():
                    action = self.keymap[event.key]
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    do_reset = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_PERIOD:
                    do_wait = not do_wait
                if event.type == pygame.KEYDOWN and event.key == pygame.K_COMMA:
                    if not recording: 
                        recording = True
                        print('Started recording.')
                    else:
                        print('Stopped recording.')    
                        self.save_recording(np.stack(segment_buffer))
                        recording = False
                        segment_buffer = []

            if action == 0:
                pressed = pygame.key.get_pressed()
                for key, action in self.keymap.items():
                    if pressed[key]:
                        break
                else:
                    action = 0

            if do_wait:
                continue

            _, reward, done, info = self.env.step(action)

            img = info['rgb'] if isinstance(self.env, gym.Env) else self.env.render()
            draw_game(img)

            if recording:
                segment_buffer.append(np.array(img))

            if self.record_mode:
                episode_buffer.append(np.array(img))

            if self.verbose:
                clear_header()
                draw_text(f'Action: {self.action_names[action]}', idx_line=0)
                draw_text(f'Reward: {reward if isinstance(reward, float) else reward.item(): .2f}', idx_line=1)
                draw_text(f'Done: {done}', idx_line=2)
                if info is not None:
                    assert isinstance(info, dict)
                    for i, (k, v) in enumerate(info.items()):
                        draw_text(f'{k}: {v}', idx_line=i, idx_column=1)

            pygame.display.flip()   # update screen
            clock.tick(self.fps)    # ensures game maintains the given frame rate

            if do_reset or done:
                self.env.reset()
                do_reset = False

                if self.record_mode:
                    if input('Save episode? [Y/n] ').lower() != 'n':
                        self.save_recording(np.stack(episode_buffer))
                    episode_buffer = []

        pygame.quit()

    def save_recording(self, frames):
        self.record_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        np.save(self.record_dir / timestamp, frames)
        make_video(self.record_dir / f'{timestamp}.mp4', fps=15, frames=frames)
        print(f'Saved recording {timestamp}.')
