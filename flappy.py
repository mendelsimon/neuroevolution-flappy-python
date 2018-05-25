import pygame
from pygame.locals import *
from random import randrange, normalvariate, random
import random
import sys
from neural_network import NeuralNetwork as nn

FPS = 60

WINDOW_HEIGHT = 700
WINDOW_WIDTH = 500
# WINDOW_HEIGHT = 1000
# WINDOW_WIDTH = 1400
GROUND_HEIGHT = 112
ground_offset = 0
PLAY_AREA_HEIGHT = WINDOW_HEIGHT - GROUND_HEIGHT
PLAYER_HEIGHT = 25
PLAYER_WIDTH = 25
# TODO calculate jump distance/frames to determine to pipe width (width = distance = frames*SPEED)
PIPE_WIDTH = 70
PIPE_HEAD_HEIGHT = 26

SPEED = 5.5
JUMP_SPEED = -7
GRAVITY = 0.4
JUMP_HEIGHT = (JUMP_SPEED ** 2) / (GRAVITY * 2)

# This area of the pipe will never have a gap
PIPE_SOLID_AREA = int(PLAY_AREA_HEIGHT / 5)
PIPE_GAP_STARTING_HEIGHT = JUMP_HEIGHT * 2.5
PIPE_GAP_MINIMUM_HEIGHT = JUMP_HEIGHT * 1.1 + PLAYER_HEIGHT

current_frame = 0
last_pipe_frame = 0  # The frame when the last pipe spawned
pipe_gap_height = PIPE_GAP_STARTING_HEIGHT

frames_per_draw = 1
pipes = []
POPULATION_SIZE = 50
population = []
generation = 0
score = 0

BIRD_IMAGE = pygame.image.load('python.png')


class Bird:
    def __init__(self, brain=nn([8, 10, 1])):
        self.x = (WINDOW_WIDTH / 2) - (PLAYER_WIDTH / 2)
        self.y = PLAY_AREA_HEIGHT / 4
        self.fall_speed = 0
        self.brain = brain

    def jump(self):
        self.fall_speed = JUMP_SPEED

    def update_position(self):
        self.y += self.fall_speed
        self.fall_speed += GRAVITY

    def get_input(self):
        pipe_coords = get_next_pipes_coords()
        guess = self.brain.guess([self.y, self.fall_speed, *pipe_coords])
        return guess[0] >= 0.5

    def check_collision(self):
        GRACE_LENGTH = 5  # The amount of collision that won't count as collision

        # Check if player left window bounds
        if self.y + GRACE_LENGTH < 0 or self.y - GRACE_LENGTH > PLAY_AREA_HEIGHT:
            return True  # Game over

        # Check if player collided with a pipe
        for pipe in pipes:
            if pipe.x < self.x + PLAYER_WIDTH - GRACE_LENGTH and pipe.x + PIPE_WIDTH - GRACE_LENGTH > int(self.x):
                if self.y < pipe.gap_y - GRACE_LENGTH or self.y + PLAYER_HEIGHT > pipe.gap_y + pipe.gap_height + GRACE_LENGTH:
                    return True  # Game over

        return False  # No collision

    @staticmethod
    def mutation(x):
        threshold = 0.85
        if random.random() > threshold:
            return normalvariate(x, abs(x) ** 0.35)
        return x


class Pipe:
    def __init__(self, gap_height):
        self.x = WINDOW_WIDTH
        self.gap_height = gap_height
        self.gap_y = randrange(PIPE_SOLID_AREA, PLAY_AREA_HEIGHT - PIPE_SOLID_AREA - int(self.gap_height))
        self.scored = False

        head_image = pygame.image.load('pipe_head.png')
        self.bottom_head_image = pygame.transform.scale(head_image, (PIPE_WIDTH, PIPE_HEAD_HEIGHT))
        self.top_head_image = pygame.transform.flip(self.bottom_head_image, False, True)
        self.body_image = pygame.image.load('pipe_body.png')

    def update_position(self):
        self.x -= SPEED


def main():
    global DISPLAY_SURFACE, FPSCLOCK, BACKGROUND_IMAGE, GROUND_IMAGE, population
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAY_SURFACE = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flappy Python")
    BACKGROUND_IMAGE = pygame.image.load('background.png')
    BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, (WINDOW_WIDTH, PLAY_AREA_HEIGHT))
    GROUND_IMAGE = pygame.image.load('ground.png')
    GROUND_IMAGE = pygame.transform.scale(GROUND_IMAGE, (WINDOW_WIDTH, GROUND_HEIGHT))

    population = [Bird() for _ in range(POPULATION_SIZE)]

    while True:
        play()
        # display_game_over()


def play():
    global score, pipes, current_frame, pipe_gap_height, population
    pipes = []
    pipe_gap_height = PIPE_GAP_STARTING_HEIGHT
    score = 0
    game_over = False

    while not game_over:
        # Get input
        for _ in range(frames_per_draw):
            get_input()

            # Compute changes
            for bird in population:
                bird.update_position()
            spawn_pipe()
            compute_pipes()
            for bird in population:
                if bird.check_collision():
                    if len(population) > 1:
                        population.remove(bird)
            if len(population) == 1 and population[0].check_collision():
                game_over = True
                for _ in range(POPULATION_SIZE - 1):
                    new_bird = Bird(population[0].brain.copy())
                    new_bird.brain.mutate(new_bird.mutation)
                    population.append(new_bird)
            current_frame += 1

        # Update screen
        draw_screen()

        # Tick
        FPSCLOCK.tick(FPS)


def get_input():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            global frames_per_draw
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key in (K_PLUS, K_KP_PLUS):
                frames_per_draw += 10
                print(f'Frameskip: {frames_per_draw - 1}')
            if event.key in (K_MINUS, K_KP_MINUS):
                frames_per_draw -= 10
                if frames_per_draw < 1:
                    frames_per_draw = 1
                print(f'Frameskip: {frames_per_draw - 1}')
    for bird in population:
        if bird.get_input():
            bird.jump()


def spawn_pipe():
    global current_frame, last_pipe_frame, pipe_gap_height

    min_distance_to_pipe = (WINDOW_WIDTH / 3) + SPEED * 5
    frames_since_pipe = current_frame - last_pipe_frame
    if frames_since_pipe * SPEED >= min_distance_to_pipe:
        # Chance of pipe spawning is 1/(FPS), or 100% if it took too long
        if randrange(int(FPS)) == 0 or frames_since_pipe * SPEED > min_distance_to_pipe * 2:
            pipes.append(Pipe(pipe_gap_height))
            last_pipe_frame = current_frame

            # Make the gap smaller in future pipes
            if pipe_gap_height > PIPE_GAP_MINIMUM_HEIGHT:
                pipe_gap_height -= JUMP_HEIGHT / 50


def compute_pipes():
    global score

    for index, pipe in enumerate(pipes):
        # Update position
        pipe.update_position()

        # Update score
        if pipe.x + PIPE_WIDTH <= (WINDOW_WIDTH / 2) and not pipe.scored:
            score += 1
            pipe.scored = True
            # print(score)

        # Remove old pipes
        if pipe.x < 0 - PIPE_WIDTH:
            del pipes[index]


def get_next_pipes_coords():
    coords = []
    for pipe, next_pipe in zip(pipes, pipes[1:]):
        if not pipe.scored:
            coords.extend((pipe.gap_y, pipe.x, pipe.gap_height))
            if next_pipe:
                coords.extend((next_pipe.gap_y, next_pipe.x, next_pipe.gap_height))
            else:
                coords.extend((-1, -1, -1))
            return coords
    for pipe in pipes:
        if not pipe.scored:
            return [pipe.gap_y, pipe.x, pipe.gap_height, -1, -1, -1]
    return [-1, -1, -1, -1, -1, -1]


def draw_screen():
    draw_background()
    draw_birds()
    draw_pipes()
    pygame.display.update()


def draw_background():
    global ground_offset
    DISPLAY_SURFACE.blit(BACKGROUND_IMAGE, (0, 0))
    DISPLAY_SURFACE.blit(GROUND_IMAGE, (-ground_offset, PLAY_AREA_HEIGHT))
    DISPLAY_SURFACE.blit(GROUND_IMAGE, (-ground_offset + WINDOW_WIDTH, PLAY_AREA_HEIGHT))
    ground_offset += SPEED
    ground_offset %= WINDOW_WIDTH


def draw_birds():
    bird_image = pygame.transform.scale(BIRD_IMAGE, (PLAYER_WIDTH, PLAYER_HEIGHT))
    for bird in population:
        DISPLAY_SURFACE.blit(bird_image, (bird.x, bird.y))


def draw_pipes():
    for pipe in pipes:
        top_head_pos = (int(pipe.x), pipe.gap_y - PIPE_HEAD_HEIGHT)
        bottom_head_pos = (int(pipe.x), pipe.gap_y + pipe.gap_height)

        bottom_height = int(PLAY_AREA_HEIGHT - (pipe.gap_y + pipe.gap_height)) + 1

        top_body_pos = (int(pipe.x), 0)
        top_body_scale = (PIPE_WIDTH, pipe.gap_y - PIPE_HEAD_HEIGHT)
        bottom_body_pos = (int(pipe.x), pipe.gap_y + int(pipe.gap_height) + PIPE_HEAD_HEIGHT)
        bottom_body_scale = (PIPE_WIDTH, bottom_height - PIPE_HEAD_HEIGHT)

        top_body_image = pygame.transform.scale(pipe.body_image, top_body_scale)
        bottom_body_image = pygame.transform.scale(pipe.body_image, bottom_body_scale)

        DISPLAY_SURFACE.blit(top_body_image, top_body_pos)
        DISPLAY_SURFACE.blit(pipe.top_head_image, top_head_pos)
        DISPLAY_SURFACE.blit(pipe.bottom_head_image, bottom_head_pos)
        DISPLAY_SURFACE.blit(bottom_body_image, bottom_body_pos)


def display_game_over():
    DISPLAY_SURFACE.fill((250, 50, 50))
    pygame.display.update()
    pygame.time.wait(500)


if __name__ == '__main__':
    main()
