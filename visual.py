
import pygame
from pygame.locals import *


GRAY = (127, 127, 127)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

x = [0,256,512,768,1024]
y = [0,144,288,432,576]

GVak = [
pygame.Rect(0, 0, 256, 144),pygame.Rect(256, 0,256, 144),pygame.Rect(512, 0, 256, 144),pygame.Rect(768, 0, 256, 144),pygame.Rect(1024, 0, 256, 144),
pygame.Rect(0, 144, 256, 144),pygame.Rect(1024, 144, 256, 144),
pygame.Rect(0, 288, 256, 144),pygame.Rect(1024, 288, 256, 144),
pygame.Rect(0, 432,256, 144),pygame.Rect(1024, 432,256, 144),
pygame.Rect(0, 576, 256, 144),pygame.Rect(256, 576,256, 144),pygame.Rect(512, 576,256, 144),pygame.Rect(768, 576,256, 144),pygame.Rect(1024, 576,256, 144)
]
ConVak = [
pygame.Rect(256, 144,256, 144),pygame.Rect(512, 144, 256, 144),pygame.Rect(768, 144, 256, 144),
pygame.Rect(256, 288,256, 144),pygame.Rect(512, 288, 256, 144),pygame.Rect(768, 288,256, 144),
pygame.Rect(256, 432,256, 144),pygame.Rect(512, 432,256, 144),pygame.Rect(768, 432, 256, 144)
]
pygame.init()
screen = pygame.display.set_mode((1280,720))
screen.fill(WHITE)

for i in GVak:
    pygame.draw.rect(screen,GRAY, i)
for i in ConVak:
    pygame.draw.rect(screen,WHITE, i)
for i in x:
    for j in y:
        pygame.draw.rect(screen, (0,0,0), (i,j,257,145), 2) 

pygame.display.update()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()
