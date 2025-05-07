import pygame as pg
import sys
import random as ra
from time import sleep

pg.init()
width, height = 2000, 1000
window = pg.display.set_mode((width, height))
pg.display.set_caption("snake, I think ... maybe")

rect_size = 50
x, y = 0, 0
speed = 50
color_snake = (255, 0, 0)
color_food = (0, 255, 0)
dirx = 1000
diry = 500
updx = x = 1
updy = y = 0
food_score = 0

body_coords = [(0, 0)]
food_coords = (ra.randrange(0, int(2000/50))*50, ra.randrange(0, int(1000/50))*50)

pg.font.init()
my_font = pg.font.SysFont('Comic Sans MS', 30)


clock = pg.time.Clock()
running = True
while running:
    clock.tick(10)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    keys = pg.key.get_pressed()
    if keys[pg.K_LEFT]:
        x = -1
        y = 0
    if keys[pg.K_RIGHT]:
        x = 1
        y = 0
    if keys[pg.K_UP]:
        x = 0
        y = -1
    if keys[pg.K_DOWN]:
        x = 0
        y = 1

    body_coords[0] = (dirx, diry)
    
    #dir x and y are cooridantes even though it say dir, i don't fucking knwo why, actually I do, I am too fucking lazy with naming variables in a reasonable fashion
    if dirx % 50 == 0 and diry % 50 == 0:
        if (updx == 0 and y == 0) or (updy == 0 and x == 0):
            updx = x
            updy = y

        for i in range(len(body_coords)-1):
            body_coords[-(i+1)] = body_coords[-(i+2)]
        
    if (dirx, diry) == food_coords:
        food_coords = (ra.randrange(0, int(2000/50))*50, ra.randrange(0, int(1000/50))*50)
        food_score += 1
        body_coords.append((dirx-(dirx%50)-50*updx*len(body_coords), diry-(diry%50)-50*updy*len(body_coords)))

        print(body_coords)


    #update coords of snaky
    dirx += updx * speed
    diry += updy * speed

    

    #collision stuff
    if dirx < 0 or diry < 0 or dirx > width or diry > height:
        text_surface = my_font.render('GAME OVER', False, (0.5, 0.5, 0.5))
        window.blit(text_surface, (0,0))
        #sleep(3)
        running = False
        
    for coo in body_coords:
        if (dirx, diry) == coo:
            if coo != body_coords[0]:
                text_surface = my_font.render('GAME OVER', False, (0.5, 0.5, 0.5))
                window.blit(text_surface, (0,0))
                #sleep(3)
                running = False

    window.fill((0, 0, 0))  # Black background
    
    pg.draw.rect(window, color_food, (food_coords[0], food_coords[1], rect_size, rect_size))

    for coo in body_coords:
        pg.draw.rect(window, color_snake, (coo[0], coo[1], rect_size, rect_size))
    

    pg.display.flip()

# Clean up
pg.quit()
sys.exit()
