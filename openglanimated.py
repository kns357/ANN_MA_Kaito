import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import time
import math

# Parameters
layers = 10
neurons_y = 10
neurons_z = 10
spacing = 10
radius = 0.3
connections_per_neuron = 4

# Camera controls
angle_x, angle_y = 30, -30
zoom = -50
pan_x, pan_y = 0, 0
mouse_last = [0, 0]
button_pressed = None

# Time tracker
start_time = time.time()

# Generate neuron positions
positions = []
for lx in range(layers):
    layer = []
    for y in range(neurons_y):
        for z in range(neurons_z):
            pos = [lx * spacing, y * spacing, z * spacing]
            layer.append(pos)
    positions.append(layer)

# Generate connections (with base color/thickness and a phase offset for animation)
connections = []
for i in range(len(positions) - 1):
    src_layer = positions[i]
    dst_layer = positions[i + 1]
    for src in src_layer:
        for _ in range(connections_per_neuron):
            dst = dst_layer[np.random.randint(0, len(dst_layer))]
            base_thickness = np.random.uniform(0.5, 3.0)
            t = np.random.rand()
            base_color = (1 - t, 1 - t, 1.0)  # white to blue
            phase = np.random.uniform(0, 2 * np.pi)
            connections.append((src, dst, base_thickness, base_color, phase))


def draw_sphere(pos, r):
    glPushMatrix()
    glTranslatef(*pos)
    quad = gluNewQuadric()
    gluSphere(quad, r, 8, 8)
    gluDeleteQuadric(quad)
    glPopMatrix()


def draw_connection(p1, p2, thickness, color):
    glLineWidth(thickness)
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex3f(*p1)
    glVertex3f(*p2)
    glEnd()


def display():
    global start_time
    t = time.time() - start_time

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Camera transforms
    glTranslatef(pan_x, pan_y, zoom)
    glRotatef(angle_y, 1, 0, 0)
    glRotatef(angle_x, 0, 1, 0)

    # Draw connections with animated color/thickness
    for (p1, p2, base_thick, base_col, phase) in connections:
        # Animate using sine wave
        pulse = 0.5 * (1 + math.sin(t * 2 + phase))
        thickness = base_thick * (0.6 + 0.4 * pulse)

        # Color interpolation: white ↔ blue
        c = [base_col[i] * (0.6 + 0.4 * pulse) for i in range(3)]
        draw_connection(p1, p2, thickness, c)

    # Draw neurons
    glColor3f(1.0, 0.3, 0.8)
    for layer in positions:
        for pos in layer:
            draw_sphere(pos, radius)

    glutSwapBuffers()
    glutPostRedisplay()  # <- loop continuously


def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / float(height or 1), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)


def mouse(button, state, x, y):
    global button_pressed
    button_pressed = button if state == GLUT_DOWN else None
    mouse_last[0], mouse_last[1] = x, y


def motion(x, y):
    global angle_x, angle_y, zoom, pan_x, pan_y
    dx = x - mouse_last[0]
    dy = y - mouse_last[1]
    if button_pressed == GLUT_LEFT_BUTTON:
        angle_x += dx * 0.5
        angle_y += dy * 0.5
    elif button_pressed == GLUT_RIGHT_BUTTON or button_pressed == GLUT_MIDDLE_BUTTON:
        pan_x += dx * 0.01
        pan_y -= dy * 0.01
    mouse_last[0], mouse_last[1] = x, y
    glutPostRedisplay()


def mouse_wheel(button, dir, x, y):
    global zoom
    zoom += dir * 2
    glutPostRedisplay()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1000, 800)
    glutCreateWindow(b"3D CNN Visualization")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)

    try:
        glutMouseWheelFunc(mouse_wheel)
    except:
        pass

    glutMainLoop()


if __name__ == '__main__':
    main()
