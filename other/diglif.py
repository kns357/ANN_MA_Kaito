import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

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

# Generate neuron positions
positions = []
for lx in range(layers):
    layer = []
    for y in range(neurons_y):
        for z in range(neurons_z):
            pos = [lx * spacing, y * spacing, z * spacing]
            layer.append(pos)
    positions.append(layer)

# Generate connections (with color and thickness)
connections = []
for i in range(len(positions) - 1):
    src_layer = positions[i]
    dst_layer = positions[i + 1]
    for src in src_layer:
        for _ in range(connections_per_neuron):
            dst = dst_layer[np.random.randint(0, len(dst_layer))]
            thickness = np.random.uniform(0.5, 3.0)

            # Random color from white (1,1,1) to blue (0,0,1)
            t = np.random.rand()
            color = (1 - t, 1 - t, 1.0)
            connections.append((src, dst, thickness, color))


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
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Camera transforms
    glTranslatef(pan_x, pan_y, zoom)
    glRotatef(angle_y, 1, 0, 0)
    glRotatef(angle_x, 0, 1, 0)

    # Draw connections
    for c in connections:
        draw_connection(c[0], c[1], c[2], c[3])

    # Draw neurons
    glColor3f(1.0, 0.3, 0.8)
    for layer in positions:
        for pos in layer:
            draw_sphere(pos, radius)

    glutSwapBuffers()


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
