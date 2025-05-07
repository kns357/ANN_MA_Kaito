import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Layer config
layers = [64, 64, 64, 64]

# Camera system
class Camera:
    def __init__(self):
        self.pos = np.array([3.0, 3.0, 3.0])
        self.yaw = -135.0  # Adjusted to face origin
        self.pitch = -35.0  # Adjusted to face origin
        self.speed = 0.1
        self.sensitivity = 0.1

cam = Camera()
keys = {}

# Connection data
simulate_20_million = True
if simulate_20_million:
    total_lines = 20000
    start_points = np.random.uniform(-1, 1, (total_lines, 3))
    end_points = np.random.uniform(-1, 1, (total_lines, 3))
else:
    # (Original layer generation code here)
    pass

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glColor3f(0.0, 1.0, 0.0)
    glLineWidth(0.1)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_DEPTH_TEST)

def update_camera():
    yaw_rad = math.radians(cam.yaw)
    pitch_rad = math.radians(cam.pitch)
    
    # Calculate front vector
    front = np.array([
        math.cos(yaw_rad) * math.cos(pitch_rad),
        math.sin(pitch_rad),
        math.sin(yaw_rad) * math.cos(pitch_rad)
    ])
    center = cam.pos + front
    
    glLoadIdentity()
    gluLookAt(*cam.pos, *center, 0, 1, 0)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    update_camera()
    
    glBegin(GL_LINES)
    for i in range(total_lines):
        glVertex3fv(start_points[i])
        glVertex3fv(end_points[i])
    glEnd()
    glutSwapBuffers()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w/h, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    keys[key] = True

def keyboard_up(key, x, y):
    keys[key] = False

def motion(x, y):
    dx = x - 500  # Assuming window center at (500,500)
    dy = y - 500
    cam.yaw += dx * cam.sensitivity
    cam.pitch -= dy * cam.sensitivity
    cam.pitch = max(-89.9, min(89.9, cam.pitch))
    glutWarpPointer(500, 500)  # Keep mouse centered
    glutPostRedisplay()

def idle():
    yaw_rad = math.radians(cam.yaw)
    move_vec = np.array([
        math.sin(yaw_rad),
        0,
        math.cos(yaw_rad)
    ]) * cam.speed
    
    if b'w' in keys: cam.pos += move_vec
    if b's' in keys: cam.pos -= move_vec
    if b'a' in keys: cam.pos -= np.cross(move_vec, [0,1,0])
    if b'd' in keys: cam.pos += np.cross(move_vec, [0,1,0])
    if b'q' in keys: cam.pos[1] += cam.speed
    if b'e' in keys: cam.pos[1] -= cam.speed
    
    glutPostRedisplay()

if __name__ == '__main__':
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(1000, 1000)
    glutCreateWindow(b"3D Neural Network - Fixed")
    glutSetCursor(GLUT_CURSOR_NONE)
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutKeyboardUpFunc(keyboard_up)
    glutPassiveMotionFunc(motion)
    glutIdleFunc(idle)
    glutMainLoop()