# coding=utf-8

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys

import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as ls
import local_shapes as loc_s
import tree


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True

# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return

    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_ESCAPE:
        sys.exit()

def gauss(x, y, s, sigma, mu):
    return s * np.exp(-((x - mu[0])**2 + (y - mu[1])**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def sumaGauss(x, y, s, sigma, mus, gaussianas):
    if gaussianas == 1:
        return gauss(x, y, s, sigma, mus[gaussianas-1])
    return gauss(x, y, s, sigma, mus[gaussianas-1]) + sumaGauss(x, y, s, sigma, mus, gaussianas-1)

# Funcion modificada de ex_2dplotter.py
def generateMesh(xs, ys, function, color):
    vertices = []
    indices = []
    start_index = 0

    # We generate a vertex for each sample x,y,z
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            x_0 = xs[i]
            y_0 = ys[j]
            x_1 = xs[i+1]
            y_1 = ys[j+1]

            a = np.array([x_0, y_0, function(x_0, y_0)])
            b = np.array([x_1, y_0, function(x_1, y_0)])
            c = np.array([x_1, y_1, function(x_1, y_1)])
            d = np.array([x_0, y_1, function(x_0, y_1)])

            vertex_, indices_ = loc_s.createColorNormalsQuadIndexation(start_index, a, b, c, d, color)

            vertices += vertex_
            indices += indices_
            start_index += 4

    return bs.Shape(vertices, indices)



if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Forest", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Defining shader programs
    # pipeline = ls.SimpleFlatShaderProgram()
    pipeline = ls.SimpleGouraudShaderProgram()
    # pipeline = ls.SimplePhongShaderProgram()
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(pipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))

    # Argumentos entregados
    archivo = sys.argv[1]
    semilla = int(sys.argv[2])
    gaussianas = int(sys.argv[3])
    terreno = int(sys.argv[4])
    arboles = int(sys.argv[5]) * int(2 * semilla / 10) ** 2
    ramas = int(sys.argv[6])


    # Se crea un area de 2*semilla x 2*semilla unidades de medida
    xs = np.ogrid[-semilla:semilla:semilla * 1j]
    ys = np.ogrid[-semilla:semilla:semilla * 1j]

    # Se aleatorizan mu's dentro de los cuadrantes para la cantidad de gaussianas pedidas
    random_mus = np.random.randint(-semilla, semilla, [gaussianas, 2])

    # Se define la funcion de la superficie sumando funciones de gauss
    gaussSurface = lambda x, y: sumaGauss(x, y, 50.0, terreno, random_mus, gaussianas)

    # Se aleatorizan posiciones (x,y) para la cantidad de arboles que se piden
    # Y se definen traslaciones y escalados aleatorios para cada arbol
    random_positions = np.random.randint(-semilla, semilla, [arboles, 2])
    random_scales = np.random.uniform(1.0, 2.0, [arboles, 3])
    transforms = []

    for i in range(arboles):
        x = random_positions[i][0]
        y = random_positions[i][1]
        z = gaussSurface(x, y)
        traslacion = tr.translate(x, y, z)
        escala = tr.scale(random_scales[i][0], random_scales[i][1], random_scales[i][2])
        transform = tr.matmul([traslacion, escala])
        transforms.append(transform)

    cpuSurface = generateMesh(xs, ys, gaussSurface, [2/256, 49/256, 31/256])
    cpuTree = tree.tree(tree.treeFractal(ramas))

    gpuSurface = es.toGPUShape(cpuSurface)
    gpuTree = es.toGPUShape(cpuTree)

    t0 = glfw.get_time()
    camera_theta = -3 * np.pi / 4
    camera_phi = np.pi / 4
    R = semilla + 5

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta -= dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta += dt

        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS) and camera_phi > 2 * np.pi / 12:
            camera_phi -= dt

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS) and camera_phi < 5 * np.pi / 12:
            camera_phi += dt

        if (glfw.get_key(window, glfw.KEY_F) == glfw.PRESS) and R > 10:
            R -= dt * 8

        if (glfw.get_key(window, glfw.KEY_B) == glfw.PRESS) and R < semilla + 20:
            R += dt * 8

        # Setting up the view transform
        camX = R * np.sin(camera_theta) * np.sin(camera_phi)
        camY = R * np.cos(camera_theta) * np.sin(camera_phi)
        camZ = R * np.cos(camera_phi)
        viewPos = np.array([camX, camY, camZ])
        view = tr.lookAt(
            viewPos,
            np.array([0, 0, 1]),
            np.array([0, 0, 1])
        )

        # Setting up the projection transform
        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Drawing shapes
        glUseProgram(pipeline.shaderProgram)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 1, 1, 1)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 1, 1, 1)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 0.01, 0.01, 0.01)

        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), semilla + 5, 0, 30)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 1)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.1)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.01)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.001)

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        pipeline.drawShape(gpuSurface)

        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.7, 0.7, 0.7)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 0.01, 0.01, 0.01)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.01)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.05)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.0)
        for i in range(arboles):
            transform = transforms[i]
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, transform)
            pipeline.drawShape(gpuTree)

#        glUseProgram(mvpPipeline.shaderProgram)
#        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
#        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
#        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
#        mvpPipeline.drawShape(gpuAxis, GL_LINES)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()
