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


def toOBJ(archivo, vertices, indices):

    file = open(archivo, "w")
    file.write("o Tree\n")
    n = len(vertices)
    m = len(indices)

    # vertices v
    for i in range(0, n, 9):
        file.write("v {} {} {}\n".format(round(vertices[i], 4), round(vertices[i + 1], 4), round(vertices[i + 2], 4)))

    # vectores normales vn
    for j in range(6, n, 9):
        file.write("vn {} {} {}\n".format(round(vertices[j], 4), round(vertices[j + 1], 4), round(vertices[j + 2], 4)))

    # caras f
    for k in range(0, m, 3):
        vertex_1 = indices[k]
        vertex_2 = indices[k + 1]
        vertex_3 = indices[k + 2]
        file.write("f {}//{} {}//{} {}//{}\n".format(vertex_1 + 1, vertex_1 + 1, vertex_2 + 1, vertex_2 + 1,
                                                     vertex_3 + 1, vertex_3 + 1))

    file.close()
    return

def treeFractal(ramas):

    # F -> [XF]F
    rule = "F"
    for i in range(ramas):

        R = float(i) % 3
        if R == 0:
            rule += "[LF]F"
        elif R == 1:
            rule += "[RF]F"
        elif R == 2:
            rule += "[BF]F"

    return rule


# Funcion modificada de auxiliar 5
def insertar_hojas(indice, punto_centro, longitud, nTheta, nPhi):

    vertices = []
    indices = []

    dtheta = np.pi / nTheta
    dphi = 2 * np.pi / nPhi
    theta = 0
    phi = 0

    # Los puntos donde se centra la esfera
    x = punto_centro[0]
    y = punto_centro[1]
    z = punto_centro[2]

    # Radio en base a la longitud de la rama
    R = longitud * 0.5

    for i in range(nTheta):
        for j in range(nPhi):
            a = np.array([x + R * np.cos(phi) * np.sin(theta), y + R * np.sin(phi) * np.sin(theta), z + R * np.cos(theta)])
            b = np.array([x + R * np.cos(phi+dphi) * np.sin(theta), y + R * np.sin(phi+dphi) * np.sin(theta), z + R * np.cos(theta)])
            c = np.array([x + R * np.cos(phi+dphi) * np.sin(theta+dtheta), y + R * np.sin(phi+dphi) * np.sin(theta+dtheta), z + R * np.cos(theta+dtheta)])
            d = np.array([x + R * np.cos(phi) * np.sin(theta+dtheta), y + R * np.sin(phi) * np.sin(theta+dtheta), z + R * np.cos(theta+dtheta)])

            _vertex, _indices = loc_s.createColorNormalsQuadIndexation(indice, b, a, d, c, color=[77/256, 121/256, 32/256])

            vertices += _vertex
            indices  += _indices
            phi += dphi
            indice += 4
        theta += dtheta

    return (vertices, indices, indice)


def vertexData_segun_cara(cara, punto_guia, punto_sgte, color, grosor, angulo):
    '''
    Se calculan los vertices de cada una de las tres caras de la rama
    Letras segun la cara de la rama (cortada transversalmente)
    ..........B.........
    ......________......
    ......\....../......
    .....R.\..../.L.....
    ........\../........
    .........\/.........
    .......Frente.......
    '''
    vertices = []
    angulo_inicial = 0
    if cara == "L":
        angulo_inicial = 0
    elif cara == "B":
        angulo_inicial = 2*np.pi/3
    elif cara == "R":
        angulo_inicial = 4*np.pi/3
    angulo_sgte = angulo_inicial + 2*np.pi/3
    # primer vertice
    vertices += [
        # vertice
        punto_guia[0] + grosor * np.cos(angulo_inicial) * np.cos(angulo),
        punto_guia[1] + grosor * np.sin(angulo_inicial) * np.cos(angulo),
        punto_guia[2] + grosor * np.sin(angulo),
        # color
        color[0], color[1], color[2],
        # normal
        np.cos(angulo_inicial) * np.cos(angulo),
        np.sin(angulo_inicial) * np.cos(angulo),
        np.sin(angulo)
    ]
    # segundo vertice
    vertices += [
        # vertice
        punto_guia[0] + grosor * np.cos(angulo_sgte) * np.cos(angulo),
        punto_guia[1] + grosor * np.sin(angulo_sgte) * np.cos(angulo),
        punto_guia[2] + grosor * np.sin(angulo),
        # color
        color[0], color[1], color[2],
        # normal
        np.cos(angulo_sgte) * np.cos(angulo),
        np.sin(angulo_sgte) * np.cos(angulo),
        np.sin(angulo)
    ]
    grosor *= 0.7
    # tercer vertice
    vertices += [
        # vertice
        punto_sgte[0] + grosor * np.cos(angulo_sgte) * np.cos(angulo),
        punto_sgte[1] + grosor * np.sin(angulo_sgte) * np.cos(angulo),
        punto_sgte[2] + grosor * np.sin(angulo),
        # color
        color[0], color[1], color[2],
        # normal
        np.cos(angulo_sgte) * np.cos(angulo),
        np.sin(angulo_sgte) * np.cos(angulo),
        np.sin(angulo)
    ]
    # cuarto vertice
    vertices += [
        # vertice
        punto_sgte[0] + grosor * np.cos(angulo_inicial) * np.cos(angulo),
        punto_sgte[1] + grosor * np.sin(angulo_inicial) * np.cos(angulo),
        punto_sgte[2] + grosor * np.sin(angulo),
        # color
        color[0], color[1], color[2],
        # normal
        np.cos(angulo_inicial) * np.cos(angulo),
        np.sin(angulo_inicial) * np.cos(angulo),
        np.sin(angulo)
    ]
    return vertices


def tree(rule):

    vertices = []
    indices = []
    indice = 0

    # Angulo de inclinacion respecto del eje z
    angulo = 0

    # Matriz de rotacion de las ramas
    rotacion = tr.identity()

    # Un punto guia que establece desde donde comienza una rama, con cierta longitud y grosor
    punto_guia = np.array([0, 0, 0])
    grosor = 0.4
    longitud = 1

    for i in rule:

        # F indica que se construye una rama de longitud y grosor establecidos a partir del punto guia
        if i=="F":

            # Se ubica el punto siguiente para crear la rama
            mover_punto = np.matmul(rotacion, np.array([0, 0, longitud, 1]))
            punto_sgte = np.array([punto_guia[0] + mover_punto[0], punto_guia[1] + mover_punto[1], punto_guia[2] + mover_punto[2]])

            # Se calculan los vertices de las caras de las ramas
            # Cara L
            vertices += vertexData_segun_cara("L", punto_guia, punto_sgte, [0.5, 0.25, 0], grosor, angulo)
            indices += [
                indice, indice + 1, indice + 2,
                indice, indice + 2, indice + 3
            ]
            indice += 4
            # Cara B
            vertices += vertexData_segun_cara("B", punto_guia, punto_sgte, [0.5, 0.25, 0], grosor, angulo)
            indices += [
                indice, indice + 1, indice + 2,
                indice, indice + 2, indice + 3
            ]
            indice += 4
            # Cara R
            vertices += vertexData_segun_cara("R", punto_guia, punto_sgte, [0.5, 0.25, 0], grosor, angulo)
            indices += [
                indice, indice + 1, indice + 2,
                indice, indice + 2, indice + 3
            ]
            indice += 4

            # Se establece el punto siguiente como punto guia con una longitud y grosor cambiados
            punto_guia = punto_sgte
            longitud *= 0.7
            grosor *= 0.7

        # Se guardan las variables para comenzar una nueva rama
        elif i == "[":
            punto_guia_aux = punto_guia
            grosor_aux = grosor
            grosor *= 0.9
            longitud_aux = longitud
            longitud *= 3

        # Se insertan las hojas y se recuperan las variables para seguir con la rama anterior
        elif i == "]":
            N=7
            vertex_, indices_, indice_ = insertar_hojas(indice, punto_guia, longitud, N, N)
            vertices += vertex_
            indices += indices_
            indice = indice_

            punto_guia = punto_guia_aux
            grosor = grosor_aux
            longitud = longitud_aux
            angulo = 0
            rotacion = tr.identity()

        # Se gira la direccion hacia donde crece la rama dependiendo de la cara elegida
        elif i == "L":
            axis = np.array([np.cos(5 * np.pi / 6), np.sin(5 * np.pi / 6), 0])
            angulo = np.pi / 6
            rotacion = tr.rotationA(angulo, axis)
        elif i == "R":
            axis = np.array([-np.cos(7 * np.pi / 6), -np.sin(7 * np.pi / 6), 0])
            angulo = np.pi / 6
            rotacion = tr.rotationA(angulo, axis)
        elif i == "B":
            axis = np.array([np.cos(3 * np.pi / 2), np.sin(3 * np.pi / 2), 0])
            angulo = np.pi / 6
            rotacion = tr.rotationA(angulo, axis)

    return bs.Shape(vertices, indices)


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Tree", None, None)

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

    cpuTree = tree(treeFractal(int(sys.argv[2])))

    toOBJ(sys.argv[1], cpuTree.vertices, cpuTree.indices)
    gpuTree = es.toGPUShape(cpuTree)

    t0 = glfw.get_time()
    camera_theta = -3 * np.pi / 4

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta -= 2 * dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta += 2 * dt

        # Setting up the view transform
        R = 8
        camX = R * np.sin(camera_theta)
        camY = R * np.cos(camera_theta)
        viewPos = np.array([camX, camY, 6])
        view = tr.lookAt(
            viewPos,
            np.array([0, 0, 3]),
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

        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.8, 0.8, 0.8)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.4, 0.4, 0.4)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 0.01, 0.01, 0.01)

        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 8, 0, 2)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 1)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.01)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.005)

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        pipeline.drawShape(gpuTree)

#        glUseProgram(mvpPipeline.shaderProgram)
#        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
#        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
#        glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
#        mvpPipeline.drawShape(gpuAxis, GL_LINES)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()
