'''
Definition of multiple shaders
'''

from render import V3, dot, BLACK, sum
from utils import color
from math import sin

def gourad(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']


    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)

    b *= intesity
    g *= intesity
    r *= intesity


    if intesity > 0:
        return r, g, b
    else:
        return 0, 0, 0


def toon(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']


    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)


    if intesity > 0 and intesity < 0.2:
        b *= 0.2
        g *= 0.2
        r *= 0.2
    elif intesity >= 0.2 and intesity < 0.4:
        b *= 0.4
        g *= 0.4
        r *= 0.4
    elif intesity >= 0.4 and intesity < 0.6:
        b *= 0.6
        g *= 0.6
        r *= 0.6
    elif intesity >= 0.6 and intesity < 0.8:
        b *= 0.8
        g *= 0.8
        r *= 0.8    
    elif intesity >= 0.8 and intesity < 1:
        b *= 1
        g *= 1
        r *= 1
                           
    if intesity > 0:
        return r, g, b
    else:
        return 0, 0, 0


def toon_mod(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']


    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)

    b *= intesity
    g *= intesity
    r *= intesity


    if intesity > 0:
        return round(r,1), round(g,1), round(b,1)
    else:
        return 0, 0, 0




def outline(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']


    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)

    b *= intesity
    g *= intesity
    r *= intesity

    if intesity > 0 and intesity <= 0.2:
        return 1,1,1


    if intesity > 0:
        return r, g, b
    else:
        return 0, 0, 0
