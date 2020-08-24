from render import Render, V2, V3

from obj import Obj

from texture import Texture

from shaders import gourad, toon, outline, toon_mod

from utils import color

r = Render(1000, 1000)

posModel = V3(0,0,-5)

#r.lookAt(posModel, V3(2,2,0))

r.active_texture = Texture('./models/horse.bmp')
r.active_shader = toon_mod

r.loadModel('./models/horse.obj', posModel, V3(1,1,1), V3(0,0,0))

r.glFinish('output.bmp')