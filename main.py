from render import Render, V2, V3

from obj import Obj

from texture import Texture

from shaders import gourad, toon, outline, toon_mod

from utils import color

r = Render()
r.glCreateWindow(1000,1000)
r.glClear()

r.active_texture = Texture('./models/horse.bmp')
r.active_shader = toon_mod

r.loadModel('./models/horse.obj', V3(500,300,0), V3(2,2,2), V3(0,90,0))



r.glFinish('output.bmp')
