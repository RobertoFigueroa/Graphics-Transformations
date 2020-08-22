from render import Render, V2, V3

from obj import Obj

from texture import Texture

from shaders import gourad, toon, outline, toon_mod

from utils import color

r = Render()
r.glCreateWindow(1000,1000)
r.glClear()

r.active_texture = Texture('./models/model.bmp')
r.active_shader = outline

#r.light = V3(1,0,0)

r.loadModel('./models/model.obj', V3(500,500,0), V3(150,150,150))

# r.active_shader = outline

# r.loadModel('./models/model.obj', V3(500,500,0), V3(150,150,150))

# r.active_shader = toon_mod

# r.loadModel('./models/model.obj', V3(750,500,0), V3(150,150,150))


r.glFinish('output2.bmp')
