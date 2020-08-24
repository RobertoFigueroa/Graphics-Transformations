from render import Render, V2, V3

from obj import Obj

from texture import Texture

from shaders import gourad, toon, outline, toon_mod, unlit

from utils import color

r = Render(1000, 1000)

r.light = V3(0,0,1)

posModel = V3(0,0,-20)

#low-angle-shot
# r.lookAt(posModel, V3(0,-8,0))

#medium shot
# r.lookAt(posModel, V3(0,0,0))
# posModel = V3(0,0,-30)


# highangle
r.lookAt(posModel, V3(0,15,0))
posModel = V3(0,5,-20)

#dutch angle
# r.camRotation = V3(0,0,15)
# r.createViewMatrix()
# r.createProjectionMatrix()
# posModel = V3(0,0,-20)

r.active_texture = Texture('./models/heli.bmp')
r.active_shader = unlit

r.loadModel('./models/heli.obj', posModel, V3(0.1,0.1,0.1), V3(0,0,0))

r.glFinish('highangle.bmp')