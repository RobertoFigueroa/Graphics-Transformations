from data_types_module import dword, word, char
from utils import color
from obj import Obj
from collections import namedtuple
from math import sin, cos, tan
import numpy


BLACK = color(0, 0, 0)
WHITE = color(1, 1, 1)
RED = color(1, 0, 0)
PI = 3.141592653589793

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z','w'])


def sum(v0, v1):
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
    return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
	
def length(v0):
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5


def norm(v0):
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)


def bbox(*vertices):
    xs = [ vertex.x for vertex in vertices ]
    ys = [ vertex.y for vertex in vertices ]

    xs.sort()
    ys.sort()

    xmin = xs[0]
    xmax = xs[-1]
    ymin = ys[0]
    ymax = ys[-1]

    return xmin, xmax, ymin, ymax

def cross(v1, v2):
    return V3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x,
    )

def baryCoords(A, B, C, P):
    # u es para la A, v es para B, w para C
    try:
        u = ( ((B.y - C.y)*(P.x - C.x) + (C.x - B.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        v = ( ((C.y - A.y)*(P.x - C.x) + (A.x - C.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w

def _deg2rad(degrees):
	radians = degrees * (PI/180)
	return radians

#only for square matrix
def matrixMul(matrix1, matrix2, isVector=False):
	matrix = [[0 for x in range(len(matrix1))] for y in range(len(matrix2[0]))]
	for i in range(len(matrix1)):
		for j in range(len(matrix2[0])):
			for k in range(len(matrix2)):
				matrix[i][j] += matrix1[i][k] * matrix2[k][j]
	return matrix

def matXvect(matrix1, vector):
	matrix = [[0 for x in range(len(matrix1))] for y in range(1)]
	for i in range(len(matrix1)):
		for j in range(1):
			for k in range(len(vector)):
				matrix[0][i] += matrix1[i][k] * vector[k]
	return matrix


def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                print("MATRIX NOT INVERTIBLE")
                return -1
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def inverse(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return ret

class Render(object):

	#constructor
	def __init__(self, width, height):
		self.framebuffer = []
		self.curr_color = WHITE
		self.light = V3(0,0,1)
		self.active_texture = None
		self.active_shader = None
		self.camPosition = V3(0,0,0)
		self.camRotation = V3(0,0,0)
		self.glCreateWindow(width, height)
		self.glClear()
		self.createViewMatrix()
		self.createProjectionMatrix()

	def createViewMatrix(self):
		camMatrix = self.createObjectMatrix(translate=self.camPosition, rotate=self.camRotation)
		self.viewMatrix = inverse(camMatrix)

	def lookAt(self, eye, camPosition = V3(0,0,0)):
		forward = sub(camPosition, eye)
		forward = norm(forward)
		
		right = cross(V3(0,1,0), forward)
		right = norm(right)
		
		up = cross(forward, right)
		up = norm(up)

		camMatrix = [[right[0],up[0],forward[0],camPosition.x],
					[right[1],up[1],forward[1],camPosition.y],
					[right[2],up[2],forward[2],camPosition.z],
					[0,0,0,1]]
		
		self.viewMatrix = inverse(camMatrix)


	def createProjectionMatrix(self, n=0.1, f=1000, fov= 60):

		t = tan(_deg2rad(fov / 2)) * n
		r = t * self.viewportWidth / self.viewportHeight

		self.projectionMatrix = [[n/r,0,0,0],
								[0,n/t,0,0],
								[0,0,-(f+n)/(f-n),-(2*f*n)/(f - n)],
								[0,0,-1,0]]


	def glCreateWindow(self, width, height):
		#width and height for the framebuffer
		self.width = width
		self.height = height
		self.glViewport(0,0,width,height)

	def glInit(self):
		self.curr_color = BLACK

	def glViewport(self, x, y, width, height):
		self.viewportX = x
		self.viewportY = y
		self.viewportWidth = width
		self.viewportHeight = height

		self.viewportMatrix = 	[[width/2,0,0,x + width/2],
								[0,height/2,0,y +height/2],
								[0,0,0.5,0.5],
								[0,0,0,1]]

	def glClear(self):
		self.framebuffer = [[BLACK for x in range(
		    self.width)] for y in range(self.height)]
		
		#Zbuffer (buffer de profundidad)
		self.zbuffer = [ [ float('inf') for x in range(self.width)] for y in range(self.height) ]
		

	def glClearColor(self, r, g, b):
		clearColor = color(
				round(r * 255),
				round(g * 255),
				round(b * 255)
			)

		self.framebuffer = [[clearColor for x in range(
		    self.width)] for y in range(self.height)]

	def glVertex(self, x, y):
		#las funciones fueron obtenidas de https://www.khronos.org/registry/OpenGL-Refpages/es2.0/xhtml/glViewport.xml
		X = round((x+1) * (self.viewportWidth/2) + self.viewportX)
		Y = round((y+1) * (self.viewportHeight/2) + self.viewportY)
		self.point(X, Y)
		

	def glVertex_coord(self, x, y, color= None):

		if x < self.viewportX or x >= self.viewportX + self.viewportWidth or  y < self.viewportY or y >= self.viewportY + self.viewportHeight:
			return 

		if x >= self.width or x < 0 or y >= self.height or y < 0:
			return
		try:
			self.framebuffer[y][x] = color or self.curr_color
		except:
			pass

	def glColor(self, r, g, b):
		self.curr_color = color(round(r * 255), round(g * 255), round(b * 255))

	def point(self, x, y):
		self.framebuffer[x][y] = self.curr_color

	def glFinish(self, filename):
		archivo = open(filename, 'wb')

		# File header 14 bytes
		archivo.write(char("B"))
		archivo.write(char("M"))
		archivo.write(dword(14+40+self.width*self.height))
		archivo.write(dword(0))
		archivo.write(dword(14+40))

		#Image Header 40 bytes
		archivo.write(dword(40))
		archivo.write(dword(self.width))
		archivo.write(dword(self.height))
		archivo.write(word(1))
		archivo.write(word(24))
		archivo.write(dword(0))
		archivo.write(dword(self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))

		#Pixeles, 3 bytes cada uno

		for x in range(self.height):
			for y in range(self.width):
				archivo.write(self.framebuffer[x][y])

		#Close file
		archivo.close()

	#class implementation
	def glLine(self, x0, y0, x1, y1):
		x0 = round(( x0 + 1) * (self.viewportWidth  / 2 ) + self.viewportX)
		x1 = round(( x1 + 1) * (self.viewportWidth  / 2 ) + self.viewportX)
		y0 = round(( y0 + 1) * (self.viewportHeight / 2 ) + self.viewportY)
		y1 = round(( y1 + 1) * (self.viewportHeight / 2 ) + self.viewportY)

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		steep = dy > dx

		if steep:
			x0, y0 = y0, x0
			x1, y1 = y1, x1

		if x0 > x1:
			x0, x1 = x1, x0
			y0, y1 = y1, y0

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		offset = 0
		limit = 0.5

		m = dy/dx
		y = y0

		for x in range(x0, x1 + 1):
			if steep:
				self.glVertex_coord(y, x)
			else:
				self.glVertex_coord(x, y)

			offset += m
			if offset >= limit:
				y += 1 if y0 < y1 else -1
				limit += 1

		
	def glLine_coord(self, x0, y0, x1, y1): # Window coordinates

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		steep = dy > dx

		if steep:
			x0, y0 = y0, x0
			x1, y1 = y1, x1

		if x0 > x1:
			x0, x1 = x1, x0
			y0, y1 = y1, y0

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		offset = 0
		limit = 0.5
		
		try:
			m = dy/dx
		except ZeroDivisionError:
			pass
		else:
			y = y0

			for x in range(x0, x1 + 1):
				if steep:
					self.glVertex_coord(y, x)
				else:
					self.glVertex_coord(x, y)

				offset += m
				if offset >= limit:
					y += 1 if y0 < y1 else -1
					limit += 1

	def transform(self, vertex, vMatrix):

		augVertex= [vertex[0], vertex[1], vertex[2], 1]
		transVertex = matXvect(self.viewportMatrix, matXvect(self.projectionMatrix, matXvect(self.viewMatrix, matXvect(vMatrix, augVertex)[0])[0])[0])

		transVertex = 	V3(transVertex[0][0] / transVertex[0][3],
						transVertex[0][1] / transVertex[0][3],
						transVertex[0][2] / transVertex[0][3])

		print(transVertex)		
		return transVertex

	def dirTransform(self, vertex, vMatrix):

		augVertex= [vertex[0], vertex[1], vertex[2], 0]
		transVertex = matXvect(vMatrix, augVertex)
		transVertex = 	V3(transVertex[0][0],
						transVertex[0][1],
						transVertex[0][2])

		
		return transVertex

	def createObjectMatrix(self, translate = V3(0,0,0), scale = V3(1,1,1), rotate=V3(0,0,0)):

		translateMatrix = [[1, 0, 0, translate.x],
							[0, 1, 0, translate.y],
							[0, 0, 1, translate.z],
							[0, 0, 0, 1]]

		scaleMatrix = [[scale.x, 0, 0, 0],
						[0, scale.y, 0, 0],
						[0, 0, scale.z, 0],
						[0, 0, 0, 1]]
								
		rotationMatrix = self.createRotationMatrix(rotate)

		return matrixMul(matrixMul(translateMatrix, rotationMatrix), scaleMatrix)

	def createRotationMatrix(self, rotate=V3(0,0,0)):

		pitch = _deg2rad(rotate.x)
		yaw = _deg2rad(rotate.y)
		roll = _deg2rad(rotate.z)

		rotationX = [[1, 0, 0, 0],
					[0, cos(pitch),-sin(pitch), 0],
					[0, sin(pitch), cos(pitch), 0],
					[0, 0, 0, 1]]

		rotationY = [[cos(yaw), 0, sin(yaw), 0],
					[0, 1, 0, 0],
					[-sin(yaw), 0, cos(yaw), 0],
					[0, 0, 0, 1]]

		rotationZ = [[cos(roll),-sin(roll), 0, 0],
					[sin(roll), cos(roll), 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1]]

		return matrixMul(matrixMul(rotationX, rotationY), rotationZ)	

	def loadModel(self, filename, translate=V3(0,0,0), scale=V3(1,1,1), rotate=V3(0,0,0), isWireframe = False):
		model = Obj(filename)

		modelMatrix = self.createObjectMatrix(translate, scale, rotate)

		rotationMatrix = self.createRotationMatrix(rotate)

		for face in model.faces:

			vertCount = len(face)

			if isWireframe:
				for vert in range(vertCount):
					
					v0 = model.vertices[ face[vert][0] - 1 ]
					v1 = model.vertices[ face[(vert + 1) % vertCount][0] - 1]

					x0 = round(v0[0] * scale[0]  + translate[0])
					y0 = round(v0[1] * scale[1]  + translate[1])
					x1 = round(v1[0] * scale[0]  + translate[0])
					y1 = round(v1[1] * scale[1]  + translate[1])

					self.glLine_coord(x0, y0, x1, y1)
			else:
				v0 = model.vertices[ face[0][0] - 1 ]
				v1 = model.vertices[ face[1][0] - 1 ]
				v2 = model.vertices[ face[2][0] - 1 ]
				if vertCount > 3:
					v3 = model.vertices[ face[3][0] - 1 ]

				v0 = self.transform(v0,modelMatrix)
				v1 = self.transform(v1,modelMatrix)
				v2 = self.transform(v2,modelMatrix)
				if vertCount > 3:
					v3 = self.transform(v3,modelMatrix)

				if self.active_texture:
					vt0 = model.texcoords[face[0][1] - 1]
					vt1 = model.texcoords[face[1][1] - 1]
					vt2 = model.texcoords[face[2][1] - 1]
					vt0 = V2(vt0[0], vt0[1])
					vt1 = V2(vt1[0], vt1[1])
					vt2 = V2(vt2[0], vt2[1])
					if vertCount > 3:
						vt3 = model.texcoords[face[3][1] - 1]
						vt3 = V2(vt3[0], vt3[1])
				else:
					vt0 = V2(0,0) 
					vt1 = V2(0,0) 
					vt2 = V2(0,0) 
					vt3 = V2(0,0)
				
				vn0 = model.normals[face[0][2] - 1]
				vn1 = model.normals[face[1][2] - 1]
				vn2 = model.normals[face[2][2] - 1]

				vn0 = self.dirTransform(vn0, rotationMatrix)
				vn1 = self.dirTransform(vn1, rotationMatrix)
				vn2 = self.dirTransform(vn2, rotationMatrix)

				if vertCount > 3:
					vn3 = model.normals[face[3][2] -1]
					vn3 = self.dirTransform(vn3, rotationMatrix)
				


				self.triangle_bc(v0,v1,v2, texcoords = (vt0,vt1,vt2), normals= (vn0, vn1, vn2))
				if vertCount > 3: #asumamos que 4, un cuadrado
					self.triangle_bc(v0,v2,v3, texcoords = (vt0,vt2,vt3), normals=(vn1, vn2, vn3))

	def drawPolygons(self, points):
		
		vertices = len(points)

		for v in range(vertices):
			
			x0 = points[v][0]
			y0 = points[v][1]

			x1 = points[(v +1)% vertices][0]
			y1 = points[(v +1) % vertices][1]

			self.glLine_coord(x0,y0,x1,y1)
			

	#Reference: https://handwiki.org/wiki/Even%E2%80%93odd_rule
	def is_point_in_path(self, x, y, poly):
		num = len(poly)
		i = 0
		j = num - 1
		c = False
		for i in range(num):
			if ((poly[i][1] > y) != (poly[j][1] > y)) and \
					(x < poly[i][0] + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
									(poly[j][1] - poly[i][1])):
				c = not c
			j = i
		return c


	def fillPoly(self, poly, color):
		for x in range(self.width):
			for y in range(self.height):
				if self.is_point_in_path(x,y,poly):
					self.framebuffer[y][x] = color
		
	def glZBuffer(self, filename):
		archivo = open(filename, 'wb')

		# File header 14 bytes
		archivo.write(bytes('B'.encode('ascii')))
		archivo.write(bytes('M'.encode('ascii')))
		archivo.write(dword(14 + 40 + self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(14 + 40))

		# Image Header 40 bytes
		archivo.write(dword(40))
		archivo.write(dword(self.width))
		archivo.write(dword(self.height))
		archivo.write(word(1))
		archivo.write(word(24))
		archivo.write(dword(0))
		archivo.write(dword(self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))

		# Minimo y el maximo
		minZ = float('inf')
		maxZ = -float('inf')
		for x in range(self.height):
			for y in range(self.width):
				if self.zbuffer[x][y] != -float('inf'):
					if self.zbuffer[x][y] < minZ:
						minZ = self.zbuffer[x][y]

					if self.zbuffer[x][y] > maxZ:
						maxZ = self.zbuffer[x][y]

		for x in range(self.height):
			for y in range(self.width):
				depth = self.zbuffer[x][y]
				if depth == -float('inf'):
					depth = minZ
				depth = (depth - minZ) / (maxZ - minZ)
				archivo.write(color(depth,depth,depth))

		archivo.close()


	def triangle(self, A, B, C, color = None):
        
		def flatBottomTriangle(v1,v2,v3):
            #self.drawPoly([v1,v2,v3], color)
			for y in range(v1.y, v3.y + 1):
				xi = round( v1.x + (v3.x - v1.x)/(v3.y - v1.y) * (y - v1.y))
				xf = round( v2.x + (v3.x - v2.x)/(v3.y - v2.y) * (y - v2.y))

				if xi > xf:
					xi, xf = xf, xi

				for x in range(xi, xf + 1):
					self.glVertex_coord(x,y, color or self.curr_color)

		def flatTopTriangle(v1,v2,v3):
			for y in range(v1.y, v3.y + 1):
				xi = round( v2.x + (v2.x - v1.x)/(v2.y - v1.y) * (y - v2.y))
				xf = round( v3.x + (v3.x - v1.x)/(v3.y - v1.y) * (y - v3.y))

				if xi > xf:
					xi, xf = xf, xi

				for x in range(xi, xf + 1):
					self.glVertex_coord(x,y, color or self.curr_color)

        # A.y <= B.y <= Cy
		if A.y > B.y:
			A, B = B, A
		if A.y > C.y:
			A, C = C, A
		if B.y > C.y:
			B, C = C, B

		if A.y == C.y:
			return

		if A.y == B.y: #En caso de la parte de abajo sea plana
			flatBottomTriangle(A,B,C)
		elif B.y == C.y: #En caso de que la parte de arriba sea plana
			flatTopTriangle(A,B,C)
		else: #En cualquier otro caso
			# y - y1 = m * (x - x1)
			# B.y - A.y = (C.y - A.y)/(C.x - A.x) * (D.x - A.x)
			# Resolviendo para D.x
			x4 = A.x + (C.x - A.x)/(C.y - A.y) * (B.y - A.y)
			D = V2( round(x4), B.y)
			flatBottomTriangle(D,B,C)
			flatTopTriangle(A,B,D)

    #Barycentric Coordinates
	def triangle_bc(self, A, B, C, texcoords = (), normals = (), _color = None):
		#bounding box
		minX = round(min(A.x, B.x, C.x))
		minY = round(min(A.y, B.y, C.y))
		maxX = round(max(A.x, B.x, C.x))
		maxY = round(max(A.y, B.y, C.y))

		for x in range(minX, maxX + 1):
			for y in range(minY, maxY + 1):
				if x >= self.width or x < 0 or y >= self.height or y < 0:
					continue

				u, v, w = baryCoords(A, B, C, V2(x, y))

				if u >= 0 and v >= 0 and w >= 0:

					z = A.z * u + B.z * v + C.z * w
					if z < self.zbuffer[y][x] and z<=1 and z>=-1:
						
						
						r, g, b = self.active_shader(
							self,
							baryCoords=(u,v,w),
							texCoords = texcoords,
							normals = normals,
							color= _color or self.curr_color)

						self.glVertex_coord(x, y, color(r,g,b))
						self.zbuffer[y][x] = z