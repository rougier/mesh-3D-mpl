# -----------------------------------------------------------------------------
# Graphic Server Protocol (GSP)
# Copyright 2023 Vispy Development Team - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
# import matplotlib
# matplotlib.use("module://mplcairo.macosx")

import types
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.transforms as mtransforms
from matplotlib.patches import PathPatch

from matplotlib.backend_bases import GraphicsContextBase, RendererBase

class GC(GraphicsContextBase):
    def __init__(self):
        super().__init__()
        self._antialias = False
def custom_new_gc(self):
    return GC()
RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)


def obj_read(filename):
    """
    Read a wavefront filename and returns vertices, texcoords and
    respective indices for faces and texcoords
    """
    
    V, T, N, Vi, Ti, Ni = [], [], [], [], [], []
    with open(filename) as f:
       for line in f.readlines():
           if line.startswith('#'):
               continue
           values = line.split()
           if not values:
               continue
           if values[0] == 'v':
               V.append([float(x) for x in values[1:4]])
           elif values[0] == 'vt':
               T.append([float(x) for x in values[1:3]])
           elif values[0] == 'vn':
               N.append([float(x) for x in values[1:4]])
           elif values[0] == 'f' :
               Vi.append([int(indices.split('/')[0]) for indices in values[1:]])
               Ti.append([int(indices.split('/')[1]) for indices in values[1:]])
               Ni.append([int(indices.split('/')[2]) for indices in values[1:]])
    return np.array(V), np.array(T), np.array(Vi)-1, np.array(Ti)-1


def warp(T1, T2):
    """
    Return an affine transform that warp triangle T1 into triangle
    T2.

    Raises
    ------

    `LinAlgError` if T1 or T2 are degenerated triangles
    """
    
    T1 = np.c_[np.array(T1), np.ones(3)]
    T2 = np.c_[np.array(T2), np.ones(3)]
    M = np.linalg.inv(T1) @ T2
    return mtransforms.Affine2D(M.T)

def textured_triangle(ax, T, UV, texture, intensity, interpolation="none"):
    """
    Parameters
    ----------
    T : (3,2) np.ndarray
      Positions of the triangle vertices
    UV : (3,2) np.ndarray
      UV coordinates of the triangle vertices
    texture: 
      Image to use for texture
    """

    w,h = texture.shape[:2]
    Z = UV*(w,h)
    xmin, xmax = int(np.floor(Z[:,0].min())), int(np.ceil(Z[:,0].max()))
    ymin, ymax = int(np.floor(Z[:,1].min())), int(np.ceil(Z[:,1].max()))
    texture = (texture[ymin:ymax, xmin:xmax,:] * intensity).astype(np.uint8)
    extent = xmin/w, xmax/w, ymin/h, ymax/h
    transform = warp (UV,T) + ax.transData
    path =  Path([UV[0], UV[1], UV[2], UV[0]], closed=True)
    im = ax.imshow(texture, interpolation=interpolation, origin='lower',
                   extent=extent, transform=transform, clip_path=(path,transform))

fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(-1,1), ax.set_xticks([])
ax.set_ylim(-1,1), ax.set_yticks([])

V, UV, Vi, UVi = obj_read("head.obj")
texture = iio.imread("uv-grid.png")[::-1,::1,:3]
V, UV = V[Vi], UV[UVi]

N = np.cross(V[:,2]-V[:,0], V[:,1]-V[:,0])
N = N / np.linalg.norm(N,axis=1).reshape(len(N),1)
L = np.dot(N, (0,0,-1))

I = np.argsort(V[:,:,2].mean(axis=1))
V = V[I][...,:2]
UV = UV[I][...,:2]
L = L[I]

for v, uv, l in zip(V, UV, L):
    if l > 0:
        try:
            textured_triangle(ax, v, uv, texture, (l+1)/2)
        except np.linalg.LinAlgError:
            pass

# plt.savefig("head.pdf")
# plt.savefig("head.png")
# plt.savefig("head.svg")
plt.show()

