
import numpy as np
from stl import mesh

def flip_normals(m):
    m.vectors=m.vectors[:,[0,2,1],:]
    return m

def elevation_mesh(x,y,z):
    P=np.stack((x,y,z))#Format [coord,x,y]
    F1=np.stack((P[:,0:-1,0:-1],P[:,1:,0:-1],P[:,0:-1,1:]))#Format [point, coord, x,y]
    F2=np.stack((P[:,0:-1,1:],P[:,1:,0:-1],P[:,1:,1:]))#Format [point, coord, x,y]
    F1=F1.reshape(*F1.shape[:2],-1)#Format [point, coord, face]
    F2=F2.reshape(*F2.shape[:2],-1)#Format [point, coord, face]
    F=np.stack((F1,F2), axis=-1).reshape(*F1.shape[:2],-1) #Format [point, coord, face]
    nfaces=2*(z.shape[0]-1)*(z.shape[1]-1)
    mymesh=mesh.Mesh(np.zeros(nfaces, dtype=mesh.Mesh.dtype))
    mymesh.vectors=F.transpose(2,0,1)
    return mymesh

def regular_elevation_mesh(z):
    x,y=np.meshgrid(np.arange(z.shape[0]), np.arange(z.shape[1]), indexing='ij')#Format [x,z]
    return elevation_mesh(x,y,z)

def elevation_body(x,y,z,surface):
    top=elevation_mesh(x,y,z)
    def side_plate(x,y,surface):
        xp=np.stack((np.ones_like(x)*surface,x))
        yp=np.stack((y,y))
        zp=np.zeros_like(xp)
        return elevation_mesh(xp,yp,zp)

    front=side_plate(z[:,0],x[:,0],surface)
    front.rotate([0,1,0],np.deg2rad(90))
    front.rotate([0,0,1], np.deg2rad(90))
    flip_normals(front)
    front.y+=y[0,0]
    back=side_plate(z[:,-1],x[:,-1],surface)
    back.rotate([0,1,0],np.deg2rad(90))
    back.rotate([0,0,1], np.deg2rad(90))
    back.y+=y[0,-1]
    left=side_plate(z[0,:],y[0,:],surface)
    left.rotate([0,1,0], np.deg2rad(90))
    left.x+=x[0,0]
    right=side_plate(z[-1,:],y[-1,:],surface)
    right.rotate([0,1,0],np.deg2rad(90))
    flip_normals(right)
    right.x+=x[-1,0]
    surface=elevation_mesh(x,y,np.ones_like(z)*surface)
    flip_normals(surface)
    return mesh.Mesh(np.concatenate([top.data,front.data,back.data,left.data,right.data, surface.data]))

def regular_elevation_body(z,surface):
    x,y=np.meshgrid(np.arange(z.shape[0]), np.arange(z.shape[1]), indexing='ij')#Format [x,z]
    return elevation_body(x,y,z,surface)


def rearrange_fourshift(a00, a01, a10, a11):
    """Given four arrays of shape (n,m), return an array of shape (2n,2m), where the four input arrays are inserted in the form:
    [a00[0,0], a01[0,0], ...]
    [a10[0,0], a11[0,0], ...]
    [   ...  ,    ...  , ...]

    Parameters
    ----------
    a00, a01, a10, a11: np.ndarray
        Arrays of shape (n,m)

    Returns
    -------
    np.ndarray
        Array of shape (2n,2m)
    """
    result=np.zeros((2*a00.shape[0], 2*a00.shape[1]))
    result[::2,::2]=a00
    result[::2,1::2]=a01
    result[1::2,::2]=a10
    result[1::2,1::2]=a11
    return result

def transform_to_columnar_elevation(x,y,z, dx=None, dy=None):
    """Transform an elevation grid into a columnar elevation grid, where each point is split into four corner points of a cell, and the elevation is the same for all four points.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of shape (n,m)
    y : np.ndarray
        Y coordinates of shape (n,m)
    z : np.ndarray
        Z coordinates of shape (n,m)
    dx : float, optional
        Shift of the corner points relative to the original point in x direction, by default None
    dy : float, optional
        Shift of the corner poitns relative to the original point in y direction, by default None

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        The new x,y,z coordinates of shape (2n,2m)
    """
    if dx is None:
        dx=(x[1,0]-x[0,0])/2.0
    if dy is None: 
        dy=(y[0,1]-y[0,0])/2.0

    xnew=rearrange_fourshift(x-dx,x-dx,x+dx,x+dx)
    ynew=rearrange_fourshift(y-dy,y+dy,y-dy,y+dy)
    znew=rearrange_fourshift(z,z,z,z)
    return xnew, ynew, znew

#Tests#####################################################
#%%
# samples=100
# wave=np.sin(np.linspace(0,8*np.pi,samples))*np.linspace(0.1,2.5,samples)
# data=np.repeat(wave[:,np.newaxis],10, axis=1)
# #%%
# rng=np.random.default_rng(1234)
# data=rng.random((40,40))
