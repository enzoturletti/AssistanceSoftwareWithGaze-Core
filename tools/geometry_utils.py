import numpy as np

class GeometryUtils():
    
    @staticmethod
    def ray_plan_intersection(origin,direction):
        p1 = np.array([0,0,0])
        p2 = np.array([1,0,0])
        p3 = np.array([0,1,0])

        v1 = p3 - p1
        v2 = p2 - p1

        cp = np.cross(v1, v2)
        a, b, c = cp

        d = np.dot(cp, p3)

        x = np.arange(-1,1,0.1)      
        y = np.arange(-1,1,0.1)      
        X, Y = np.meshgrid(x, y)

        Z = (d - a * X - b * Y) / c

        x0 = origin[0]*1000
        y0 = origin[1]*1000    
        z0 = origin[2]*1000

        mx = direction[0]
        my = direction[1]
        mz = direction[2]

        t_intersection = (d-a*x0-b*y0-c*z0)/(a*mx+b*my+c*mz)

        return  origin*1000 + t_intersection*direction