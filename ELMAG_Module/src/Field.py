from numba import njit
import numpy as np
import matplotlib.pyplot as plt


class Field():
    '''
    # Critical
    !!NEEDS NUMPY AND NUMBA TO FUNCTION!!\n
    
    # General Info
    Class for calculating and plotting electromagnetic fields and potentials in 3d. Requires numba and numpy\n\n
    
    # Functions
    
    ## Efield, Epot
    Field and potential from point charge\n
    efieldLine, epotLine - Field and potential from line charge parallel to x, y or z axis. Can be placed anywhere in 3D\n
    
    
    ## EfieldCircle, EpotCircle 
    Field and potential from circle charge in origin\n
    
    ## BfieldLine
    Field from line current parallel to x, y or z axis. Can be placed anywhere in 3D\n
    
    ## BfieldCircle
    Field from circular current in origin\n\n
    
    ## PlotVector
    Plots vector field. Customize colorscheme, density, and figsize\n
    
    ## PlotContour
    Plots vector field. Customize colorscheme, levels, norm and figsize\n
    
    ## PlotCircle
    Plots circle just using radius
        
    # See example code below:

    ## Single Point Charge
        
    >>> L = 2
    >>> N = 10
    >>> Q = [1.0]
    >>> r_Q = np.array([[0.0, 0.0, 0.0]])
    >>> plane = 'xy'
    
    >>> rx, ry, Ex, Ey = Field.CalculateEfield(L, N, Q, r_Q, plane)
    >>> Field.PlotVector(rx, ry, Ex, Ey, 'quiver', show = True)
    
    >>> N = 100
    >>> rx, ry, V = Field.CalculateEpot(L, N, Q, r_Q, plane)
    >>> Field.PlotContour(rx, ry, V, show=True)
    
    
    ## Double Point Charge Example     
    
    >>> L = 2
    >>> N = 100
    >>> Q = [2.0, -2.0]
    >>> r_Q = [-1, 1]
    >>> r_Q = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> plane = 'xy'
    
    >>> rx, ry, Ex, Ey = Field.CalculateEfield(L, N, Q, r_Q, plane)
    >>> Field.PlotVector(rx, ry, Ex, Ey, 'stream', show = True, broken_streamlines = False)
    
    ## Double Line Charge Example       
    
    >>> L = 2
    >>> N = 50
    >>> line_charges = [-1, 1]
    >>> line_lengths = [1, 1]
    >>> line_center_coords = [[0, 0, -1], [0, 0, 1]]
    >>> axis = ['x', 'x']
    >>> plane = 'xz'
    
    >>> rx, rz, Ex, Ez = Field.CalculateEfieldLine(L, N, line_charges, line_lengths, line_center_coords, axis, plane)
    >>> rx, rz, V = Field.CalculateEpotLine(L, N, line_charges, line_lengths, line_center_coords, axis, plane)
    
    >>> Field.PlotVector(rx, rz, Ex, Ez, 'stream', broken_streamlines = False, show = True)
    >>> Field.PlotContour(rx, rz, V, show = True, norm = 'linear') 
    
    
    ## Circular Charge Example
    
    >>> L = 5
    >>> N = 100
    >>> circle_charge = [5]
    >>> radius = [2]
    >>> plane = 'yz'
    >>> plane_circles = ['yz']
    >>> ry, rz, Ey, Ez = Field.CalculateEfieldCircle(L, N, circle_charge, radius, plane, plane_circles)
    >>> Field.PlotVector(ry, rz, Ey, Ez, 'stream', show = False, equal = True)

    >>> t = np.linspace(0, 2*np.pi, 100)
    >>> plt.plot(radius[0]*np.cos(t), radius[0]*np.sin(t))
    >>> plt.show()

    >>> N = 500
    >>> ry, rz, V = Field.CalculateEpotCircle(L, N, circle_charge, radius, plane, plane_circles)
    >>> Field.PlotContour(ry, rz, V, show = False, equal = True)
    
    >>> t = np.linspace(0, 2*np.pi, 100)
    >>> plt.plot(radius[0]*np.cos(t), radius[0]*np.sin(t))
    >>> plt.show()
    
    
    ## Line Current Example
    
    >>> L = 5
    >>> N = 24
    >>> line_currents = [5]
    >>> line_lengths = [1]
    >>> line_center_coords = [[0.0, 0.0, 0.0]]
    >>> axis = ['x']
    >>> plane = 'yz'
    
    >>> rx, rz, Bx, Bz = Field.CalculateBfieldLine(L, N, line_currents, line_lengths, line_center_coords, axis, plane)
    
    >>> Field.PlotVector(rx, rz, Bx, Bz, 'quiver', title = 'Magnetic Field from Lin e Current', show = True)

    ## Circular Current Example

    >>> L = 8
    >>> N = 40
    >>> circle_currents = [5]
    >>> radii = [5]
    >>> plane = 'xz'
    >>> circle_planes = ['xy']
    >>> rx, rz, Bx, Bz = Field.CalculateBfieldCircle(L, N, circle_currents, radii, plane, circle_planes)
    
    >>> Field.PlotVector(rx, rz, Bx, Bz, 'stream', broken_streamlines=False, show = True, cmap = 'inferno', density = .5)    
    '''
    
    @staticmethod
    @njit
    def Efield(r: np.ndarray, particle_pos: np.ndarray, q: float, eps = 8.854187817E-12) -> np.ndarray:
        '''
        Calculates electric field from point charge\n
        
        ## Input
        r - [x,y,z] Point of observation\n
        particle_pos - [x,y,z] particle position\n
        q - [float] Charge of particle\n
        
        ### Optional
        ?? - Permittivity\n
        
        ## Returns
        3D array
        '''
        
        ?? = eps
        
        R = r - particle_pos
        R_norm = np.linalg.norm(R)
        E = q/(4*np.pi*??) * (R/R_norm**3)
        
        return E


    @classmethod
    def CalculateEfield(cls, L: float, N: int, Q: list, R: np.ndarray, plane: str, eps: float = 8.854187817E-12) -> np.ndarray:
        '''
        Calculates electric field from one or more point charges\n
        
        ## Input
        L - [float] Length of side of cube area calculated\n
        N - [int] Number of points in the meshgrid\n
        Q - [list] List of each particles charge. Must be nested (see example code)\n
        R - [ndarray] List of each particles position in 3D. Must be nested (see example code)\n
        plane [str] - Plane of interest for plotting. 
        
        ### Optional
        ?? - Permittivity, default being ??_0\n
        
        ## Returns
        r1, r2, E1, E2
        '''
        
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {plane}")
        
        if len(Q) != len(R):
            raise ValueError(f"Argument 'Q', must have same length {len(Q)} as argument 'R' {len(R)} as they represent all charges and their respective positions")
        
        elif plane == 'xy':
            x, y = [np.linspace(-L, L, N) for i in range(2)]
            rx, ry = np.meshgrid(x, y)
            Ex, Ey = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], ry.flat[i], 0.0])
                for j in range(len(Q)):
                    Ex_temp, Ey_temp, Ez_temp = cls.Efield(r, R[j], Q[j], eps)
                    Ex.flat[i] += Ex_temp
                    Ey.flat[i] += Ey_temp
                    
            return rx, ry, Ex, Ey

        elif plane == 'xz':
            x, z = [np.linspace(-L, L, N) for i in range(2)]
            rx, rz = np.meshgrid(x, z)
            Ex, Ez = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], 0.0, rz.flat[i]])
                for j in range(len(Q)):
                    Ex_temp, Ey_temp, Ez_temp = cls.Efield(r, R[j], Q[j], eps)
                    Ex.flat[i] += Ex_temp
                    Ez.flat[i] += Ez_temp
                    
            return rx, rz, Ex, Ez
                    
        elif plane == 'yz':
            y, z = [np.linspace(-L, L, N) for i in range(2)]
            ry, rz = np.meshgrid(y, z)
            Ey, Ez = np.zeros((2,N,N))  
            for i in range(len(ry.flat)):
                r = np.array([0.0, ry.flat[i], rz.flat[i]])
                for j in range(len(Q)):
                    Ex_temp, Ey_temp, Ez_temp = cls.Efield(r, R[j], Q[j], eps)
                    Ey.flat[i] += Ey_temp
                    Ez.flat[i] += Ez_temp
                    
            return ry, rz, Ey, Ez
        

    @staticmethod
    @njit
    def Epot(r: np.ndarray, particle_pos: np.ndarray, q: float, eps = 8.854187817E-12) -> np.ndarray:
        '''
        Calculates electric potential from point charge\n
        
        ## Input
        r - [x,y,z] Point of observation\n
        particle_pos - [x,y,z] particle position\n
        q - [float] Charge of particle\n
        
        ### Optional
        ?? - Permittivity\n
        
        ## Returns
        3D array
        '''
        ?? = eps
        
        R_norm = np.linalg.norm(r-particle_pos)
        V = q/(4*np.pi*??*R_norm)
        return V
    
    @classmethod
    def CalculateEpot(cls, L: float, N: int, Q: list, R: list, plane: str, eps: float = 8.854187817E-12) -> np.ndarray:
        '''
        Calculates electric potential from one or more point charges\n
        
        ## Input
        L - [float] Length of side of cube area calculated\n
        N - [int] Number of points in the meshgrid\n
        Q - [list] List of each particle's charge. Must be nested (see example code)\n
        R - [ndarray] List of each particle's position in 3D. Must be nested (see example code)\n
        plane [str] - Plane of interest for plotting\n
        
        ### Optional
        ?? - Permittivity, default being ??_0\n
        
        
        ## Returns
        r1, r2, V
        '''
        
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {plane}")
        
        if len(Q) != len(R):
            raise ValueError(f"Argument 'Q', must have same length {len(Q)} as argument 'R' {len(R)} as they represent all charges and their respective positions")
        
        elif plane == 'xy':
            x, y = [np.linspace(-L, L, N) for i in range(2)]
            rx, ry = np.meshgrid(x, y)
            V = np.zeros((N,N))  
            V_temp = 0
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], ry.flat[i], 0.0])
                for j in range(len(Q)):
                    V_temp = cls.Epot(r, R[j], Q[j], eps)
                    V.flat[i] += V_temp
                    
            return rx, ry, V

        elif plane == 'xz':
            x, z = [np.linspace(-L, L, N) for i in range(2)]
            rx, rz = np.meshgrid(x, z)
            V = np.zeros((N,N))  
            V_temp = 0
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], 0.0, rz.flat[i]])
                for j in range(len(Q)):
                    V_temp = cls.Epot(r, R[j], Q[j], eps)
                    V.flat[i] += V_temp
                    
            return rx, rz, V
                    
        elif plane == 'yz':
            y, z = [np.linspace(-L, L, N) for i in range(2)]
            ry, rz = np.meshgrid(y, z)
            V = np.zeros((N,N))  
            V_temp = 0
            for i in range(len(ry.flat)):
                r = np.array([0.0, ry.flat[i], rz.flat[i]])
                for j in range(len(Q)):
                    V_temp = cls.Epot(r, R[j], Q[j], eps)
                    V.flat[i] += V_temp
                    
            return ry, rz, V

    @staticmethod
    @njit
    def EfieldLine(r: np.ndarray, q: float, line_length: float, axis: str, x: float = 0, y: float = 0, z: float = 0, eps: float = 8.854187817E-12, N: int = 100) -> np.ndarray:
        '''
        Calculates electric field from line charge parallel with either x, y or z axis\n
        
        ## Input
        r - [x,y,z] Point of observation\n
        q - [float] Charge of particle\n
        line_length - Length of charged line\n
        axis - ("x", "y" or "z") Axis which line charge is parallel\n
        
        ### Optional
        x - x coordinate of line center, default is 0\n
        y - y coordinate of line center, default is 0\n
        z - z coordinate of line center, default is 0\n
        N - Accuracy, the higher the better, but slower. Default is 100\n
        ?? - Permittivity, default is vacuum\n
        
        
        ## Returns
        3D array
        '''
        ?? = eps
        
        E = np.zeros(3)
        dl = line_length/N
        dq = q/N
        
        if axis == 'x':
            for i in range(N):
                r_q = np.array([-line_length/2 + i*dl +x , y, z])
                R = r - r_q
                R_norm = np.linalg.norm(R) 
                E += dq/(4*np.pi*??) * R/R_norm**3
                
            return E
        
        elif axis == 'y':
            for i in range(N):
                r_q = np.array([x, -line_length/2 + i*dl + y, z])
                R = r - r_q
                R_norm = np.linalg.norm(R) 
                E += dq/(4*np.pi*??) * R/R_norm**3
                
            return E            
        
        elif axis == 'z':
            for i in range(N):
                r_q = np.array([x, y, -line_length/2 + i*dl + z])
                R = r - r_q
                R_norm = np.linalg.norm(R) 
                E += dq/(4*np.pi*??) * R/R_norm**3
                
            return E  
        
        else:
            raise ValueError("Argument axis must be either 'x', 'y', or 'z'")


    @classmethod
    def CalculateEfieldLine(cls, L: float, N: int, line_charges: list, line_lengths: list, line_center_coords: list, axis: list, plane: str, eps = 8.854187817E-12, n = 100) -> np.ndarray:
        '''
        Calculates electric field from one or more line charges\n
        
        ## Input
        L - [float] Length of side of cube area calculated\n
        N - [int] Number of points in the meshgrid\n
        line_charges - [list] List of each line's charge. Must be nested (see example code)\n
        line_lengths - [list] List of each line's length\n
        line_center_coords - [list] List of each particle's position in 3D. Must be nested (see example code)\n
        axis - [list] List of which axis each line is runs parallel\n
        plane [str] - Plane of interest for plotting\n
        
        ### Optional
        ?? - Permittivity, default being ??_0\n
        
        
        ## Returns
        r1, r2, E1, E2
        '''
        
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {plane}")
        
        if len(line_lengths) != len(line_center_coords) or len(line_lengths) != len(axis) or len(line_lengths) != len(line_charges):
            raise ValueError(f"Argument 'line_charges', 'line_lengths', 'line_center_coords' and 'axis must have same lengths {len(line_charges)}, {len(line_lengths)}, {len(line_center_coords)}, {len(axis)}")
        
        elif plane == 'xy':
            x, y = [np.linspace(-L, L, N) for i in range(2)]
            rx, ry = np.meshgrid(x, y)
            Ex, Ey = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], ry.flat[i], 0.0])
                for j in range(len(line_charges)):
                    Ex_temp, Ey_temp, Ez_temp = cls.EfieldLine(r, line_charges[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], eps, n)
                    Ex.flat[i] += Ex_temp
                    Ey.flat[i] += Ey_temp
                    
            return rx, ry, Ex,Ey

        elif plane == 'xz':
            x, z = [np.linspace(-L, L, N) for i in range(2)]
            rx, rz = np.meshgrid(x, z)
            Ex, Ez = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], 0.0, rz.flat[i]])
                for j in range(len(line_charges)):
                    Ex_temp, Ey_temp, Ez_temp = cls.EfieldLine(r, line_charges[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], eps, n)
                    Ex.flat[i] += Ex_temp
                    Ez.flat[i] += Ez_temp
                    
            return rx, rz, Ex, Ez
                    
        elif plane == 'yz':
            y, z = [np.linspace(-L, L, N) for i in range(2)]
            ry, rz = np.meshgrid(y, z)
            Ey, Ez = np.zeros((2,N,N))  
            for i in range(len(ry.flat)):
                r = np.array([0.0, ry.flat[i], rz.flat[i]])
                for j in range(len(line_charges)):
                    Ex_temp, Ey_temp, Ez_temp = cls.EfieldLine(r, line_charges[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], eps, n)
                    Ey.flat[i] += Ey_temp
                    Ez.flat[i] += Ez_temp
                    
            return ry, rz, Ey, Ez

    @staticmethod
    @njit
    def EpotLine(r: np.ndarray, q: float, line_length: float, axis: str, x: float = 0, y: float = 0, z: float = 0, eps: float = 8.854187817E-12, N: int = 100) -> np.ndarray:
        '''
        Calculates electric potential from line charge parallel with either x, y or z axis \n
        
        ## Input
        r - [x,y,z] Point of observation\n
        q - [float] Charge of particle\n
        line_length - Length of charged line\n
        axis - ("x", "y" or "z") Axis which line charge is parallel\n
        
        ### Optional
        x - x coordinate of line center\n
        y - y coordinate of line center\n
        z - z coordinate of line center\n
        N - Accuracy, the higher the better, but slower. Default is 100\n
        ?? - Permittivity, default is vacuum\n
        
        
        ## Returns
        3D array
        '''
        ?? = eps
        
        V = 0
        dl = line_length/N
        dq = q/N
        
        if axis == 'x':
            for i in range(N):
                r_q = np.array([-line_length/2 + i*dl +x , y, z])
                R = r - r_q
                R_norm = np.linalg.norm(R) 
                V += dq/(4*np.pi*??*R_norm)
                
            return V
        
        elif axis == 'y':
            for i in range(N):
                r_q = np.array([x, -line_length/2 + i*dl + y, z])
                R = r - r_q
                R_norm = np.linalg.norm(R) 
                V += dq/(4*np.pi*??*R_norm)
                
            return V            
        
        elif axis == 'z':
            for i in range(N):
                r_q = np.array([x, y, -line_length/2 + i*dl + z])
                R = r - r_q
                R_norm = np.linalg.norm(R) 
                V += dq/(4*np.pi*??*R_norm)
                
            return V  
        
        else:
            raise ValueError("Argument axis must be either 'x', 'y', or 'z'")
        
    @classmethod
    def CalculateEpotLine(cls, L: float, N: int, line_charges: list, line_lengths: list, line_center_coords: list, axis: str, plane: str, eps = 8.854187817E-12, n = 100) -> np.ndarray:
        '''
        Calculates electric potential from one or more line charges\n
        
        ## Input
        L - [float] Length of side of cube area calculated\n
        N - [int] Number of points in the meshgrid\n
        line_charges - [list] List of each line's charge. Must be nested (see example code)\n
        line_lengths - [list] List of each line's length\n
        line_center_coords - [list] List of each particle's position in 3D. Must be nested (see example code)\n
        plane [str] - Plane of interest for plotting\n
        
        ### Optional
        ?? - Permittivity, default being ??_0\n
        n - [int] How small pieces to divide the line, default being 100\n
        
        
        ## Returns
        r1, r2, V
        '''
        
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {plane}")
        
        if len(line_lengths) != len(line_center_coords) or len(line_lengths) != len(axis) or len(line_lengths) != len(line_charges):
            raise ValueError(f"Argument 'line_charges', 'line_lengths', 'line_center_coords' and 'axis must have same lengths {len(line_charges)}, {len(line_lengths)}, {len(line_center_coords)}, {len(axis)}")
        
        elif plane == 'xy':
            x, y = [np.linspace(-L, L, N) for i in range(2)]
            rx, ry = np.meshgrid(x, y)
            V = np.zeros((N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], ry.flat[i], 0.0])
                for j in range(len(line_charges)):
                    V_temp = cls.EpotLine(r, line_charges[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], eps, n)
                    V.flat[i] += V_temp 
                    
            return rx, ry, V

        elif plane == 'xz':
            x, z = [np.linspace(-L, L, N) for i in range(2)]
            rx, rz = np.meshgrid(x, z)
            V = np.zeros((N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], 0.0, rz.flat[i]])
                for j in range(len(line_charges)):
                    V_temp = cls.EpotLine(r, line_charges[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], eps, n)
                    V.flat[i] += V_temp 
                    
            return rx, rz, V
                    
        elif plane == 'yz':
            y, z = [np.linspace(-L, L, N) for i in range(2)]
            ry, rz = np.meshgrid(y, z)
            V = np.zeros((N,N))  
            for i in range(len(ry.flat)):
                r = np.array([0.0, ry.flat[i], rz.flat[i]])
                for j in range(len(line_charges)):
                    V_temp = cls.EpotLine(r, line_charges[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], eps, n)
                    V.flat[i] += V_temp 
                    
            return ry, rz, V


    @staticmethod
    @njit
    def EfieldCircle(r: np.ndarray, q: float, rad: float, plane: str, eps = 8.854187817E-12, N = 200) -> np.ndarray:
        '''
        Calculates electric field from circular charge centered in origin\n
        
        ## Input
        r - [x,y,z] Point of observation\n
        q - [float] Charge of circle\n
        rad - [float] Radius of the circle\n
        plane - [str] Plane of the circle\n
        
        ### Optional
        ?? - Permittivity, default is vacuum\n
        N - Accuracy, the higher the better, but slower. Default is 100\n\n
        
        
        ## Returns
        3D array
        '''
        ?? = eps
        E = np.zeros(3)
        dq = q/N
            
        if plane == 'xy':
            if np.linalg.norm(r) < rad and r[2] == 0:
                return E
                
            else:
                for i in range(N):
                    ?? = 2*np.pi/N * i
                    r_q = rad * np.array([np.cos(??), np.sin(??), 0.0])
                    R = r - r_q
                    R_norm = np.linalg.norm(R)
                    E += dq/(4*np.pi*??) * R/R_norm**3
                    
                return E


        elif plane == 'xz':
            if np.linalg.norm(r) < rad and r[1] == 0:
                return E
                
            else:
                for i in range(N):
                    ?? = 2*np.pi/N * i
                    r_q = rad * np.array([np.cos(??), 0.0, np.sin(??)])
                    R = r - r_q
                    R_norm = np.linalg.norm(R)
                    E += dq/(4*np.pi*??) * R/R_norm**3
                    
                return E
        
        elif plane == 'yz':
            if np.linalg.norm(r) < rad and r[0] == 0:
                return E
                
            else:        
                for i in range(N):
                    ?? = 2*np.pi/N * i
                    r_q = rad * np.array([0.0, np.cos(??), np.sin(??)])
                    R = r - r_q
                    R_norm = np.linalg.norm(R)
                    E += dq/(4*np.pi*??) * R/R_norm**3
                    
                return E

        else:
            raise ValueError("Argument 'plane' must be either 'xy', 'xz', or 'yz'")
            
    @classmethod
    def CalculateEfieldCircle(cls, L: float, N: int, circle_charges: list,  radii: list, plane: str, plane_circles: list, eps: float = 8.854187817E-12, n: int = 100) -> np.ndarray:
        '''
        Calculates electric field from one or more circular charges centered around origin\n
        
        ## Input
        L - [float] Length of side of cube area calculated\n
        N - [int] Number of points in the meshgrid\n
        circle_charges - [list] List of each circle's charge. Must be nested (see example code)\n
        radii - [list] List of each circle's radius\n
        plane - [str] Plane of interest for plotting\n
        plane_circles - [str] Plane of the circle\n
        
        ### Optional
        ?? - Permittivity, default being ??_0\n
        n - [int] How small pieces to divide the line, default being 100\n
        
        
        ## Returns
        r1, r2, E1, E2
        '''
        
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {plane}")
        
        if len(circle_charges) != len(radii) or len(circle_charges) != len(plane_circles):
            raise ValueError(f"Argument 'circle_charges', 'radii' and 'plane_circles' must have same lengths {len(circle_charges)}, {len(radii)}, {len(plane_circles)}")
        
        elif plane == 'xy':
            x, y = [np.linspace(-L, L, N) for i in range(2)]
            rx, ry = np.meshgrid(x, y)
            Ex, Ey = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], ry.flat[i], 0.0])
                for j in range(len(circle_charges)):
                    Ex_temp, Ey_temp, Ez_temp = cls.EfieldCircle(r, circle_charges[j], radii[j], plane_circles[j], eps, n)
                    Ex.flat[i] += Ex_temp
                    Ey.flat[i] += Ey_temp
                    
            return rx, ry, Ex,Ey

        elif plane == 'xz':
            x, z = [np.linspace(-L, L, N) for i in range(2)]
            rx, rz = np.meshgrid(x, z)
            Ex, Ez = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], 0.0, rz.flat[i]])
                for j in range(len(circle_charges)):
                    Ex_temp, Ey_temp, Ez_temp = cls.EfieldCircle(r, circle_charges[j], radii[j], plane_circles[j], eps, n)
                    Ex.flat[i] += Ex_temp
                    Ez.flat[i] += Ez_temp
                    
            return rx, rz, Ex, Ez
                    
        elif plane == 'yz':
            y, z = [np.linspace(-L, L, N) for i in range(2)]
            ry, rz = np.meshgrid(y, z)
            Ey, Ez = np.zeros((2,N,N))  
            for i in range(len(ry.flat)):
                r = np.array([0.0, ry.flat[i], rz.flat[i]])
                for j in range(len(circle_charges)):
                    Ex_temp, Ey_temp, Ez_temp = cls.EfieldCircle(r, circle_charges[j], radii[j], plane_circles[j], eps, n)
                    Ey.flat[i] += Ey_temp
                    Ez.flat[i] += Ez_temp
                    
            return ry, rz, Ey, Ez
        

        
    @staticmethod
    @njit
    def EpotCircle(r: np.ndarray, q: float, rad: float, plane: str, eps: float = 8.854187817E-12, N: int = 100) -> np.ndarray:
        '''
        Calculates electric potential from circular charge centered in origin\n
        
        ## Input
        r - [x,y,z] Point of observation\n
        q - [float] Charge of circle\n
        rad - [float] Radius of the circle\n
        
        ### Optional
        N - Accuracy, the higher the better, but slower. Default is 100\n
        ?? - Permittivity, default is vacuum\n
        
        
        ## Returns
        3D array
        '''
        ?? = eps
        
        V = 0
        dq = q/N
    
        if plane == 'xy':
            if np.linalg.norm(r) < rad and r[2] == 0:
                return q/(4*np.pi*??*rad)
            
            else:
                for i in range(N):
                    ?? = 2*np.pi/N * i
                    r_q = rad * np.array([np.cos(??), np.sin(??), 0.0])
                    R = r - r_q
                    R_norm = np.linalg.norm(R)
                    V += dq/(4*np.pi*??*R_norm)
                
                return V

        elif plane == 'xz':
            if np.linalg.norm(r) < rad and r[1] == 0:
                return q/(4*np.pi*??*rad)
            
            else:
                for i in range(N):
                    ?? = 2*np.pi/N * i
                    r_q = rad * np.array([np.cos(??), 0.0, np.sin(??)])
                    R = r - r_q
                    R_norm = np.linalg.norm(R)
                    V += dq/(4*np.pi*??*R_norm)
                
                return V
        
        elif plane == 'yz':
            if np.linalg.norm(r) < rad and r[0] == 0:
                return q/(4*np.pi*??*rad)
            
            else:
                for i in range(N):
                    ?? = 2*np.pi/N * i
                    r_q = rad * np.array([0.0, np.cos(??), np.sin(??)])
                    R = r - r_q
                    R_norm = np.linalg.norm(R)
                    V += dq/(4*np.pi*??*R_norm)
                    
                return V
        else:
            raise ValueError("Argument 'plane' must be either 'xy', 'xz', or 'yz'")
        
    @classmethod
    def CalculateEpotCircle(cls, L: float, N: int, circle_charges: list,  radii: list, plane: str, plane_circles: list, eps: float = 8.854187817E-12, n: int = 100) -> np.ndarray:
        '''
        Calculates electric potential from one or more circular charges centered around origin\n
        
        ## Input
        L - [float] Length of side of cube area calculated\n
        N - [int] Number of points in the meshgrid\n
        circle_charges - [list] List of each circle's charge. Must be nested (see example code)\n
        radii - [list] List of each circle's radius\n
        plane - [str] Plane of interest for plotting\n
        plane_circles - [list] List of the plane to place the circle\n
        
        ### Optional
        ?? - Permittivity, default being ??_0\n
        n - [int] How small pieces to divide the line, default being 100\n\n
        
        
        ## Returns
        r1, r2, V
        '''

        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {plane}")
        
        if len(circle_charges) != len(radii):
            raise ValueError(f"Argument 'circle_charges' and 'radii' must have same lengths {len(circle_charges)}, {len(radii)}")
        
        elif plane == 'xy':
            x, y = [np.linspace(-L, L, N) for i in range(2)]
            rx, ry = np.meshgrid(x, y)
            V = np.zeros((N,N))
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], ry.flat[i], 0.0])
                for j in range(len(circle_charges)):
                    V_temp = cls.EpotCircle(r, circle_charges[j], radii[j], plane_circles[j], eps, n)
                    V.flat[i] += V_temp
                    
            return rx, ry,V

        elif plane == 'xz':
            x, z = [np.linspace(-L, L, N) for i in range(2)]
            rx, rz = np.meshgrid(x, z)
            V = np.zeros((N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], 0.0, rz.flat[i]])
                for j in range(len(circle_charges)):
                    V_temp = cls.EpotCircle(r, circle_charges[j], radii[j], plane_circles[j], eps, n)
                    V.flat[i] += V_temp
                    
            return rx, rz, V
                    
        elif plane == 'yz':
            y, z = [np.linspace(-L, L, N) for i in range(2)]
            ry, rz = np.meshgrid(y, z)
            V = np.zeros((N,N))  
            for i in range(len(ry.flat)):
                r = np.array([0.0, ry.flat[i], rz.flat[i]])
                for j in range(len(circle_charges)):
                    V_temp = cls.EpotCircle(r, circle_charges[j], radii[j], plane_circles[j], eps, n)
                    V.flat[i] += V_temp
                    
            return ry, rz, V
        
    
    @staticmethod
    @njit
    def BfieldLine(r: np.ndarray, line_length: float, I: float, axis: str, x: float = 0, y: float = 0, z: float = 0, mu: float = 4*np.pi*1E-7, N: int = 100) -> np.ndarray:
        '''
        Calculates magnetic field from line current parallel to x, y or z axis. Can be placed anywhere in 3D\n
        
        ## Input
        r - [x,y,z] Point of observation\n
        line_length - Length of the line\n
        I - Magnitude of current\n
        axis - ('x', 'y' or 'z') Which axis line runs parallel\n
        x, y, z - How much to shift the line in x, y or z direction\n
        
        ### Optional
        ?? - Permeability of magnetic field, vacuum being default\n
        N - Accuracy, the higher the better, but slower. Default is 100\n\n
        
        
        ## Returns
        3D array
        '''
        ?? = mu
        B = np.zeros(3)
        line_segment = line_length/N
        
        if axis == 'x':
            dl = line_segment * np.array([1.0, 0.0, 0.0])
            for i in range(N):
                di = np.array([-line_length/2 + line_segment*i + x, 0.0 + y, 0.0 + z])
                R = r - di
                R_norm = np.linalg.norm(R)
                B += ??/(4*np.pi)* I/R_norm**3 * np.cross(dl, R)
            
            return B
        
        elif axis == 'y':
            dl = line_segment * np.array([0.0, 1.0, 0.0])
            for i in range(N):
                di = np.array([0.0 +x, -line_length/2 + line_segment*i + y, 0.0 + z])
                R = r - di
                R_norm = np.linalg.norm(R)
                B += ??/(4*np.pi)* I/R_norm**3 * np.cross(dl, R)
            
            return B
    
        elif axis == 'z':
            dl = line_segment * np.array([0.0, 0.0, 1.0])
            for i in range(N):
                di = np.array([0.0 + x, 0.0 + y, -line_length/2 + line_segment*i + z])
                R = r - di
                R_norm = np.linalg.norm(R)
                B += ??/(4*np.pi)* I/R_norm**3 * np.cross(dl, R)
            
            return B

        else:
            raise ValueError("Argument axis must be either 'x', 'y', or 'z'")
    
 
    @classmethod
    def CalculateBfieldLine(cls, L: float, N: int, line_currents: list, line_lengths: list, line_center_coords: list, axis: list, plane: str, mu: float = 4*np.pi*1E-12, n: float = 100) -> np.ndarray:
        '''
        Calculates magnetic field from one or more lines charge\n
        
        ## Input
        L - [float] Length of side of cube area calculated\n
        N - [int] Number of points in the meshgrid\n
        line_current - [list] List of the magnitude of each line's current. Must be nested (see example code)\n
        line_lengths - [list] List of each line's length
        line_center_coords - [list] List of each lines's center position in 3D. Must be nested (see example code)\n
        axis - [list] List containing axis each line runs parallel\n
        plane [str] - Plane of interest for plotting\n
        
        ### Optional
        ?? - Permittivity, default being ??_0\n
        n - [int] How small pieces to divide the line, default being 100\n\n
        
        
        ## Returns
        r1, r2, B1, B2
        '''
        
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {plane}")
        
        if len(line_lengths) != len(line_center_coords) or len(line_lengths) != len(axis) or len(line_lengths) != len(line_currents):
            raise ValueError(f"Argument 'line_charges', 'line_lengths', 'line_center_coords' and 'axis must have same lengths {len(line_currents)}, {len(line_lengths)}, {len(line_center_coords)}, {len(axis)}")
        
        elif plane == 'xy':
            x, y = [np.linspace(-L, L, N) for i in range(2)]
            rx, ry = np.meshgrid(x, y)
            Bx, By = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], ry.flat[i], 0.0])
                for j in range(len(line_currents)):
                    Bx_temp, By_temp, Bz_temp = cls.BfieldLine(r, line_currents[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], mu, n)
                    Bx.flat[i] += Bx_temp
                    By.flat[i] += By_temp
                    
            return rx, ry, Bx, By

        elif plane == 'xz':
            x, z = [np.linspace(-L, L, N) for i in range(2)]
            rx, rz = np.meshgrid(x, z)
            Bx, Bz = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], 0.0, rz.flat[i]])
                for j in range(len(line_currents)):
                    Bx_temp, By_temp, Bz_temp = cls.BfieldLine(r, line_currents[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], mu, n)
                    Bx.flat[i] += Bx_temp
                    Bz.flat[i] += Bz_temp
                    
            return rx, rz, Bx, Bz
                    
        elif plane == 'yz':
            y, z = [np.linspace(-L, L, N) for i in range(2)]
            ry, rz = np.meshgrid(y, z)
            By, Bz = np.zeros((2,N,N))  
            for i in range(len(ry.flat)):
                r = np.array([0.0, ry.flat[i], rz.flat[i]])
                for j in range(len(line_currents)):
                    Bx_temp, By_temp, Bz_temp = cls.BfieldLine(r, line_currents[j], line_lengths[j], axis[j], line_center_coords[j][0], line_center_coords[j][1], line_center_coords[j][2], mu, n)
                    By.flat[i] += By_temp
                    Bz.flat[i] += Bz_temp
                    
            return ry, rz, By, Bz   
    
    
    
    @staticmethod
    @njit
    def BfieldCircle(r: np.ndarray, I: float, rad: float, plane: str, mu: float = 4*np.pi*1E-7, N: float = 100) -> np.ndarray:
        '''
        Calculates magnetic field from circular current centered in origin in the xy, xz or yz plane in 3D\n
        
        ## Input
        r - [x,y,z] Point of observation\n
        I - Magnitude of current\n
        rad - [float] Radius of the the circle\n
        plane - [str] Plane of the circle\n
        
        ### Optional
        ?? - Permeability of magnetic field, ??_0 being default\n
        N - Accuracy, the higher the better, but slower. Default is 100\n
        
        
        ## Returns
        3D array
        '''
        
        if plane == 'xy':
            ?? = mu
            d?? = 2*np.pi/N 
            dl = rad*2*np.pi/N
            B = np.zeros(3)
            for i in range(N):
                current_pos = rad*np.array([np.cos(d??*i), np.sin(d??*i), 0])
                Idl         = I*dl*np.array([-np.sin(d??*i), np.cos(d??*i), 0])
                R           = r - current_pos
                R_norm      = np.linalg.norm(R)
                B          += ??/(4*np.pi)* np.cross(Idl, R)/R_norm**3
            return B
        
        elif plane == 'xz':
            ?? = mu
            d?? = 2*np.pi/N 
            dl = rad*2*np.pi/N
            B = np.zeros(3)
            for i in range(N):
                current_pos = rad*np.array([np.cos(d??*i), 0, np.sin(d??*i)])
                Idl         = I*dl*np.array([-np.sin(d??*i), 0, np.cos(d??*i)])
                R           = r - current_pos
                R_norm      = np.linalg.norm(R)
                B          += ??/(4*np.pi)* np.cross(Idl, R)/R_norm**3
            return B
        
        elif plane == 'yz':
            ?? = mu
            d?? = 2*np.pi/N 
            dl = rad*2*np.pi/N
            B = np.zeros(3)
            for i in range(N):
                current_pos = rad*np.array([0, np.cos(d??*i), np.sin(d??*i)])
                Idl         = I*dl*np.array([0, -np.sin(d??*i), np.cos(d??*i)])
                R           = r - current_pos
                R_norm      = np.linalg.norm(R)
                B          += ??/(4*np.pi)* np.cross(Idl, R)/R_norm**3
            return B
    
    @classmethod
    def CalculateBfieldCircle(cls, L: float, N: int, circle_currents: list, radii: list, plane: str, circle_planes: list, mu: float = 4*np.pi*1E-7, n: int = 100) -> np.ndarray:
        '''
        Calculates magnetic field from one or more circular currents\n
        
        ## Input
        L - [float] Length of side of cube area calculated\n
        N - [int] Number of points in the meshgrid\n
        line_currents - [list] List of each circle's current. Must be nested (see example code)\n
        radii - [list] List of each circle's radii\n
        plane [str] - Plane of interest for plotting\n
        circle_planes - [list] List of which plane to place each circle\n
        
        ### Optional
        ?? - Permittivity, default being ??_0\n
        n - [int] How small pieces to divide the line, default being 100\n\n
        
        
        ## Returns
        r1, r2, B1, B2
        '''
        
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {plane}")
        
        for c_plane in circle_planes:
            if c_plane not in ['xy', 'xz', 'yz']:
                raise ValueError(f"Argument 'plane' must be 'xy', 'xz' or yz not {c_plane}")
        
        if len(circle_currents) != len(radii) or len(circle_currents) != len(circle_planes):
            raise ValueError(f"Argument 'circle_charges', 'radii' and 'circle_planes' must have same lengths {len(circle_currents)}, {len(radii)}, {len(circle_planes)}")
        
        elif plane == 'xy':
            x, y = [np.linspace(-L, L, N) for i in range(2)]
            rx, ry = np.meshgrid(x, y)
            Bx, By = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], ry.flat[i], 0.0])
                for j in range(len(circle_currents)):
                    Bx_temp, By_temp, Bz_temp = cls.BfieldCircle(r, circle_currents[j], radii[j], circle_planes[j], mu, n)
                    Bx.flat[i] += Bx_temp
                    By.flat[i] += By_temp
                    
            return rx, ry, Bx,By

        elif plane == 'xz':
            x, z = [np.linspace(-L, L, N) for i in range(2)]
            rx, rz = np.meshgrid(x, z)
            Bx, Bz = np.zeros((2,N,N))  
            for i in range(len(rx.flat)):
                r = np.array([rx.flat[i], 0.0, rz.flat[i]])
                for j in range(len(circle_currents)):
                    Bx_temp, By_temp, Bz_temp = cls.BfieldCircle(r, circle_currents[j], radii[j], circle_planes[j], mu, n)
                    Bx.flat[i] += Bx_temp
                    Bz.flat[i] += Bz_temp
                    
            return rx, rz, Bx, Bz
                    
        elif plane == 'yz':
            y, z = [np.linspace(-L, L, N) for i in range(2)]
            ry, rz = np.meshgrid(y, z)
            By, Bz = np.zeros((2,N,N))  
            for i in range(len(ry.flat)):
                r = np.array([0.0, ry.flat[i], rz.flat[i]])
                for j in range(len(circle_currents)):
                    Bx_temp, By_temp, Bz_temp = cls.BfieldCircle(r, circle_currents[j], radii[j], circle_planes[j], mu, n)
                    By.flat[i] += By_temp
                    Bz.flat[i] += Bz_temp
                    
            return ry, rz, By, Bz

        
    @staticmethod
    def PlotVector(r1: np.ndarray, r2: np.ndarray, U1: np.ndarray, U2: np.ndarray, type: str, title: str = '', figsize: tuple = (16,9), broken_streamlines: bool = True, density: float = 1, cmap: str = 'cool', equal: bool = False, show: bool = False, log10: bool = True) -> None:
        '''
        Function which plots vector\n
        
        ## Input
        r1, r2 - Arrays containing the meshgrid\n
        U1, U1 - Arrays containing the values of the vectors at each point in the grid\n
        type - ('stream', 'quiver') Choose whether to do a streamplot or quiver\n
        
        ### Optional
        title - [str] title of the plot\n
        figsize - [tuple] (M, N) size of the plot, default is (16,9)\n
        broken_streamlines - [bool] Default is True\n
        density - [float] How dense the plot will be filled, default is 1\n
        cmap - [str] Color map to use, default is 'cool'\n
        equal - [bool] Wether to set axis equal, default is False \n
        show - [bool] Whether to show the figure, default is False\n
        log10 - [bool] If to use log10 of colors. Makes for a smoother transition, default is True\n
        
        
        ## Returns
        None
        '''
        mag = np.sqrt(U1**2 + U2**2)
        U = U1/mag
        V = U2/mag
        if log10:
            color = np.log10(mag)
        else:
            color = mag
            
        plt.figure(figsize=figsize)
        
        if type == 'stream':
            
            if broken_streamlines:       
                plt.streamplot(r1, r2, U, V, density = density, color = color, cmap = cmap)
                plt.title(title)
                plt.colorbar()
                
            else:
                plt.streamplot(r1, r2, U, V, broken_streamlines = False, density = density, color = color, cmap = cmap)
                plt.title(title)
                plt.colorbar()

            
        elif type == 'quiver':
            plt.quiver(r1, r2, U, V, color, cmap = cmap)
            plt.title(title)
            plt.colorbar() 
            
        else:
            raise ValueError('"type" must be either "stream" or "quiver"')       
        
        if equal:
            plt.axis('equal')
            
        if show:
            plt.show()

        
    @staticmethod
    def PlotContour(r1: np.ndarray, r2: np.ndarray, V: np.ndarray, title: str = '', figsize: tuple = (16,9), levels: int = 200, norm: str = 'symlog', cmap: str = 'inferno', equal: bool = False, show: bool = False) -> None:
        '''
        Function which plots potential\n
        
        ## Input
        r1, r2 - Arrays containing the meshgrid\n
        V - Array containing the values of the potential at each point in the grid\n
        
        ### Optional
        title - [str] title of the plot\n
        figsize - [tuple] (M, N) size of the plot, default is (16,9)\n
        levels - [int] How many the contour plot will show, default is 200\n
        norm - [str] Scale of contour, default is 'symlog'\n
        cmap - [str] Color map to use, default is 'cool'\n
        equal - [bool] Wether to set axis equal, default is False \n
        show - [bool] Whether to show the figure, default is False\n
        
        
        ## Returns
        None
        '''
        
        plt.figure(figsize = figsize)
        if equal:
            plt.axis('equal')
        plt.contour(r1, r2, V, levels = levels, norm = norm, cmap = cmap)
        plt.contourf(r1, r2, V, levels = levels, norm = norm, cmap = cmap)
        plt.title(title)
        plt.colorbar()
        
        if show:
            plt.show()
            
            
    def PlotCircle(radius: float) -> None:
        '''
        ## Input
        Plots circle\n
        radius - [float]
        
        ## Returns
        None
        '''
        t = np.linspace(0, 2*np.pi, 100)
        plt.plot(radius*np.cos(t), radius*np.sin(t))  