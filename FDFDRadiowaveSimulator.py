# -*- coding: utf-8 -*-
"""
v2:14/09/2020
"""
import numpy as np
from PIL import Image
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from pandas import *
import time



class FDFDRadiowaveSimulator():
    
    def __init__(self, name,verbose = False):
        """
        FDFDRadiowaveSimulator is a Python class to simulate the Helmoltz equation

        Parameters
        ----------
        name : string
            Name of the bitmap file.
        
        verbose : bool, optional
            Display information or not.

        Returns
        -------
        None.

        """
        #Load the image
        self.name = name
        self.path = "sources/" +name
        self.verbose = verbose
        t=time.time()
        self.im = self.load()
        if self.verbose:
            print("Elapsed time to import the image : " + str(time.time()-t))
        self.Ny,self.Nx=self.im.shape
        
        #Creation of dummys for each material
        self.wall_dummy()
        self.wall_neighbour_dummy()
        self.air_dummy()
        self.external_absorbing_layer()
        self.create_adjacency_matrix()
        
        
        #Variables to store stats
        self.sum =0
        self.solve_counter = 0
        self.stats = ""
        
        
        
    def set_parameters(self,dx, wavelength, optic_index_walls, alpha_walls):        
        """
        Set the simulator's parameters, create of the system matrix and LU factorization        

        Parameters
        ----------
        dx : float
            Discretization step of the map (in meter)
        wavelength : float
            Wavelength to simulate (in meter)
        optic_index_walls : float
            Optic index of the walls
        alpha_walls : float
            Alpha parameter (mu x sigma) of the walls.

        Returns
        -------
        None.

        """

        self.dx=dx
        self.wavelength = wavelength
        self.optic_index_walls = optic_index_walls
        self.alpha_walls = alpha_walls
        T=wavelength/(3*(10**8))
        omega = (2*np.pi/(T))
        
        #Create finite difference matrix
        t=time.time()
        beta=(self.dx*omega)/(3*(10**8))
        diag=beta*beta*(self.air_vector+optic_index_walls*optic_index_walls*self.wall_vector)-1j*self.external_sponge_vector-1j*alpha_walls*self.wall_neighbour_vector
        diag=diag-4
        M=self.neighbour_matrix+(diags([diag],[0], None, 'csc', None))   
        if self.verbose:
            print("Elapsed time to import create the FD matrix : " + str(time.time()-t))
        
        #LU factorization
        t=time.time()
        self.LU_decomposition = spla.splu(M) 
        duree = time.time()-t
        self.stats = self.stats + "LU TIME : "+str(duree)+"\n"

        if self.verbose:
            print("Elapsed time for the LU factorization : "+str(duree))
  
    def load(self):
        """
        Load the Bitmap describing the architecture, where 0 = wall and 255 = air.

        Returns
        -------
        np.array
            Flattened map.
        """
        im = Image.open(self.path+".png")
        im = im.convert("L")
        return np.asarray(im)
    

    def wall_dummy(self):
        """
        Create the walls' binary map: where there is a wall -> 1, no wall ->0
        Returns
        -------
        None.

        """
        self.wall_vector=(np.where((self.im)==0,1,0)).reshape(self.Nx*self.Ny)
    
    def wall_neighbour_dummy(self):
        """
        Create the walls' neighbour binary map : wall and neighbour pixels -> 1, no wall in the neighbourhood ->0

        Returns
        -------
        None.

        """
        self.wall_neighbour_vector=np.zeros(self.Nx*self.Ny)
        for i in range(self.Nx*self.Ny):
            
            if self.wall_vector[i]==1:
                for k in [i,i+1,i-1,i-self.Nx,i+self.Nx]:
                    self.wall_neighbour_vector[k]=1
        
    def air_dummy(self):
        """
        Create the air's binary map : where there is a wall -> 0, no wall ->1

        Returns
        -------
        None.

        """
        self.air_vector=(np.where((self.im)==255,1,0)).reshape(self.Nx*self.Ny)
        

    def external_absorbing_layer(self,L = 50,sigmax = 5):
        """
        Creates a vector describe the external absorbing layer (with non-null conductivity)
        to avoid fictitious reflection on the boundaries.
        
        Parameters
        ----------
        L : int, optional
            Length of the absorbing layer. The default is 50.
        sigmax : float, optional
            The conductivity of the external "sponge" layer. The default is 5.

        Returns
        -------
        None.

        """
        res=np.zeros((self.Ny,self.Nx))
        for x in range(L):
            for y in range(self.Ny):
                d=min(x,y,self.Ny-1-y,self.Nx-1-x)
                res[y][x]=(1-(float(d)/float(L)))*sigmax
                res[y][self.Nx-1-x]=(1-(float(d)/float(L)))*sigmax
        
        for y in range(L):
            for x in range(self.Nx):
                d=min(x,y,self.Ny-1-y,self.Nx-1-x)
                res[y][x]=(1-(float(d)/float(L)))*sigmax
                res[self.Ny-1-y][x]=(1-(float(d)/float(L)))*sigmax
    
        self.external_sponge_vector=res.reshape(self.Nx*self.Ny)
    

    def create_adjacency_matrix(self):
        """
        Creates a sparse (CSC format) matrix with 4 diagonals of 1 :
                - diagonal i=j+1
                - diagonal i=j-1
                - diagonal i=j+Nx
                - diagonal i=j-Nx 
                
                The matrix is null anywhere else

        Returns
        -------
        None.

        """
        data=np.zeros(self.Nx*self.Ny*5,dtype=np.complex64)        
        column=np.zeros(self.Nx*self.Ny*5)
        lines=np.zeros(self.Nx*self.Ny*5)
        counter=0
        a=1
        
        for x in range(self.Nx):
            for y in range(self.Ny):
                k=y*self.Nx+x
                if x!=0 :
                    #Left
                    data[counter]=a
                    column[counter]=k-1
                    lines[counter]=k
                    counter+=1
                if x<=(self.Nx-2):
                    #Right
                    data[counter]=a
                    column[counter]=k+1
                    lines[counter]=k
                    counter+=1
                if y!=0:
                    #Up
                    data[counter]=a
                    column[counter]=k-self.Nx
                    lines[counter]=k
                    counter+=1
                if y<=(self.Ny-2):
                    #Down
                    data[counter]=a
                    column[counter]=k+self.Nx
                    lines[counter]=k
                    counter+=1

        data=data[:counter]
        lines=lines[:counter]
        column=column[:counter]
        self.neighbour_matrix=csc_matrix((data, (lines, column)), shape=(self.Nx*self.Ny, self.Nx*self.Ny))
        
   
    def solve(self,x,y, generate_bitmap = False):
        """
        Compute the electromagnectic field generated by a given source point
        of intensity 1

        Parameters
        ----------
        x : int
            x-coordinate.
        y : int
            y coordinate.
        generate_bitmap : boolean, optional
            if True, generate a .png file representing the field. The default value is false.

        Returns
        -------
        Psi : np.array
            Electromagnetic field, normalised by the source power.

        """
        source_point = y*self.Nx +x
        if self.verbose:
            print("Propagation from point ("+str(source_point%self.Nx)+","+str(source_point//self.Nx)+")")
            print("Wavelength : " +str(self.wavelength) +"m")
            print("Optic index of the walls : "+str(self.optic_index_walls))
            print("Alpha walls (diffusive index) : "+str(self.alpha_walls))
        
        #source vector
        S=np.array([0]*self.Nx*self.Ny)
        S[source_point]=-1
        
        #System resolution
        t=time.time()
        self.E = self.LU_decomposition.solve(S) 
        duree = time.time()-t
        self.sum += duree
        self.solve_counter+= 1
        if self.verbose:
            print("Elapsed time for the resolution : "+str(duree))
        
        #Layout
        Psi2=np.absolute(self.E)
        Psi3 = Psi2*Psi2
        Psi3b = (1/Psi3[source_point])*Psi3
        Psi=Psi3b.reshape((self.Ny,self.Nx))
        if generate_bitmap:
            plt.imshow(np.power(Psi, 0.15), cmap="seismic", origin="upper")
            plt.savefig("output/"+self.name+str(x)+"_"+str(y)+".png")
            plt.show()
        return Psi
        
        
        
    def gain_matrix(self, list_of_points):
        """
        Parameters
        ----------
        list_of_points : list of tuples (x,y)
            The list of coordinates of the points.

        Returns
        -------
        M : np.array
            Return the gain main matrix from all points to all other points.

        """
        
        p = len(list_of_points)
        M=np.zeros((p,p))
        
        for i in range(p):
            x,y = list_of_points[i]
            field = self.solve(x,y)
            
            for j in range(p):
                u,v = list_of_points[j]
                M[i][j]=field[v][u]
        
        return M
    
    def export_stats(self):
        """
        Generates a file with the simulator's statistics

        Returns
        -------
        None.

        """
        self.stats = self.stats+ "Gain matrix, Average SolveTime : "+str((self.sum/self.solve_counter))+"\n"
        f = open("output/"+self.name+"_statistics.txt","w") 
        f.write(self.stats)
        f.close()
                


