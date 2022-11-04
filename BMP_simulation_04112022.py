#%%
# Simulation: Maxwell - Bloch - Provotorov, Dipole Dipole Interaction and Bo Inhomogenety
# Author: Vineeth Thalakottoor (vineethfrancis.physics@gmail.com)
# Created: 12 November 2021
# Modified: 4 November 2022
# https://github.com/VThalakottoor/MASER.git
#%%

'''
Read Me
This program can solve Maxwell - Bloch - Provotorov, along with dipole-dipole interaction and Bo inkomogenety. 
This is a beta program, its not that user friendly.
Go to line 579 for setting the parameters
output files:
1) Mxi.txt save (t, np.sqrt((np.sum(M[0:nx*ny*nz*3:3]))**2 + (np.sum(M[1:nx*ny*nz*3:3]))**2), np.sum(M[2:nx*ny*nz*3:3])) in defined instant of time (line 72)
2) Bdip.txt save dipolar field and magnetization at defiend time # caution consume a lot of space
row 7*i + 0 : time, t
row 7*i + 1 : Bdipole_X (t)
row 7*i + 2 : Bdipole_Y (t)
row 7*i + 3 : Bdipole_X (t)
row 7*i + 4 : M_X (t)
row 7*i + 5 : M_Y (t)
row 7*i + 6 : M_Z (t)
use the the file splitBdip.sh to split each into different files
3) dataMx.txt save average Mx 
   dataMy.txt save average My
   dataMz.txt save average Mz
   datat.txt save time
4) dataMxall.txt save individual Mx
   dataMyall.txt save individual My
   dataMzall.txt save individual Mz
5) When simulation complete it will plot average transverse and longitudinal magnetization   

For any question write to me

make two folder codes and data. save this program in codes and run it. Output will be in folder data.
'''

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import matplotlib as mpl
mpl.use('Agg')
import sys
import math
import random
import os
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
np.set_printoptions(threshold=np.inf)

Counter = 0 # Counter for counting the time points used by integrator

#%%
# MPI
MPI = False # MPI only in supercomputer
if MPI:
    from mpi4py import MPI # MPI
    comm = MPI.COMM_WORLD # MPI
    my_rank = comm.Get_rank() # MPI
    p = comm.Get_size() # MPI

#%%
# Function for plotting
def sub_fun1(save,no,xlist,ylist,stylelist,labellist,xlabel,ylabel,title,folder,name):
    plt.figure(no,figsize=(25,10))
    for (x,y,style,labeling) in zip(xlist,ylist,stylelist,labellist):
        plt.plot(x,y,style,linewidth=1.0,markersize=5,label=labeling)
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.xticks(weight='bold', size=20)
    plt.yticks(weight='bold', size=20)
    #plt.xlim(0.0,0.0005)
    #plt.ylim(-20e-22,20e-22)
    plt.legend(fontsize=20,frameon=False)
    plt.xlabel(xlabel,fontsize=20, color='black',fontweight='bold')
    plt.ylabel(ylabel,fontsize=20, color='black',fontweight='bold')
    plt.title(title, fontsize=20, color='black',fontweight='bold')
    
    if save:
        plt.savefig(folder + '/' + name + '.png',transparent=False, dpi=300, bbox_inches='tight')
#%%
# Maxwell-Bloch-Provotrov Equation
def sub_fun2(t,M,Mo,nx,ny,nz,W1dinho,Winho_Dist,paddx,paddy,paddz,maskxyz,Gamma,A,B,Woff,M_lattice,
            Mz_in,Rdi,Rne,K,permeability,spacing,R1,R2,model_B_or_A,dpeffect,rdeffect,rpeffect,disk_or_cylinder,save_Bdip,save_Mxi,save_Myi,save_Mzi,R2_polarization,foldername):
    # save M at each time step
    global Counter
    if Counter%5 == 1:
        save_Mxi.write("%.10f %.10f %.10f \n" % (t, np.sqrt((np.sum(M[0:nx*ny*nz*3:3]))**2 + (np.sum(M[1:nx*ny*nz*3:3]))**2), np.sum(M[2:nx*ny*nz*3:3])))
        save_Mxi.flush()
        os.fsync(save_Mxi.fileno())        
        #save_Myi.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f \n" % (t, My[0,0,15], My[0,0,16],My[2,0,0],My[1,0,0],My[0,2,0],My[0,1,0]))
        #save_Mzi.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f \n" % (t, Mz[0,0,15], Mz[0,0,16],Mz[2,0,0],Mz[1,0,0],Mz[0,2,0],Mz[0,1,0]))
    
        
    # Dipolar
    Wdp = np.zeros((nx*ny*nz*3))
    if dpeffect:
        Wdp = sub_fun3(t,M,nx,ny,nz,K,permeability,spacing,paddx,paddy,paddz,Gamma,maskxyz,Wdp,foldername,save_Mxi,save_Myi,save_Mzi)
        if disk_or_cylinder:
            Wdp[0:nx*ny*nz*3:3] =  Wdp[0:nx*ny*nz*3:3] + (-Gamma * permeability/3.0) * np.average(-M[0:nx*ny*nz*3:3]) 
            Wdp[1:nx*ny*nz*3:3] =  Wdp[1:nx*ny*nz*3:3] + (-Gamma * permeability/3.0) * np.average(-M[1:nx*ny*nz*3:3]) 
            Wdp[2:nx*ny*nz*3:3] =  Wdp[2:nx*ny*nz*3:3] +  (-Gamma * permeability/3.0) * 2.0 * np.average(M[2:nx*ny*nz*3:3]) 
        else:
            Wdp[0:nx*ny*nz*3:3] =  Wdp[0:nx*ny*nz*3:3] + (-Gamma * permeability/6.0) * np.average(-M[0:nx*ny*nz*3:3])
            Wdp[1:nx*ny*nz*3:3] =  Wdp[1:nx*ny*nz*3:3] + (-Gamma * permeability/6.0) * np.average(-M[1:nx*ny*nz*3:3])
            Wdp[2:nx*ny*nz*3:3] =  Wdp[2:nx*ny*nz*3:3] +  (-Gamma * permeability/6.0) * 2.0 * np.average(M[2:nx*ny*nz*3:3]) 
            
        # Save Time
        if Counter%10 == 1: # save data at 10th iteration
            save_Bdip.write(str(t))   
            save_Bdip.write("\n")  
            # Save B Dipole   
            save_Bdip.write(str(Wdp[0:nx*ny*nz*3:3]).replace('[', '').replace(']', '').replace('\n',''))
            save_Bdip.write("\n")
            save_Bdip.write(str(Wdp[1:nx*ny*nz*3:3]).replace('[', '').replace(']', '').replace('\n',''))
            save_Bdip.write("\n")
            save_Bdip.write(str(Wdp[2:nx*ny*nz*3:3]).replace('[', '').replace(']', '').replace('\n',''))
            save_Bdip.write("\n")
            # Save M
            save_Bdip.write(str(M[0:nx*ny*nz*3:3]).replace('[', '').replace(']', '').replace('\n',''))
            save_Bdip.write("\n")
            save_Bdip.write(str(M[1:nx*ny*nz*3:3]).replace('[', '').replace(']', '').replace('\n',''))
            save_Bdip.write("\n")
            save_Bdip.write(str(M[2:nx*ny*nz*3:3]).replace('[', '').replace(']', '').replace('\n',''))
            save_Bdip.write("\n")

    Wrd = np.zeros((nx*ny*nz*3))
    if rdeffect:
        Wrd = sub_fun4(M,A,B,nx,ny,nz,Wrd)        

    # Frequency        
    omegax = np.zeros((nx*ny*nz))
    omegay = np.zeros((nx*ny*nz))
    omegaz = np.zeros((nx*ny*nz))
    omegax = Wrd[0:nx*ny*nz*3:3] + Wdp[0:nx*ny*nz*3:3]
    omegay = Wrd[1:nx*ny*nz*3:3] + Wdp[1:nx*ny*nz*3:3]
    omegaz = Woff + Wdp[2:nx*ny*nz*3:3] 
    
    RP = np.zeros((nx*ny*nz))
    Mo_st = np.zeros((nx*ny*nz))
    if rpeffect:
        RP,Mo_st = sub_fun5(t,M,M_lattice,Mz_in,Rdi,nx,ny,nz,RP,model_B_or_A)
    
    Mdot = np.zeros((nx*ny*nz*3))
    
    if R2_polarization:
        # R2 changing with polarization
        Mdot[0:nx*ny*nz*3:3] = -omegaz*M[1:nx*ny*nz*3:3] - omegay*M[2:nx*ny*nz*3:3] - R2*(1-(Mo_st/Mz_in)**2)*M[0:nx*ny*nz*3:3]
        Mdot[1:nx*ny*nz*3:3] = omegaz*M[0:nx*ny*nz*3:3] + omegax*M[2:nx*ny*nz*3:3] - R2*(1-(Mo_st/Mz_in)**2)*M[1:nx*ny*nz*3:3]
        Mdot[2:nx*ny*nz*3:3] = omegay*M[0:nx*ny*nz*3:3] - omegax*M[1:nx*ny*nz*3:3] + RP * Rne
    
    else:
        # R2 constant with polarization
        Mdot[0:nx*ny*nz*3:3] = -omegaz*M[1:nx*ny*nz*3:3] - omegay*M[2:nx*ny*nz*3:3] - R2*M[0:nx*ny*nz*3:3]
        Mdot[1:nx*ny*nz*3:3] = omegaz*M[0:nx*ny*nz*3:3] + omegax*M[2:nx*ny*nz*3:3] - R2*M[1:nx*ny*nz*3:3]
        Mdot[2:nx*ny*nz*3:3] = omegay*M[0:nx*ny*nz*3:3] - omegax*M[1:nx*ny*nz*3:3] + RP * Rne
    
    
    
    print("t = %.10f" %t, end='\r')

    Counter = Counter + 1
    return Mdot
#%%
# Function: Dipole
def sub_fun3(t,M,nx,ny,nz,K,permeability,spacing,paddx,paddy,paddz,Gamma,maskxyz,Wdp,foldername,save_Mxi,save_Myi,save_Mzi):
    M_ = np.reshape(M,(nx,ny,nz,3))
    Mx = M_[:,:,:,0]
    My = M_[:,:,:,1]
    Mz = M_[:,:,:,2]
    
    # Zero padding
    Mx = np.pad(Mx, pad_width=((0,paddx),(0,paddy),(0,paddz)), mode='constant', constant_values=0)
    My = np.pad(My, pad_width=((0,paddx),(0,paddy),(0,paddz)), mode='constant', constant_values=0)
    Mz = np.pad(Mz, pad_width=((0,paddx),(0,paddy),(0,paddz)), mode='constant', constant_values=0)
    # FFT
    Mx_ = np.fft.fftn(Mx)
    My_ = np.fft.fftn(My)
    Mz_ = np.fft.fftn(Mz)
    Mx_ = np.fft.fftshift(Mx_)
    My_ = np.fft.fftshift(My_)
    Mz_ = np.fft.fftshift(Mz_)
    Mx_ = (1/6) * (1-3 * K[:,:,:,2]) * Mx_ * permeability # Equation 4
    My_ = (1/6) * (1-3 * K[:,:,:,2]) * My_ * permeability
    Mz_ = (2/6) * (3 * K[:,:,:,2] - 1) * Mz_ * permeability
    # Inverse Transform
    Mx_ = np.fft.ifftshift(Mx_)
    My_ = np.fft.ifftshift(My_)
    Mz_ = np.fft.ifftshift(Mz_)
    Mx = np.fft.ifftn(Mx_).real
    My = np.fft.ifftn(My_).real
    Mz = np.fft.ifftn(Mz_).real
    # Remove padding
    Mx = Mx[0:nx,0:ny,0:nz]
    My = My[0:nx,0:ny,0:nz]
    Mz = Mz[0:nx,0:ny,0:nz]
    Wdp[0:nx*ny*nz*3:3] = -Gamma * np.reshape(Mx,(1,nx*ny*nz)) * maskxyz
    Wdp[1:nx*ny*nz*3:3] = -Gamma * np.reshape(My,(1,nx*ny*nz)) * maskxyz
    Wdp[2:nx*ny*nz*3:3] = -Gamma * np.reshape(Mz,(1,nx*ny*nz)) * maskxyz
    return Wdp
#%%
# Function: Radiation damping
def sub_fun4(M,A,B,nx,ny,nz,Wrd):
    Wrd[0:nx*ny*nz*3:3] = A * np.sum(M[0:nx*ny*nz*3:3]) - B * np.sum(M[1:nx*ny*nz*3:3])
    Wrd[1:nx*ny*nz*3:3] = B * np.sum(M[0:nx*ny*nz*3:3]) + A * np.sum(M[1:nx*ny*nz*3:3])
    return Wrd
#%%
# Function: Repolarization
def sub_fun5(t,M,M_lattice,Mz_in,Rdi,nx,ny,nz,RP,model_B_or_A):
    if model_B_or_A:
        Mo_st = M_lattice # Model B    
    else:
        Mo_st= M_lattice - (M_lattice - Mz_in) * np.exp(-Rdi*t) # Model A
    RP = Mo_st - M[2:nx*ny*nz*3:3] 
    return RP,Mo_st

#%%
# Main Function
def main_fun(
        nu_H,nu_D,nu_E,
        MPI, save, boltzmann, inhomog, model_B_or_A, dpeffect, rdeffect, rpeffect, R2_polarization, mask, maskType, disk_or_cylinder, real_data,
	    nx, ny, nz, a, b, c, r, Vol,
	    avagardo, mp, kb, permeability, hbar, Gamma,
	    Bo, T, Mo, etaQ, inhomo_width,
	    offset, PhiFLIP, 
	    R1, R2, 
	    Psi, Tnee, Tdi, paddx, paddy, paddz,
	    ti, tf, AcqN, AcqTime_r, 
	    rank, data_file, simtype, date, M_lattice_scale):
    
    # section: create folder
    if save:
        if MPI:
            path = "/ccc/scratch/cont003/gen12736/thalakov/TGCC_"
        else:
            path = os.getcwd()
        
        Exp_Number = 1
        while True:
            try:
                if MPI:
                    foldername = path + "_rank_" + str(rank) + date + str(Exp_Number) # Super Computer
                else:
                    foldername = path + "/../data/" + simtype + "_rank_" + str(rank) + date + str(Exp_Number) # Personal Computer  
                    prev_data =  path + "/../data/" + simtype + "_rank_" + str(rank) + date + str(174) # Personal Computer, 174 corresponds to the file from which you need to restart the simulation                  
                os.mkdir(foldername)
                break
            except OSError as err:
                Exp_Number = Exp_Number + 1
                
        save_file = open(foldername + "/info.txt", "w")
        save_file.write("Vineeth Thalakottoor")
        save_file.close()
        
        save_file = open(foldername + "/info.txt", "a")
        save_file.write("\nDirectory = " + foldername)
        save_file.close()
        
    # section: lattice
    unit_no = nx*ny*nz
    unit_vol = Vol/unit_no
    spacing =  ((Mo)/200.0)**(1.0/3.0) #float(sys.argv[1]) #unit_vol**(1.0/3.0) # Default 10.0e-10 

    if mask:
        xp = np.linspace(-(nx-1)/2,(nx-1)/2,nx,endpoint=True)
        yp = np.linspace(-(ny-1)/2,(ny-1)/2,ny,endpoint=True)
        zp = np.linspace(-(nz-1)/2,(nz-1)/2,nz,endpoint=True)
        xpr, ypr, zpr = np.meshgrid(xp,yp,zp,sparse=False,indexing='ij') 
        xprc, yprc, zprc = np.meshgrid(xp,yp,zp,sparse=False,indexing='ij')
        if maskType == "Circle": maskc = xpr**2/(a**2) + ypr**2/(b**2) + zpr**2/(c**2) <= 1 
        if maskType == "Cylinder": maskc = xpr**2/(a**2) + ypr**2/(b**2) <= 1 
        if maskType == "Both": 
            maskc1 = xpr**2/(a**2) + ypr**2/(b**2) <= 1
            maskc = np.full(xprc.shape, False) 
            for i in range(400):
                maskc2 = (xpr - random.randrange(-a, a))**2/(r**2) + (ypr - random.randrange(-b, b))**2/(r**2) + (zpr - random.randrange(-c, c))**2/(r**2) <= 1
                maskc3 = maskc1 * maskc2
                maskc = maskc + maskc3
            
        maskxyz = maskc.astype(int)
    
        xprc = xprc * maskxyz; yprc = yprc * maskxyz; zprc = zprc * maskxyz
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.scatter(xprc, yprc, zprc)
    
        ax.set_xlim(-nx/2, nx/2)
        ax.set_ylim(-ny/2,ny/2)
        ax.set_zlim(-nz/2,nz/2)
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        ax.view_init(90, 0)
        plt.savefig(foldername + '/mask1.png',transparent=True, dpi=300, bbox_inches='tight')
        ax.view_init(-160, 30)
        plt.savefig(foldername + '/mask2.png',transparent=True, dpi=300, bbox_inches='tight')
    
    else:
        maskxyz = np.ones((nx,ny,nz))
    
    no_particles = np.sum(maskxyz)
    
    # section: constants
    Wo = -Gamma * Bo
    
    # section: pulse
    Wrf = Wo - 2.0*np.pi*offset
    Woff = (Wo-Wrf) * np.ones(nx*ny*nz)
    
    # sectio: inhomogenety
    W1dinho = np.zeros(nx*ny*nz)
    Winho_Dist = np.ones(nx*ny*nz)
    if inhomog:
        if False:
            a = 4
            Winho = np.linspace(-a*inhomo_width,a*inhomo_width,nz)
            dWinho = Winho[1]-Winho[0]
		    
            Winho_Gaussian = np.exp(-Winho**2/(2*inhomo_width**2)) 
            Winho_Lorn = (np.pi*inhomo_width/2)*(inhomo_width/(2*np.pi))/((Winho)**2+(inhomo_width/2)**2)
            Winho_Dist1d = Winho_Lorn
		    
            plt.figure(6, figsize=(25,10))
            plt.plot(Winho,Winho_Gaussian,'o-', linewidth=1.0 , markersize=5, label = r"Gaussian")
            plt.plot(Winho,Winho_Lorn,'*-', linewidth=1.0 , markersize=5, label = r"Lorentzian")
            #plt.grid(True)
            plt.axhline(0, color='black',linewidth=1)
            plt.axvline(0, color='black',linewidth=1)
            plt.xticks(weight='bold', size=20)
            plt.yticks(weight='bold', size=20)
            #plt.xlim(0.0,0.0005)
            #plt.ylim(-20e-22,20e-22)
            plt.legend(fontsize=20,frameon=False)
            plt.xlabel("Frequency",fontsize=20, color='black',fontweight='bold')
            plt.ylabel("Distribution",fontsize=20, color='black',fontweight='bold')
            plt.title("Bo Inhomogenety", fontsize=20, color='black',fontweight='bold')
		            
            if save:
                plt.savefig(foldername + '/Bo_inhomo.png',transparent=True, dpi=300, bbox_inches='tight')
		    
            W3dinho = np.ones((nx,ny,nz))
            Winho_Dist3d = np.ones((nx,ny,nz))
            for i in range(nz):
                W3dinho[:,:,i] = Winho[i]
                Winho_Dist3d[:,:,i] = Winho_Dist1d[i]
		    
            W1dinho = np.reshape(W3dinho,(nx*ny*nz,))
            Winho_Dist = np.reshape(Winho_Dist3d,(nx*ny*nz,))
		    
        else:
            a = 4
            W1dinho = np.linspace(-a*inhomo_width,a*inhomo_width,nx*ny*nz)
            dWinho = W1dinho[1]-W1dinho[0]
		   
            Winho_Gaussian = np.exp(-W1dinho**2/(2*inhomo_width**2)) 
            Winho_Lorn = (np.pi*inhomo_width/2)*(inhomo_width/(2*np.pi))/((W1dinho)**2+(inhomo_width/2)**2)
            Winho_Dist = Winho_Lorn	   

            plt.figure(6, figsize=(25,10))
            plt.plot(W1dinho,Winho_Gaussian,'o-', linewidth=1.0 , markersize=5, label = r"Gaussian")
            plt.plot(W1dinho,Winho_Lorn,'*-', linewidth=1.0 , markersize=5, label = r"Lorentzian")
            #plt.grid(True)
            plt.axhline(0, color='black',linewidth=1)
            plt.axvline(0, color='black',linewidth=1)
            plt.xticks(weight='bold', size=20)
            plt.yticks(weight='bold', size=20)
            #plt.xlim(0.0,0.0005)
            #plt.ylim(-20e-22,20e-22)
            plt.legend(fontsize=20,frameon=False)
            plt.xlabel("Frequency",fontsize=20, color='black',fontweight='bold')
            plt.ylabel("Distribution",fontsize=20, color='black',fontweight='bold')
            plt.title("Bo Inhomogenety", fontsize=20, color='black',fontweight='bold')
		            
            if save:
                plt.savefig(foldername + '/Bo_inhomo.png',transparent=True, dpi=300, bbox_inches='tight')
                    
    # section: radiation damping
        
    Xi = Gamma*permeability*etaQ/2
    RDrate = Xi * Mo
    A = np.sin(Psi * np.pi/180.0) * (Xi/Vol) * unit_vol
    B = np.cos(Psi * np.pi/180.0) * (Xi/Vol) * unit_vol
    
    # section: boltzman
    if boltzmann:
        theta = np.linspace(0,180.0,10000,endpoint=True) # Polar angle
        E = -Bo*mp*np.cos(theta*np.pi/180.0) # Energy of magnetic dipole
        pro_theta = np.exp(-E/(kb*T)) # Probabilty
        pro_theta = pro_theta/np.sum(pro_theta) # Probabilty normalized
        #print("Sum of all probability =",np.sum(pro_theta)) 
    
        Theta = np.random.choice(theta, size=(nx,ny,nz), p=pro_theta) # Polar angle
        Phi = np.random.uniform(low=0.0, high=360.0, size=(nx, ny, nz)) # Azimuthal angle
        #Phi = 0.0 * np.ones((nx,ny,nz))
    
        plt.figure(1, figsize=(25,10))
        plt.subplot(121)
        plt.plot(theta,pro_theta,'-.r', linewidth=3.0)
        plt.grid(True)
        plt.xticks(weight='bold', size=20)
        plt.yticks(weight='bold', size=20)
        plt.xlabel("Angle (deg)",fontsize=20, color='black',fontweight='bold')
        plt.ylabel("Probabilty",fontsize=20, color='black',fontweight='bold')
        plt.title("Boltzmann Distribution", fontsize=20, color='black',fontweight='bold')
    
        plt.subplot(122)
        plt.plot(theta,E/(kb*T),'-.b', linewidth=3.0)
        plt.grid(True)
        plt.xticks(weight='bold', size=20)
        plt.yticks(weight='bold', size=20)
        plt.xlabel("Angle (deg)",fontsize=20, color='black',fontweight='bold')
        plt.ylabel("E/(kb*T)",fontsize=20, color='black',fontweight='bold')
        plt.title("Boltzmann Distribution", fontsize=20, color='black',fontweight='bold')
        plt.savefig(foldername + '/Boltmann.png',transparent=True, dpi=300, bbox_inches='tight')
    
        plt.figure(2, figsize=(25,17))
        plt.subplot(211)
        plt.hist(np.reshape(Theta,(nx*ny*nz,)), bins = 180, density=False)
        plt.grid(True)
        plt.xticks(weight='bold', size=20)
        plt.yticks(weight='bold', size=20)
        plt.xlabel("Angle (deg)",fontsize=20, color='black',fontweight='bold')
        plt.ylabel("Count",fontsize=20, color='black',fontweight='bold')
        plt.title("Histogram Of Randomly generated Theta (polar angle)", fontsize=20, color='black',fontweight='bold')
    
        plt.subplot(212)
        plt.hist(np.reshape(Phi,(nx*ny*nz,)), bins = 360, density=False)
        plt.grid(True)
        plt.xticks(weight='bold', size=20)
        plt.yticks(weight='bold', size=20)
        plt.xlabel("Angle (deg)",fontsize=20, color='black',fontweight='bold')
        plt.ylabel("Count",fontsize=20, color='black',fontweight='bold')
        plt.title("Histogram Of Randomly generated Phi (azimuthal angle)", fontsize=20, color='black',fontweight='bold')
        plt.savefig(foldername + '/Boltmann_Angles.png',transparent=True, dpi=300, bbox_inches='tight')
    
    else:
        Theta = 0.0 * np.ones((nx,ny,nz))
        Phi = 0.0 * np.ones((nx,ny,nz))
    
    # section: initial distribution
    Mtemp = np.zeros((nx,ny,nz,3)) 
    M = np.zeros((nx,ny,nz,3)) 
    Mtemp[:,:,:,0] = Mo * np.sin(Theta * np.pi/180.0) * np.cos(Phi * np.pi/180.0) * maskxyz 
    Mtemp[:,:,:,1] = Mo * np.sin(Theta * np.pi/180.0) * np.sin(Phi * np.pi/180.0) * maskxyz 
    Mtemp[:,:,:,2] = Mo * np.cos(Theta * np.pi/180.0) * maskxyz
    
    M[:,:,:,1] = Mtemp[:,:,:,1]
    M[:,:,:,1] = Mtemp[:,:,:,1] * np.cos(PhiFLIP * np.pi/180.0) - Mtemp[:,:,:,2] * np.sin(PhiFLIP * np.pi/180.0)
    M[:,:,:,2] = Mtemp[:,:,:,1] * np.sin(PhiFLIP * np.pi/180.0) + Mtemp[:,:,:,2] * np.cos(PhiFLIP * np.pi/180.0)

    M = np.reshape(M,(nx*ny*nz*3,))
    if True:
        M[0:nx*ny*nz*3:3] = M[0:nx*ny*nz*3:3] * Winho_Dist
        M[1:nx*ny*nz*3:3] = M[1:nx*ny*nz*3:3] * Winho_Dist
        M[2:nx*ny*nz*3:3] = M[2:nx*ny*nz*3:3] * Winho_Dist
    else:
        # For continuing simulation from previous simulation. Last M value will be the input of current simulation. go to line 246 to set the folder path. Dont forget to set new initial and final time
        dfx = pd.read_csv(prev_data + "/dataMxall.txt", delim_whitespace=True, header=None)
        dfy = pd.read_csv(prev_data + "/dataMyall.txt", delim_whitespace=True, header=None)
        dfz = pd.read_csv(prev_data + "/dataMzall.txt", delim_whitespace=True, header=None)
        M[0:nx*ny*nz*3:3] = dfx.loc[:,8749].to_numpy() # 8749 means the column corresponding to last value to Mx, which will intial value for new program
        M[1:nx*ny*nz*3:3] = dfy.loc[:,8749].to_numpy() # 8749 means the column corresponding to last value to My, which will intial value for new program
        M[2:nx*ny*nz*3:3] = dfz.loc[:,8749].to_numpy() # 8749 means the column corresponding to last value to Mz, which will intial value for new program      
         
    np.savetxt(foldername + '/Min_data.txt', M, fmt='%1.10e') # save initial magnetization values
    
    # section: repolarization
    Rne = 1.0/Tnee 
    Rdi = 1.0/Tdi
    
    Mz_in = np.zeros((nx,ny,nz))
    Mz_in = M[2:nx*ny*nz*3:3]
    
    M_lattice = M_lattice_scale * Mz_in 
    np.savetxt(foldername + '/Mlattice_data.txt', M_lattice, fmt='%1.10e')
    
    # section: k-space
    kx = np.linspace(-1/(2), 1/(2), nx + paddx, endpoint=True) # Attention: Unit spacing is 1
    ky = np.linspace(-1/(2), 1/(2), ny + paddy, endpoint=True) 
    kz = np.linspace(-1/(2), 1/(2), nz + paddz, endpoint=True) 
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij') 
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    KX_unit = np.nan_to_num(KX/Kmag , nan = 0.0)
    KY_unit = np.nan_to_num(KY/Kmag , nan = 0.0)
    KZ_unit = np.nan_to_num(KZ/Kmag , nan = 0.0)
    K = np.zeros((nx + paddx, ny + paddy, nz + paddz, 3))
    K[:,:,:,0] = KX_unit * KX_unit
    K[:,:,:,1] = KY_unit * KY_unit
    K[:,:,:,2] = KZ_unit * KZ_unit
    
    # section: save to file
    if save:
        save_file = open(foldername + "/info.txt", "a")
        save_file.write("\nModel B (True) and Model A (False) = %r" %model_B_or_A)
        save_file.write("\nmy_rank = %d" %rank)
        save_file.write("\nnx,ny,nz = %d,%d,%d" %(nx,ny,nz))
        save_file.write("\nSpacing = %e" %spacing)
        save_file.write("\nTotal Voulme = %e" %Vol)
        save_file.write("\nUnit Voulme = %e" %unit_vol)
        save_file.write("\nBo = %f" %Bo)
        save_file.write("\nWo = %f" %Wo)
        save_file.write("\nMo = %f" %Mo)
        save_file.write("\nT = %f" %T)
        save_file.write("\nOffset = %e" %offset)
        save_file.write("\nPhiFLIP = %f" %PhiFLIP)
        save_file.write("\nRd_rate = %e" %RDrate)
        save_file.write("\nRd_Xi = %e" %Xi)
        save_file.write("\nRd_etaQ = %e" %etaQ)
        save_file.write("\nRd_Psi = %f" %Psi)
        save_file.write("\nRd_A = %e" %A)
        save_file.write("\nRd_B = %e" %B)
        save_file.write("\nR1 = %e" %R1)
        save_file.write("\nR2 = %e" %R2) 
        save_file.write("\nInhomogeneous width = %e" %inhomo_width)       
        save_file.write("\nTnee = %f" %Tnee)
        save_file.write("\nTdi = %f" %Tdi)
        save_file.write("\nM lattice scale = %f" %M_lattice_scale)
        save_file.write("\nPadding X = %d" %paddx)
        save_file.write("\nPadding Y = %d" %paddy)
        save_file.write("\nPadding Z = %d" %paddz)
        save_file.write("\nInhomogenety = %r" %inhomog)
        save_file.write("\nBoltzmann = %r" %boltzmann)
        save_file.write("\nRadiation damping field = %r" %rdeffect)
        save_file.write("\nRepolarization effect = %r" %rpeffect)
        save_file.write("\nDipole field = %r" %dpeffect)
        save_file.write("\nR2 dependence on polarization = %r" %R2_polarization)                                                      
        save_file.write("\nMask = %r " %mask + ", Mask Type = " + maskType + ",Disk (True) or Cylinder (False) = %r" %disk_or_cylinder + ", Number of particles = %d" %no_particles)
        save_file.close()
        
    # section: ode solve
    maskxyz = np.reshape(maskxyz,(nx*ny*nz,))
    np.savetxt(foldername + '/mask_data.txt', maskxyz)
    
    save_Bdip =open(foldername + "/Bdip.txt", "a")
    save_Mxi =open(foldername + "/Mxi.txt", "a")
    save_Myi =open(foldername + "/Myi.txt", "a+")
    save_Mzi =open(foldername + "/Mzi.txt", "a+")
    
    start_time = time.time()
    sol = solve_ivp(sub_fun2, [ti,tf], M, method = 'DOP853', t_eval= np.linspace(ti, tf, num=AcqN, endpoint=True), args = (Mo,nx,ny,nz,W1dinho,Winho_Dist,paddx,paddy,paddz,maskxyz,Gamma,A,B,Woff,M_lattice,
            Mz_in,Rdi,Rne,K,permeability,spacing,R1,R2,model_B_or_A,dpeffect,rdeffect,rpeffect,disk_or_cylinder,save_Bdip,save_Mxi,save_Myi,save_Mzi,R2_polarization,foldername), atol = 1e-10, rtol = 1e-10)
    end_time = time.time()
    timetaken = end_time - start_time
    print("Total time = %s seconds " % (timetaken))
    save_file = open(foldername + "/info.txt", "a")
    save_file.write("\nTotal time = %f seconds" %timetaken)
    save_file.close()
    save_Bdip.close()
    save_Mxi.close()
    save_Myi.close()
    save_Mzi.close()
    
    # section: solutions
    t = sol.t
    Mr = sol.y
    
    Mx = np.sum(Mr[0:3*nx*ny*nz:3,:], axis=0)
    My = np.sum(Mr[1:3*nx*ny*nz:3,:], axis=0)
    Mz = np.sum(Mr[2:3*nx*ny*nz:3,:], axis=0)
    Mabs = np.sqrt(Mx**2 + My**2)

    np.savetxt(foldername + '/datat.txt',t, fmt='%1.10e')
    np.savetxt(foldername + '/dataMx.txt',Mx, fmt='%1.10e')
    np.savetxt(foldername + '/dataMy.txt',My, fmt='%1.10e')
    np.savetxt(foldername + '/dataMz.txt',Mz, fmt='%1.10e')

    np.savetxt(foldername + '/dataMxall.txt',Mr[0:3*nx*ny*nz:3,:], fmt='%1.10e')
    np.savetxt(foldername + '/dataMyall.txt',Mr[1:3*nx*ny*nz:3,:], fmt='%1.10e')
    np.savetxt(foldername + '/dataMzall.txt',Mr[2:3*nx*ny*nz:3,:], fmt='%1.10e')
        
    # section: real data
    if real_data:
        dat = np.fromfile(data_file,dtype=np.int32)
        t_r = np.linspace(0,AcqTime_r,dat.shape[0]//2)
        t_r = np.linspace(0,AcqTime_r,dat.shape[0]//2)
        Dt = t_r[1]-t_r[0]

        inter = interp1d(t, Mabs, kind='cubic',fill_value="extrapolate")
        t_inter = np.arange(0,tf,Dt) 
    
        x_r = dat[0::2] + 1j * dat[1::2]
        Mx_r = x_r[0:t_inter.shape[0]].real
        My_r = x_r[0:t_inter.shape[0]].imag
        Ma_r = np.sqrt(Mx_r**2 + My_r**2)
    
    # section: plotting
    sub_fun1(save,no=3,xlist=[t,t,t],ylist=[Mabs,Mx,My],stylelist=["o-","*","+"],labellist=["Mt","Mx","My"],
         xlabel="Time",ylabel="$M_T$",title="FID",folder = foldername, name = "simu_data1") 
    
    sub_fun1(save,no=4,xlist=[t,t],ylist=[Mabs,Mz],stylelist=["o-","."],labellist=["Mt","Mz"], 
         xlabel="Time",ylabel="$M_T$",title="FID",folder = foldername, name = "simu_data2")
    
#%%
# Input Parameters
# If you want to input parameter from terminal when you run prgram, like python BMP_simulation.py parameter_1 parameter_2 ... parameter_i, use int(sys.argv[i]) for integer and float(sys.argv[1]) for flot

model_B_or_A = True # True model B and False Model A
save = True # save into file
boltzmann = False # True: Boltzmann distribution of protons; False: We consider a bulk proton sample coarse-grained and all unit cells has same inital equlibrium magnetization  
inhomog = False # Bo Inhomogenety
dpeffect = True # Dipolar Interaction
rdeffect = True # Radiation damping
rpeffect = True # Repolarisation effect
mask = True # Mask, shape of sample
maskType = "Cylinder" # Shape of smaple: Cylinder (Disk or cylinder), Circle (not tested)
disk_or_cylinder = True # True:Disk and False:Cylinder
R2_polarization = False # R2 dependence on polarization
real_data = False # Compare exmperimental data with real fid data

# Dimension of unit cell
nx = 21 #11
ny = 21 # 11
nz = 7 #21

# Mask
# Elliosoid equation: x**2/a**2 + y**2/b**2 + z**2/c**2 = 1
a = 10 #int(sys.argv[1])
b = 10 #int(sys.argv[1])
c = 10
r = 2

# Volume of unit cell
Vol = 150.0e-9 

# Zero padding
paddx = 100 #int(sys.argv[1])
paddy = 100 #int(sys.argv[2])
paddz = 150 #int(sys.argv[3])

# Constants
avagardo = 6.022e26
mp = 1.4e-26
kb = 1.38e-23
permeability = 4 * np.pi * 1.0e-7
hbar = 1.054e-34
Gamma = 2.675e8

Bo = 6.7
T = 1.5e-3
inhomo_width = 8000.0
offset = 0.0
PhiFLIP = 179.0 #1.0
M_lattice_scale = 0.95 # I used 0.95 for model B and 0.05 for model A, it says what will equlibrium magnetization wrt to intialmagnetization

R1 = 0.0 # Longitudinal relaxation
R2 = 650.0 # float(sys.argv[1]) # Transverse relaxation

scaleMo = 0.001065
Mo = 400.0 #2 * 0.213/scaleMo () 
etaQ = 2.0 * 6.0 / Mo # product of Q-factor and filling factor. Mo * etaQ = constant for maintaing radiation damping rate constant while changing Mo 
Tnee = 0.05 # equilibration time between proton zeeman and electron dipole, in paper this parameter is inverse of gamma_ee,n
Tdi = 1.0 # equilibration time between deuterium zeeman and electron dipole, in paper this parameter is inverse of gamma_st
Psi = 0.0 # Phase between voltage and current in resinant circuit

ti = 0 # Start time
tf = 1.0 # Final time
AcqN = 10000 # Number of points
AcqTime_r = 1 # Final time when you use real data

data_file = "/media/vineeth/23f1aba5-1c33-48e6-9768-d41ffbc8299e/Polarizer6.7/DNP_2016-06-16/998/fid"

simtype = "/bmp_"
date = "_2022_11_04_" 

nu_H = 1 # discard
nu_D = 1 # discard
nu_E = 1 # discard   
    
#%% 
# MPI
if MPI:
    rank = my_rank
    
    '''
    parameter1 = np.array([0.0,0.0,0.0,0.0,0.0])
    parameter2 = np.array([0.008,0.006,0.004,0.002,0.0009])
    parameter3 = np.array([0.001,0.002,0.0033,0.004,0.005])
    
    mpi_i = 0
    for j in range(parameter1.shape[-1]):
        for k in range(parameter2.shape[-1]):
            for l in range(parameter3.shape[-1]):
                if my_rank == mpi_i:
                    Mo = parameter1[j]
                    etaQ = parameter2[k]
                    Tdi_mpi = parameter3[l]
                    print(my_rank,Mo,etaQ,Tdi_mpi)
                mpi_i = mpi_i + 1
    '''

    if rank == 0:
        Mo = 50.0 #0.01
        etaQ = 2.97485875 #14874.29374691

    if rank == 1:
        Mo = 100.0 #0.05
        etaQ = 1.48742937 #2974.85874938

    if rank == 2:
        Mo = 200.0 #0.1
        etaQ = 0.74371469 #1487.42937469

    if rank == 3:
        Mo = 300.0 #0.5
        etaQ = 0.49580979 #297.48587494

    if rank == 4:
        Mo = 400.0 #1.0
        etaQ = 0.37185734 #148.74293747

    if rank == 5:
        Mo = 500.0 #5.0
        etaQ = 0.29748587 #29.7485874

    if rank == 6:
        Mo = 600.0 #10.0
        etaQ = 0.2479049 #14.87429375


    main_fun(
        nu_H,nu_D,nu_E,
        MPI, save, boltzmann, inhomog, model_B_or_A, dpeffect, rdeffect, rpeffect, R2_polarization, mask, maskType, disk_or_cylinder, real_data,
	    nx, ny, nz, a, b, c, r, Vol,
	    avagardo, mp, kb, permeability, hbar, Gamma,
	    Bo, T, Mo, etaQ, inhomo_width,
	    offset, PhiFLIP, 
	    R1, R2, 
	    Psi, Tnee, Tdi, paddx, paddy, paddz,
	    ti, tf, AcqN, AcqTime_r, 
	    rank, data_file, simtype, date, M_lattice_scale)
else:
    rank = 1

    main_fun(
        nu_H,nu_D,nu_E,
        MPI, save, boltzmann, inhomog, model_B_or_A, dpeffect, rdeffect, rpeffect, R2_polarization, mask, maskType, disk_or_cylinder, real_data,
	    nx, ny, nz, a, b, c, r, Vol,
	    avagardo, mp, kb, permeability, hbar, Gamma,
	    Bo, T, Mo, etaQ, inhomo_width,
	    offset, PhiFLIP, 
	    R1, R2, 
	    Psi, Tnee, Tdi, paddx, paddy, paddz,
	    ti, tf, AcqN, AcqTime_r, 
	    rank, data_file, simtype, date, M_lattice_scale)
	    

