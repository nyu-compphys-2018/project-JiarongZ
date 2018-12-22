#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:26:18 2018

@author: zhujiarong
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio

class Brio_wu:
    
    def __init__(self,a,b,Nx,t):
        
        self.a = a
        self.b = b
        self.Nx = int(Nx)
        self.t = t
        
        self.dx = (b - a)/float(Nx)
        self.x = a + self.dx*( np.arange(Nx) + 0.5 )
        self.u = np.zeros((3,Nx))
        self.rho = np.zeros(Nx)
        self.v = np.zeros(Nx)
        self.p = np.zeros(Nx)
        
        self.cfl = 0.475
        self.gamma = 2.0
        
        self.cs = np.zeros(Nx)
        self.lp = np.zeros(Nx)
        self.lm = np.zeros(Nx)
        
    #set up initial conditions
    def setInitcondion(self):
        self.rho[:int(self.Nx/2)]=1.0
        self.rho[int(self.Nx/2):]=0.125        
        self.p[:int(self.Nx/2)]=1.0
        self.p[int(self.Nx/2):]=0.1            
        self.v[:int(self.Nx/2)]=0.0
        self.v[int(self.Nx/2):]=0.0
        
        self.u[0]= self.rho
        self.u[1] = self.rho*self.v  
        self.u[2] = self.u[1]**2/(2*self.u[0]) + self.p/(self.gamma-1)
        
        self.cs = np.sqrt(self.gamma*self.p/self.u[0])
        self.lp = self.v + self.cs
        self.lm = self.v - self.cs
        
    
    def get_dt(self):
        return self.cfl * self.dx / max(np.max( self.lp),np.max(-self.lm ))
       
    def evolve(self,tfinal):
        
        while self.t < tfinal:
            
            dt = self.get_dt()
            if self.t+dt > tfinal:
                dt = tfinal - self.t
                
            LU = self.LU()
            
            for i in range(3):
                self.u[i] += LU[i]*dt

            self.t += dt
            
            self.rho = self.u[0]
            self.v = self.u[1]/self.u[0]
            self.p = (self.gamma-1)*(self.u[2] - 1/2*self.u[1]**2/self.u[0])
            self.cs = np.sqrt(self.gamma*self.p/self.rho)
            self.lp = self.v + self.cs
            self.lm = self.v - self.cs
        
    def LU(self):        
        ap = np.zeros(self.Nx-1)
        am = np.zeros(self.Nx-1)
        
        for i in range(self.Nx-1):
            ap[i] = max(0,self.lp[i],self.lp[i+1])
            am[i] = max(0,-self.lm[i],-self.lm[i+1])
        
        F = np.zeros((3,self.Nx))
        F[0] = self.rho*self.v
        F[1] = self.rho*self.v**2 + self.p
        F[2] = self.v*(1/2*self.rho*self.v**2+self.gamma*self.p/(self.gamma-1))

        FL = np.zeros((3,self.Nx-1))
        FR = np.zeros((3,self.Nx-1))
        UL = np.zeros((3,self.Nx-1))
        UR = np.zeros((3,self.Nx-1))
        FHLL = np.zeros((3,self.Nx-1))
        
        for i in range(3):
            FL[i] = F[i][:-1]
            FR[i] = F[i][1:]        
            UL[i] = self.u[i][:-1]
            UR[i] = self.u[i][1:]
            FHLL[i] = (ap*FL[i] + am*FR[i] - ap*am*(UR[i]-UL[i]))/(ap+am)
    
        LU = np.zeros((3,self.Nx))
        for i in range(3):
            LU[i][1:-1] = -(FHLL[i][1:] - FHLL[i][:-1])/self.dx

        return LU
       
   
if __name__=='__main__':
    
    b = Brio_wu(-1,1,2000,0.0)
    b.setInitcondion()
    
    def animate(t):
        fig = plt.figure()
        frames = []
        
       # plt.ion()
        
        for ti in np.arange(0,t+0.01,0.01):
            plt.cla()
            plt.title("t = {0}s".format(ti))
            
            b.evolve(ti)
            x = b.x
              #can be changed into vx,vy,vz,By,Bz,p
            
            plt.xlabel('position')
       #     plt.ylabel('p_euler')
            
            plt.plot(x,b.rho,label='rho')
            plt.plot(x,b.v,label='v')
            plt.plot(x,b.p,label='p')
            plt.legend()
            
            fig.savefig('all_euler.png')
            frames.append(imageio.imread('all_euler.png'))

        imageio.mimsave('all_euler_evolve.gif',frames,'GIF',duration=0.2)
        return
    
    animate(0.2)



            
