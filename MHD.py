#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:26:18 2018

@author: zhujiarong
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio


class Brio_Wu:
    
    def __init__(self,a,b,Nx,Bx,t):
        
        self.a = a
        self.b = b
        self.Nx = int(Nx)
        self.t = t
        
        self.dx = (b - a)/float(Nx)
        self.x = a + self.dx*( np.arange(Nx) + 0.5 )
        self.u = np.zeros((7,Nx))
        self.rho = np.zeros(Nx)
        self.vx = np.zeros(Nx)
        self.vy = np.zeros(Nx)
        self.vz = np.zeros(Nx)
        self.By = np.zeros(Nx)
        self.Bz = np.zeros(Nx)
        self.p = np.zeros(Nx)
        
        self.cfl = 0.475
        self.gamma = 2.0
        self.Bx = Bx  
        
        self.cs = np.zeros(Nx)
        self.lp = np.zeros(Nx)
        self.lm = np.zeros(Nx)
        
    #set up initial conditions 
    def setInitcondion(self):
        self.rho[:int(self.Nx/2)]=0.5
        self.rho[int(self.Nx/2):]=0.1
        self.vx[:int(self.Nx/2)]=0.0
        self.vx[int(self.Nx/2):]=0.0
        self.vy[:int(self.Nx/2)]=1.0
        self.vy[int(self.Nx/2):]=0.0
        #self.vy[int(self.Nx/3):int(self.Nx*2/3)]=1
        self.vz[:int(self.Nx/2)]=0.0
        self.vz[int(self.Nx/2):]=0.0
        self.By[:int(self.Nx/2)]=0.0
        self.By[int(self.Nx/2):]=0.0
        self.Bz[:int(self.Nx/2)]=0.0
        self.Bz[int(self.Nx/2):]=0.0
        self.p[:int(self.Nx/2)]=1.0
        self.p[int(self.Nx/2):]=0.2


        self.u[0] = self.rho
        self.u[1] = self.rho * self.vx  
        self.u[2] = self.rho * self.vy
        self.u[3] = self.rho * self.vz
        self.u[4] = self.By
        self.u[5] = self.Bz
        self.u[6] = 0.5 * self.rho * (self.vx **2 + self.vy **2 + self.vz **2)\
        + self.p / (self.gamma - 1) + (self.Bx **2 + self.By **2 + self.Bz **2)/2 
        
        temp = self.gamma * self.p + self.Bx **2 + self.By **2 + self.Bz **2
        self.cf = np.sqrt( (temp + np.sqrt(temp **2 - 4 * self.gamma * self.p * self.Bx**2)) / (2 * self.rho))

        self.lp = self.vx + self.cf
        self.lm = self.vx - self.cf
        
    
    def get_dt(self):
        return self.cfl * self.dx / max(np.max( self.lp),np.max(-self.lm ))
       
    def evolve(self,tfinal):
        
        while self.t < tfinal:
            
            dt = self.get_dt()
            if self.t+dt > tfinal:
                dt = tfinal - self.t
                
            LU = self.LU()
            
            for i in range(7):
                self.u[i] += LU[i]*dt

            self.t += dt
            
            self.rho = self.u[0]
            self.vx = self.u[1]/self.u[0]
            self.vy = self.u[2]/self.u[0]
            self.vz = self.u[3]/self.u[0]
            self.By = self.u[4]
            self.Bz = self.u[5]
            self.p = (self.gamma-1)*(self.u[6] - 1/2*(self.u[1]**2 + self.u[2]**2 \
                     + self.u[3]**2)/self.u[0] - 1/2 * (self.Bx**2 \
                             +self.u[4]**2 + self.u[5]**2))
            
            temp = self.gamma * self.p + self.Bx **2 + self.By **2 + self.Bz **2
            self.cf = np.sqrt( (temp + np.sqrt(np.abs(temp **2 - 4 * self.gamma * self.p * self.Bx**2))) / (2 * self.rho))

            self.lp = self.vx + self.cf
            self.lm = self.vx - self.cf

    def LU(self):        
        ap = np.zeros(self.Nx-1)
        am = np.zeros(self.Nx-1)
        
        for i in range(self.Nx-1):
            ap[i] = max(0,self.lp[i],self.lp[i+1])
            am[i] = max(0,-self.lm[i],-self.lm[i+1])
        
        F = np.zeros((7,self.Nx))
        F[0] = self.rho*self.vx
        F[1] = self.rho*self.vx**2 + self.p + 1/2 * (self.By**2 + self.Bz**2)
        F[2] = self.rho * self.vx * self.vy - self.Bx * self.By
        F[3] = self.rho * self.vx * self.vz - self.Bx * self.Bz
        F[4] = self.By * self.vx - self.Bx * self.vy
        F[5] = self.Bz * self.vx - self.Bx * self.vz
        F[6] = self.vx * (0.5 * self.rho * (self.vx **2 + self.vy **2 + self.vz **2)\
        + self.p * self.gamma / (self.gamma - 1) + (self.Bx **2 + self.By **2 + self.Bz **2))  \
        - self.Bx * (self.vx * self.Bx + self.vy * self.By + self.vz * self.Bz)
         

        FL = np.zeros((7,self.Nx-1))
        FR = np.zeros((7,self.Nx-1))
        UL = np.zeros((7,self.Nx-1))
        UR = np.zeros((7,self.Nx-1))
        FHLL = np.zeros((7,self.Nx-1))
        
        for i in range(7):
            FL[i] = F[i][:-1]
            FR[i] = F[i][1:]        
            UL[i] = self.u[i][:-1]
            UR[i] = self.u[i][1:]
            FHLL[i] = (ap*FL[i] + am*FR[i] - ap*am*(UR[i]-UL[i]))/(ap+am)
    
        LU = np.zeros((7,self.Nx))
        for i in range(7):
            LU[i][1:-1] = -(FHLL[i][1:] - FHLL[i][:-1])/self.dx

        return LU
       
   
if __name__=='__main__':
    
    b = Brio_Wu(-1,1,2000,3.0,0) #Brio_Wu(a,b,Nx,Bx,t)
    b.setInitcondion()
        
    
    def animate(t):
        fig = plt.figure()
        frames = []
        
       # plt.ion()
        
        for ti in np.arange(0,t+0.01,0.01):
            plt.cla()
            plt.title("t = {0}s".format(ti))
            
            b.evolve(ti)
          #  x = b.x
          #  y = b.vy  #can be changed into vx,vy,vz,By,Bz,p
            
            plt.xlabel('x')
           # plt.ylabel('vy')
            
            plt.plot(b.x,b.rho,label='rho')
            plt.plot(b.x,b.vx,label='vx')
            plt.plot(b.x,b.vy,label='vy')
            plt.plot(b.x,b.vz,label='vz')
            plt.plot(b.x,b.By,label='By')
            plt.plot(b.x,b.Bz,label='Bz')
            plt.plot(b.x,b.p,label='p')
            plt.legend()
            
            
            
            fig.savefig('5.png')
            frames.append(imageio.imread('5.png'))

        imageio.mimsave('5.gif',frames,'GIF',duration=0.1)
        return
    
    animate(0.1)
            
    
            
   
            
