#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 21:08:40 2018

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
        self.rho[:int(self.Nx/2)]=1.0
        self.rho[int(self.Nx/2):]=0.125
        self.vx[:int(self.Nx/2)]=0.0
        self.vx[int(self.Nx/2):]=0.0
        self.vy[:int(self.Nx/2)]=0.0
        self.vy[int(self.Nx/2):]=0.0
     #   self.vy[int(self.Nx/4):int(self.Nx*3/4)]=0.001
        self.vz[:int(self.Nx/2)]=0.0
        self.vz[int(self.Nx/2):]=0.0
        self.By[:int(self.Nx/2)]=1.0
        self.By[int(self.Nx/2):]=-1.0
        self.Bz[:int(self.Nx/2)]=0.0
        self.Bz[int(self.Nx/2):]=0.0
        self.p[:int(self.Nx/2)]=1.0
        self.p[int(self.Nx/2):]=0.1


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
            
            LU = np.zeros((3,7,self.Nx))
            LU[0] = self.LU(self.u)
            
            self.u1 = self.u + LU[0]*dt
            LU[1] = self.LU(self.u1)
            self.u2 = 3/4*self.u + 1/4*self.u1 + 1/4*dt*LU[1]
            LU[2] = self.LU(self.u2)
            self.u = 1/3*self.u + 2/3*self.u2 + 2/3*dt*LU[2]

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
            self.cf = np.sqrt( (temp + np.sqrt(temp **2 - 4 * self.gamma * self.p * self.Bx**2)) / (2 * self.rho))

            self.lp = self.vx + self.cf
            self.lm = self.vx - self.cf
            
    def LU(self,U): 
        
        rhoL = np.zeros(self.Nx-1)
        rhoR = np.zeros(self.Nx-1)
        vxL = np.zeros(self.Nx-1)
        vxR = np.zeros(self.Nx-1)
        vyL = np.zeros(self.Nx-1)
        vyR = np.zeros(self.Nx-1)
        vzL = np.zeros(self.Nx-1)
        vzR = np.zeros(self.Nx-1)
        ByL = np.zeros(self.Nx-1)
        ByR = np.zeros(self.Nx-1)
        BzL = np.zeros(self.Nx-1)
        BzR = np.zeros(self.Nx-1)        
        pL = np.zeros(self.Nx-1)
        pR = np.zeros(self.Nx-1)
        
        theta = 1.0

        
        rho= U[0]
        rhoL[0] = rho[0]
        rhoL[1:]=rho[1:-1] + 0.5*self.minmod(theta*(rho[1:-1]-rho[0:-2]),0.5*(rho[2:]-rho[0:-2]),theta*(rho[2:]-rho[1:-1]))
        rhoR[-1] = rho[-1]
        rhoR[:-1]=rho[1:-1] - 0.5*self.minmod(theta*(rho[1:-1]-rho[0:-2]),0.5*(rho[2:]-rho[0:-2]),theta*(rho[2:]-rho[1:-1]))        

        vx = U[1]/U[0]
        vxL[0] = vx[0]
        vxL[1:] = vx[1:-1] + 0.5*self.minmod(theta*(vx[1:-1]-vx[0:-2]),0.5*(vx[2:]-vx[0:-2]),theta*(vx[2:]-vx[1:-1]))
        vxR[-1] =vx[-1]
        vxR[:-1] = vx[1:-1] - 0.5*self.minmod(theta*(vx[1:-1]-vx[0:-2]),0.5*(vx[2:]-vx[0:-2]),theta*(vx[2:]-vx[1:-1]))             

        vy = U[2]/U[0]
        vyL[0] = vy[0]
        vyL[1:] = vy[1:-1] + 0.5*self.minmod(theta*(vy[1:-1]-vy[0:-2]),0.5*(vy[2:]-vy[0:-2]),theta*(vy[2:]-vy[1:-1]))
        vyR[-1] =vy[-1]
        vyR[:-1] = vy[1:-1] - 0.5*self.minmod(theta*(vy[1:-1]-vy[0:-2]),0.5*(vy[2:]-vy[0:-2]),theta*(vy[2:]-vy[1:-1]))             

        vz = U[3]/U[0]
        vzL[0] = vz[0]
        vzL[1:] = vz[1:-1] + 0.5*self.minmod(theta*(vz[1:-1]-vz[0:-2]),0.5*(vz[2:]-vz[0:-2]),theta*(vz[2:]-vz[1:-1]))
        vzR[-1] = vz[-1]
        vzR[:-1] = vz[1:-1] - 0.5*self.minmod(theta*(vz[1:-1]-vz[0:-2]),0.5*(vz[2:]-vz[0:-2]),theta*(vz[2:]-vz[1:-1]))             

        By = U[4]
        ByL[0] = By[0]
        ByL[1:] = By[1:-1] + 0.5*self.minmod(theta*(By[1:-1]-By[0:-2]),0.5*(By[2:]-By[0:-2]),theta*(By[2:]-By[1:-1]))
        ByR[-1] =By[-1]
        ByR[:-1] = By[1:-1] - 0.5*self.minmod(theta*(By[1:-1]-By[0:-2]),0.5*(By[2:]-By[0:-2]),theta*(By[2:]-By[1:-1]))             

        Bz = U[5]
        BzL[0] = Bz[0]
        BzL[1:] = Bz[1:-1] + 0.5*self.minmod(theta*(Bz[1:-1]-Bz[0:-2]),0.5*(Bz[2:]-Bz[0:-2]),theta*(Bz[2:]-Bz[1:-1]))
        BzR[-1] = Bz[-1]
        BzR[:-1] = Bz[1:-1] - 0.5*self.minmod(theta*(Bz[1:-1]-Bz[0:-2]),0.5*(Bz[2:]-Bz[0:-2]),theta*(Bz[2:]-Bz[1:-1]))             

        p = (self.gamma-1)*(U[6]- 1/2*(U[1]**2 + U[2]**2 \
                     + U[3]**2)/U[0] - 1/2 * (self.Bx**2 \
                             +U[4]**2 + U[5]**2))
        pL[0] = p[0]
        pL[1:] = p[1:-1] + 0.5*self.minmod(theta*(p[1:-1]-p[0:-2]),0.5*(p[2:]-p[0:-2]),theta*(p[2:]-p[1:-1]))
        pR[-1] = p[-1]
        pR[:-1]=p[1:-1] - 0.5*self.minmod(theta*(p[1:-1]-p[0:-2]),0.5*(p[2:]-p[0:-2]),theta*(p[2:]-p[1:-1]))             

        tempL = self.gamma * pL + self.Bx **2 + ByL **2 + BzL **2
        tempR = self.gamma * pR + self.Bx **2 + ByR **2 + BzR **2
        
        cfL = np.sqrt((tempL + np.sqrt(tempL**2 - 4 * self.gamma * pL * self.Bx**2))/(2*rhoL))
        cfR = np.sqrt((tempR + np.sqrt(tempR**2 - 4 * self.gamma * pR * self.Bx**2))/(2*rhoR))

        lpL = vxL + cfL
        lpR = vxR + cfR
        lmL = vxL - cfL
        lmR = vxR - cfR
                
        ap = np.maximum(np.maximum(lpL,lpR),np.zeros(self.Nx-1))
        am = np.maximum(np.maximum(-lmL,-lmR),np.zeros(self.Nx-1))

        FL = np.zeros((7,self.Nx-1))
        FR = np.zeros((7,self.Nx-1))
        FL[0],FR[0] = rhoL*vxL,rhoR*vxR
        FL[1],FR[1] = rhoL*vxL**2 + pL + 1/2 * (ByL**2 + BzL*2),rhoR*vxR**2 + pR + 1/2 * (ByR**2 + BzR*2)
        FL[2],FR[2] = rhoL * vxL * vyL - self.Bx * ByL, rhoR * vxR * vyR - self.Bx * ByR
        FL[3],FR[3] = rhoL * vxL * vzL - self.Bx * BzL, rhoR * vxR * vzR - self.Bx * BzR
        FL[4],FR[4] = ByL * vxL - self.Bx * vyL, ByR * vxR - self.Bx * vyR
        FL[5],FR[5] = BzL * vxL - self.Bx * vzL, BzR * vxR - self.Bx * vzR
        FL[6],FR[6] = vxL * (1/2 * rhoL *(vxL**2 + vyL **2 + vzL**2) + self.gamma/(self.gamma - 1) * pL \
          + (self.Bx**2 + ByL**2 + BzL**2)) - self.Bx * (vxL * self.Bx + vyL * ByL + vzL * BzL),\
          vxR * (1/2 * rhoR *(vxR**2 + vyR **2 + vzR**2) + self.gamma/(self.gamma - 1) * pR \
          +  (self.Bx**2 + ByR**2 + BzR**2)) - self.Bx * (vxR * self.Bx + vyR * ByR + vzR * BzR)
        
                   
        FHLL = np.zeros((7,self.Nx-1))
        UL = np.zeros((7,self.Nx-1))
        UR = np.zeros((7,self.Nx-1))
        
        UL[0],UR[0] = rhoL,rhoR
        UL[1],UR[1] = rhoL * vxL, rhoR * vxR
        UL[2],UR[2] = rhoL * vyL, rhoR * vyR
        UL[3],UR[3] = rhoL * vzL, rhoR * vzR
        UL[4],UR[4] = ByL, ByR
        UL[5],UR[5] = BzL, BzR
        UL[6],UR[6] = 1/2 * rhoL * (vxL**2 + vyL **2 + vzL**2) + pL/(self.gamma - 1) + 1/2 * (self.Bx**2 + ByL**2 + BzL**2),\
        1/2 * rhoR * (vxR**2 + vyR **2 + vzR**2) + pR/(self.gamma - 1) + 1/2 * (self.Bx**2 + ByR**2 + BzR**2)
        
        for i in range(7):
            FHLL[i] = (ap*FL[i] + am*FR[i] - ap*am*(UR[i]-UL[i]))/(ap+am)
            
        LU = np.zeros((7,self.Nx))
        for i in range(7):
            LU[i][1:-1] = -(FHLL[i][1:] - FHLL[i][:-1])/self.dx
        
        return LU

    def minmod(self,x,y,z):
        return 1/4*np.fabs((np.sign(x)+np.sign(y)))*(np.sign(x)+np.sign(z))*np.minimum(np.fabs(z),np.minimum(np.fabs(x),np.fabs(y)))
     
   
if __name__=='__main__':
    
    b = Brio_Wu(-1,1,500,0.75,0)
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
            
            
            
            fig.savefig('Brio_high.png')
            frames.append(imageio.imread('Brio_high.png'))

        imageio.mimsave('Brio_high_evolve.gif',frames,'GIF',duration=0.2)
        return
    
    animate(0.2)
            
  


    
    
