###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
import numpy as np
import matplotlib.pyplot as plt

# Problem-specific imports
import CapsuleEntryUtilities as Util

###########################################################################
# ACTIVATE SECTIONS #######################################################
###########################################################################

benchmark = True

###########################################################################
# BENCHMARK PLOTS #########################################################
###########################################################################

if benchmark == True:
    # Import the benchmark states

    benchmark_1_states = np.loadtxt('SimulationOutput/benchmarks/benchmark_1_states.dat')
    benchmark_2_states = np.loadtxt('SimulationOutput/benchmarks/benchmark_2_states.dat')
    benchmark
    benchmark_1_times = benchmark_1_states[:,0]
    benchmark_2_times = benchmark_2_states[:,0]
    benchmark_1_states = benchmark_1_states[:,1:4]
    benchmark_2_states = benchmark_2_states[:,1:4]
    '''
    # Transform states to spherical coordinates [r,theta,phi]
    
    states_size_1 = benchmark_1_states.shape
    states_size_2 = benchmark_2_states.shape
    print("shape of the benchmark 1 states = ",states_size_1)
    print("shape of the benchmark 2 states = ",states_size_2)
    
    spherical_benchmark_states_1 = np.zeros(states_size_1)
    spherical_benchmark_states_2 = np.zeros(states_size_2)
    
    for i in range(spherical_benchmark_states_1.shape[0]):
        r = Util.get_absolute_distance_from_origin(benchmark_1_states[i,:])
        theta = np.arccos(benchmark_1_states[i,2]/r)
        phi = np.arctan(benchmark_1_states[i,0]/benchmark_1_states[i,1])
        spherical_benchmark_states_1[i] = [r,theta,phi]
    for j in range(spherical_benchmark_states_2.shape[0]):
        r = Util.get_absolute_distance_from_origin(benchmark_2_states[j,:])
        theta = np.arccos(benchmark_2_states[j,2]/r)
        phi = np.arctan(benchmark_2_states[j,0]/benchmark_2_states[j,1])
        spherical_benchmark_states_2[j] = [r,theta,phi]
    
    altitude_benchmark_1 = spherical_benchmark_states_1[:,0] - 6378136
    altitude_benchmark_2 = spherical_benchmark_states_2[:,0] - 6378136
    '''
    dep_vars_benchmark_1 = np.loadtxt('SimulationOutput/benchmarks/benchmark_1_dependent_variables.dat')
    dep_vars_benchmark_2 = np.loadtxt('SimulationOutput/benchmarks/benchmark_2_dependent_variables.dat')

    altitude_benchmark_1 = np.delete(dep_vars_benchmark_1, 1, axis=1)
    altitude_benchmark_2 = np.delete(dep_vars_benchmark_2, 1, axis=1)

    mach_benchmark_1 = np.delete(dep_vars_benchmark_1, 2, axis=1)
    mach_benchmark_2 = np.delete(dep_vars_benchmark_2, 2, axis=1)
    # Actual plots

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.plot(xs=benchmark_1_states[:, 0], ys=benchmark_1_states[:, 1], zs=benchmark_1_states[:, 2])
    #ax1.plot(xs=benchmark_2_states[:, 0], ys=benchmark_2_states[:, 1], zs=benchmark_2_states[:, 2])
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.legend(['Benchmark 1 re-entry trajectory'])
    ax1.set_title('Re-entry trajectory')
    #ax1.legend(['Benchmark 1 re-entry trajectory','Benchmark 2 re-entry trajectory'])

    fig2 = plt.figure()
    plt.plot(altitude_benchmark_1[:,0],altitude_benchmark_1[:,1],label = ('Benchmark 1'))
    plt.plot(altitude_benchmark_2[:,0],altitude_benchmark_2[:,1],label = ('Benchmark 2'))
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Altitude (m)')
    plt.grid()
    plt.title('Altitude vs time')
    plt.legend()

    fig3 = plt.figure()
    plt.plot(mach_benchmark_1[:,0],mach_benchmark_1[:,1],label = ('Benchmark 1'))
    plt.plot(mach_benchmark_2[:,0],mach_benchmark_2[:,1],label = ('Benchmark 2'))
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Mach number (-)')
    plt.grid()
    plt.title('Mach number vs time')
    plt.legend()

    fig4 = plt.figure()
    plt.plot(mach_benchmark_1[:,1],altitude_benchmark_1[:,1],label = ('Benchmark 1'))
    plt.plot(mach_benchmark_2[:,1],altitude_benchmark_2[:,1],label = ('Benchmark 2'))
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Mach number (-)')
    plt.grid()
    plt.title('Altitude vs Mach number')
    plt.legend()

    plt.show()


