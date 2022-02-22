'''
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Shape Optimization
First name: ***Andreas***
Last name: ***Zafiropoulos***
Student number: ***4474538***

This module defines useful functions that are used to process the output data.
'''


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

# This option runs the whole propagation again
#from CapsuleEntryPropagation import variable_step_size_integrator_index
#from CapsuleEntryPropagation import fixed_step_size_integrator_index
# Instead you can also manually copy the values
variable_step_size_integrator_index = 4
fixed_step_size_integrator_index = 6

###########################################################################
# ACTIVATE SECTIONS #######################################################
###########################################################################

benchmark = True
shape = False
q2b = False

###########################################################################
# BENCHMARK PLOTS #########################################################
###########################################################################

if benchmark == True:
    # Import the benchmark states

    benchmark_1_states = np.loadtxt('SimulationOutput/benchmarks/benchmark_1_states.dat')
    benchmark_2_states = np.loadtxt('SimulationOutput/benchmarks/benchmark_2_states.dat')
    benchmark_states_difference_0_1 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.1_s.dat')
    benchmark_states_difference_0_2 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.2_s.dat')
    benchmark_states_difference_0_3 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.3_s.dat')
    benchmark_states_difference_0_4 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.4_s.dat')
    benchmark_states_difference_0_5 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.5_s.dat')
    benchmark_states_difference_0_6 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.6_s.dat')
    benchmark_states_difference_0_7 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.7_s.dat')
    benchmark_states_difference_0_8 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.8_s.dat')
    benchmark_states_difference_0_9 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_0.9_s.dat')
    benchmark_states_difference_1 = np.loadtxt('SimulationOutput/benchmarks/benchmarks_state_difference_time_step_1.0_s.dat')
    benchmark_states_difference_1_1 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_1.1_s.dat')
    benchmark_states_difference_1_2 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_1.2_s.dat')
    benchmark_states_difference_1_3 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_1.3_s.dat')
    benchmark_states_difference_1_4 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_1.4_s.dat')
    benchmark_states_difference_1_5 = np.loadtxt(
        'SimulationOutput/benchmarks/benchmarks_state_difference_time_step_1.5_s.dat')
    benchmark_states_difference_2 = np.loadtxt('SimulationOutput/benchmarks/benchmarks_state_difference_time_step_2.0_s.dat')
    benchmark_states_difference_4 = np.loadtxt('SimulationOutput/benchmarks/benchmarks_state_difference_time_step_4.0_s.dat')
    benchmark_states_difference_8 = np.loadtxt('SimulationOutput/benchmarks/benchmarks_state_difference_time_step_8.0_s.dat')
    benchmark_states_difference_12 = np.loadtxt('SimulationOutput/benchmarks/benchmarks_state_difference_time_step_12.0_s.dat')
    benchmark_1_times = benchmark_1_states[:,0]
    benchmark_2_times = benchmark_2_states[:,0]
    benchmark_difference_times_0_1 = benchmark_states_difference_0_1[:, 0]
    benchmark_difference_times_0_2 = benchmark_states_difference_0_2[:, 0]
    benchmark_difference_times_0_3 = benchmark_states_difference_0_3[:, 0]
    benchmark_difference_times_0_4 = benchmark_states_difference_0_4[:, 0]
    benchmark_difference_times_0_5 = benchmark_states_difference_0_5[:, 0]
    benchmark_difference_times_0_6 = benchmark_states_difference_0_6[:, 0]
    benchmark_difference_times_0_7 = benchmark_states_difference_0_7[:, 0]
    benchmark_difference_times_0_8 = benchmark_states_difference_0_8[:, 0]
    benchmark_difference_times_0_9 = benchmark_states_difference_0_9[:, 0]
    benchmark_difference_times_1 = benchmark_states_difference_1[:,0]
    benchmark_difference_times_1_1 = benchmark_states_difference_1_1[:, 0]
    benchmark_difference_times_1_2 = benchmark_states_difference_1_2[:, 0]
    benchmark_difference_times_1_3 = benchmark_states_difference_1_3[:, 0]
    benchmark_difference_times_1_4 = benchmark_states_difference_1_4[:, 0]
    benchmark_difference_times_1_5 = benchmark_states_difference_1_5[:, 0]
    benchmark_difference_times_2 = benchmark_states_difference_2[:, 0]
    benchmark_difference_times_4 = benchmark_states_difference_4[:, 0]
    benchmark_difference_times_8 = benchmark_states_difference_8[:, 0]
    benchmark_difference_times_12 = benchmark_states_difference_12[:, 0]
    benchmark_1_states = benchmark_1_states[:,1:4]
    benchmark_2_states = benchmark_2_states[:,1:4]
    benchmark_pos_difference_0_1 = benchmark_states_difference_0_1[:, 1:4]
    benchmark_pos_difference_0_2 = benchmark_states_difference_0_2[:, 1:4]
    benchmark_pos_difference_0_3 = benchmark_states_difference_0_3[:, 1:4]
    benchmark_pos_difference_0_4 = benchmark_states_difference_0_4[:, 1:4]
    benchmark_pos_difference_0_5 = benchmark_states_difference_0_5[:, 1:4]
    benchmark_pos_difference_0_6 = benchmark_states_difference_0_6[:, 1:4]
    benchmark_pos_difference_0_7 = benchmark_states_difference_0_7[:, 1:4]
    benchmark_pos_difference_0_8 = benchmark_states_difference_0_8[:, 1:4]
    benchmark_pos_difference_0_9 = benchmark_states_difference_0_9[:, 1:4]
    benchmark_pos_difference_1 = benchmark_states_difference_1[:,1:4]
    benchmark_pos_difference_1_1 = benchmark_states_difference_1_1[:, 1:4]
    benchmark_pos_difference_1_2 = benchmark_states_difference_1_2[:, 1:4]
    benchmark_pos_difference_1_3 = benchmark_states_difference_1_3[:, 1:4]
    benchmark_pos_difference_1_4 = benchmark_states_difference_1_4[:, 1:4]
    benchmark_pos_difference_1_5 = benchmark_states_difference_1_5[:, 1:4]
    benchmark_pos_difference_2 = benchmark_states_difference_2[:, 1:4]
    benchmark_pos_difference_4 = benchmark_states_difference_4[:, 1:4]
    benchmark_pos_difference_8 = benchmark_states_difference_8[:, 1:4]
    benchmark_pos_difference_12 = benchmark_states_difference_12[:, 1:4]

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

    benchmark_pos_error_1 = []
    benchmark_pos_error_0_1 = []
    benchmark_pos_error_0_2 = []
    benchmark_pos_error_0_3 = []
    benchmark_pos_error_0_4 = []
    benchmark_pos_error_0_5 = []
    benchmark_pos_error_0_6 = []
    benchmark_pos_error_0_7 = []
    benchmark_pos_error_0_8 = []
    benchmark_pos_error_0_9 = []
    benchmark_pos_error_1_1 = []
    benchmark_pos_error_1_2 = []
    benchmark_pos_error_1_3 = []
    benchmark_pos_error_1_4 = []
    benchmark_pos_error_1_5 = []
    benchmark_pos_error_2 = []
    benchmark_pos_error_4 = []
    benchmark_pos_error_8 = []
    benchmark_pos_error_12 = []

    for i in range(benchmark_pos_difference_0_1.shape[0]):
        benchmark_pos_error_0_1.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_1[i, :]))
    for i in range(benchmark_pos_difference_0_2.shape[0]):
        benchmark_pos_error_0_2.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_2[i, :]))
    for i in range(benchmark_pos_difference_0_3.shape[0]):
        benchmark_pos_error_0_3.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_3[i, :]))
    for i in range(benchmark_pos_difference_0_4.shape[0]):
        benchmark_pos_error_0_4.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_4[i, :]))
    for i in range(benchmark_pos_difference_0_5.shape[0]):
        benchmark_pos_error_0_5.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_5[i, :]))
    for i in range(benchmark_pos_difference_0_6.shape[0]):
        benchmark_pos_error_0_6.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_6[i, :]))
    for i in range(benchmark_pos_difference_0_7.shape[0]):
        benchmark_pos_error_0_7.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_7[i, :]))
    for i in range(benchmark_pos_difference_0_8.shape[0]):
        benchmark_pos_error_0_8.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_8[i, :]))
    for i in range(benchmark_pos_difference_0_9.shape[0]):
        benchmark_pos_error_0_9.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_0_9[i, :]))
    for i in range(benchmark_pos_difference_1.shape[0]):
        benchmark_pos_error_1.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_1[i,:]))
    for i in range(benchmark_pos_difference_1_1.shape[0]):
        benchmark_pos_error_1_1.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_1_1[i, :]))
    for i in range(benchmark_pos_difference_1_2.shape[0]):
        benchmark_pos_error_1_2.append(
            Util.get_absolute_distance_from_origin(benchmark_pos_difference_1_2[i, :]))
    for i in range(benchmark_pos_difference_1_3.shape[0]):
        benchmark_pos_error_1_3.append(
            Util.get_absolute_distance_from_origin(benchmark_pos_difference_1_3[i, :]))
    for i in range(benchmark_pos_difference_1_4.shape[0]):
        benchmark_pos_error_1_4.append(
            Util.get_absolute_distance_from_origin(benchmark_pos_difference_1_4[i, :]))
    for i in range(benchmark_pos_difference_1_5.shape[0]):
        benchmark_pos_error_1_5.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_1_5[i,:]))
    for i in range(benchmark_pos_difference_2.shape[0]):
        benchmark_pos_error_2.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_2[i, :]))
    for i in range(benchmark_pos_difference_4.shape[0]):
        benchmark_pos_error_4.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_4[i, :]))
    for i in range(benchmark_pos_difference_8.shape[0]):
        benchmark_pos_error_8.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_8[i, :]))
    for i in range(benchmark_pos_difference_12.shape[0]):
        benchmark_pos_error_12.append(Util.get_absolute_distance_from_origin(benchmark_pos_difference_12[i, :]))

    # Actual plots

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    #ax1.plot(xs=benchmark_1_states[:, 0], ys=benchmark_1_states[:, 1], zs=benchmark_1_states[:, 2])
    ax1.plot(xs=benchmark_2_states[:, 0], ys=benchmark_2_states[:, 1], zs=benchmark_2_states[:, 2])
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.legend(['Benchmark $\Delta$t = 0.5s re-entry trajectory'])
    ax1.set_title('Re-entry trajectory')
    #ax1.legend(['Benchmark 1 re-entry trajectory','Benchmark 2 re-entry trajectory'])

    fig2 = plt.figure()
    #plt.plot(altitude_benchmark_1[:,0],altitude_benchmark_1[:,1],label = ('Benchmark 1'))
    plt.plot(altitude_benchmark_2[:,0],altitude_benchmark_2[:,1],label = ('$\Delta$t = 0.5s'))
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Altitude (m)')
    plt.grid()
    plt.title('Altitude vs time')
    plt.legend()

    fig3 = plt.figure()
    #plt.plot(mach_benchmark_1[:,0],mach_benchmark_1[:,1],label = ('Benchmark 1'))
    plt.plot(mach_benchmark_2[:,0],mach_benchmark_2[:,1],label = ('$\Delta$t = 0.5s'))
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Mach number (-)')
    plt.grid()
    plt.title('Mach number vs time')
    plt.legend()

    fig4 = plt.figure()
    #plt.plot(mach_benchmark_1[:,1],altitude_benchmark_1[:,1],label = ('Benchmark 1'))
    plt.plot(mach_benchmark_2[:,1],altitude_benchmark_2[:,1],label = ('$\Delta$t = 0.5s'))
    plt.xlabel('Mach number (-)')
    plt.ylabel('Altitude (m)')
    plt.grid()
    plt.title('Altitude vs Mach number')
    plt.legend()

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(1,1,1)
    ax5.plot(benchmark_difference_times_0_1, benchmark_pos_error_0_1, label=('$\Delta$t = 0.1s'))
    ax5.plot(benchmark_difference_times_0_2, benchmark_pos_error_0_2, label=('$\Delta$t = 0.2s'))
    ax5.plot(benchmark_difference_times_0_3, benchmark_pos_error_0_3, label=('$\Delta$t = 0.3s'))
    ax5.plot(benchmark_difference_times_0_4, benchmark_pos_error_0_4, label=('$\Delta$t = 0.4s'))
    ax5.plot(benchmark_difference_times_0_5, benchmark_pos_error_0_5, label=('$\Delta$t = 0.5s'))
    ax5.plot(benchmark_difference_times_0_6, benchmark_pos_error_0_6, label=('$\Delta$t = 0.6s'))
    ax5.plot(benchmark_difference_times_0_7, benchmark_pos_error_0_7, label=('$\Delta$t = 0.7s'))
    #ax5.plot(benchmark_difference_times_0_8, benchmark_pos_error_0_8, label=('$\Delta$t = 0.8s'))
    #ax5.plot(benchmark_difference_times_0_9, benchmark_pos_error_0_9, label=('$\Delta$t = 0.9s'))
    #ax5.plot(benchmark_difference_times_1, benchmark_pos_error_1, label=('$\Delta$t = 1.0s'))
    #ax5.plot(benchmark_difference_times_1_1, benchmark_pos_error_1_1, label=('$\Delta$t = 1.1s'))
    #ax5.plot(benchmark_difference_times_1_2, benchmark_pos_error_1_2, label=('$\Delta$t = 1.2s'))
    #ax5.plot(benchmark_difference_times_1_3, benchmark_pos_error_1_3, label=('$\Delta$t = 1.3s'))
    #ax5.plot(benchmark_difference_times_1_4, benchmark_pos_error_1_4, label=('$\Delta$t = 1.4s'))
    #ax5.plot(benchmark_difference_times_1_5, benchmark_pos_error_1_5, label=('$\Delta$t = 1.5s'))
    #ax5.plot(benchmark_difference_times_2, benchmark_pos_error_2, label=('$\Delta$t = 2.0s'))
    #ax5.plot(benchmark_difference_times_4, benchmark_pos_error_4, label=('$\Delta$t = 4.0s'))
    #ax5.plot(benchmark_difference_times_8, benchmark_pos_error_8, label=('$\Delta$t = 8.0s'))
    #ax5.plot(benchmark_difference_times_12, benchmark_pos_error_12, label=('$\Delta$t = 12.0s'))
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position error (m)')
    #ax5.set_yticks()
    ax5.grid()
    ax5.set_title('Position error of the benchmarks as a function of time')
    ax5.set_yscale('log')
    ax5.set_ylim([0.000001,0.1])
    ax5.set_xlim([1000, 1408])
    ax5.legend()

elif shape == True:
    shapefile = np.loadtxt('SimulationOutput/prop_0/int_0/step_size_0/ShapeFile.dat')
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(projection='3d')
    ax6.scatter(shapefile[:,1],shapefile[:,2],shapefile[:,3])

elif q2b == True:
    RK45_dict = {}
    RK45_keys = [10**y for y in [-10 + x for x in range(variable_step_size_integrator_index)]]
    print("RK45_keys = ",RK45_keys)
    for key in RK45_keys:
        i = RK45_keys.index(key)
        print("key = ",key)
        state_difference = np.loadtxt('SimulationOutput/prop_0/int_0/step_size_' + str(i) + '/state_difference_wrt_benchmark.dat')
        #print("state_difference = ",state_difference)
        print("shape = ",state_difference.shape)
        RK45_value = state_difference[:,0]
        print("type(RK45_value) = ",type(RK45_value))
        pos_error = []
        for j in range(state_difference.shape[0]):
            pos_error.append(Util.get_absolute_distance_from_origin(state_difference[j,1:4]))
        #RK45_value = np.reshape(RK45_value,(RK45_value.shape[0],1))
        print("shape of times = ",np.reshape(RK45_value,(RK45_value.shape[0],1)).shape)
        print("shape of error = ",np.array(pos_error).shape)
        #RK45_dict[key] = np.concatenate((np.reshape(RK45_value,(RK45_value.shape[0],1)),pos_error),axis=1)
        #print(RK45_dict[key])
#print("RK45_dict = ",RK45_dict)


plt.show()


