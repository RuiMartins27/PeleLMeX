# AMReX
DIM = 3
COMP = gnu
PRECISION = DOUBLE
USE_EB = TRUE
USE_HYPRE = TRUE

# Profiling
PROFILE = FALSE
TINY_PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
USE_GPROF = FALSE

# Performance
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE
USE_SYCL = FALSE

# Debugging
DEBUG = FALSE
FSANITIZER = FALSE
THREAD_SANITIZER = FALSE

# PeleLMeX
USE_EFIELD = FALSE
USE_MWSOLVER = TRUE

# PeleLMeX
#CEXE_headers += EBUserDefined.H

CXXFLAGS += -O3 -march=znver4 -mtune=znver4
CFLAGS   += -O3 -march=znver4 -mtune=znver4

# PelePhysics
Chemistry_Model = airThermal
Eos_Model = Fuego
Transport_Model = Simple

PELE_HOME ?= ../../..

ifeq ($(USE_MWSOLVER),TRUE)
  HARPS_SRC_DIR = ../../../../harps/src
  CEXE_sources += harps.cpp CartesianCoordinates.cpp ConfigParser.cpp MatrixSolver.cpp
  VPATH_LOCATIONS   += $(HARPS_SRC_DIR)
  INCLUDE_LOCATIONS += $(HARPS_SRC_DIR)

  PETSC_DIR := $(shell spack location -i petsc)
  INCLUDE_LOCATIONS += $(PETSC_DIR)/include
  LIBRARY_LOCATIONS += $(PETSC_DIR)/lib
  LIBRARIES         += -lpetsc
endif

include $(PELE_HOME)/Exec/Make.PeleLMeX
