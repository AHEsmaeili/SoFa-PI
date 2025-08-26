
CXX=g++
CXXFLAGS  = -std=c++11 -fopenmp -O2  -fPIC -c
SWIGFLAGS = -c++ -python -shadow  

# location of the Python header files
PYTHON_VERSION = 3.10
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)


all: _JansenRitNet.so 

JansenRit : _JansenRit.so

OBJS = JansenRitNet_wrap.o 

# -----------------------------------------------------------------------------
_JansenRitNet.so: JansenRitNet_wrap.o
	$(CXX) -shared -fopenmp JansenRitNet_wrap.o -o _JansenRitNet.so

JansenRitNet_wrap.o : JansenRitNet_wrap.cpp JansenRitNet.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE) 

JansenRitNet_wrap.cpp : JansenRitNet.i 
	swig $(SWIGFLAGS) -o JansenRitNet_wrap.cpp JansenRitNet.i
# -----------------------------------------------------------------------------

.PHONEY:clean
clean :
	rm -rf *.o  *.so *.pyc
	rm -rf JansenRitNet_wrap.cpp JansenRitNet.py 
	rm -rf __pycache__

.PHONEY : clean all JansenRitNet
