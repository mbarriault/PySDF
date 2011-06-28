//
//  pysdfmodule.cpp
//  pydsfmodule
//
//  Created by Michael Barriault on 11-04-27.
//  Copyright 2011 MikBarr Studios. All rights reserved.
//

#include "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/Python.h"
#include "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/oldnumeric.h"
#define NUMERICS_PACKAGE "NumPy"
#include <bbhutil.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

static PyObject* pysdf_read(PyObject* self, PyObject* args) {
    // Get arguments
    char* file_name;
    int level;
    if ( !PyArg_ParseTuple(args, "si", &file_name, &level) )
        return NULL;
    
    // Parameters for gft_read_full
    string gf_name;
    vector<int> shape;
    string cnames;
    int rank;
    double time = 0.;
    vector<double> *coords;
    vector<double> *data;
   
    int res;
    // Get the grid function name
    if ( !(res = gft_read_name(file_name, level, &gf_name[0])) )
        return Py_BuildValue("i", res);
    
    // Get the dimensionality to preallocate arrays
    if ( !(res = gft_read_rank(gf_name.c_str(), level, &rank)) )
        return Py_BuildValue("i", res);
    
    // Get the shape to preallocate arrays
    shape.assign(rank,0);
    if ( !(res = gft_read_shape(gf_name.c_str(), level, &shape[0])) )
        return Py_BuildValue("i", res);
    
    // Preallocate arrays
    int cnames_count = 0;
    int coords_count = 0;
    int data_count = 1;
    for ( int i=0; i<rank; i++ ) {
        cnames_count += 2;
        coords_count += shape[i];
        data_count *= shape[i];
    }
    cnames.assign(cnames_count,' ');
    coords = new vector<double>(coords_count, 0.);
    data = new vector<double>(data_count, 0.);
    
    // Read in data
    res = gft_read_full(gf_name.c_str(), level, &shape[0], &cnames[0], rank, &time, &coords->at(0), &data->at(0));
    
    vector<int> shape_rev(rank,0);
    for ( int i=0; i<rank; i++ ) shape_rev.at(i) = shape.at(rank-i-1);
    
    // Prepare Python objects for arrays
    PyArrayObject* pcoords = (PyArrayObject*)PyArray_FromDimsAndData(1, &coords_count, PyArray_DOUBLE, (char*)(&coords->at(0)));
    Py_IncRef((PyObject*)pcoords);
    PyArrayObject* pdata = (PyArrayObject*)PyArray_FromDimsAndData(rank, &shape_rev[0], PyArray_DOUBLE, (char*)(&data->at(0)));
    Py_IncRef((PyObject*)pdata);
    
    // Return tuple
    // grid function name, coordinate names, time value, packed coordinates, data
    return Py_BuildValue("sdsOO", gf_name.c_str(), time, cnames.c_str(), (PyObject*)pcoords, (PyObject*)pdata);
    
    // Note we DO NOT delete coords and data
    // PyArray_FromDimsAndData only copies the pointer, so we do not wish to call the vector constructor
}

static PyObject* pysdf_write(PyObject* self, PyObject* args) {
    // Get arguments
    char* file_name;
    double time;
    char* cnames;
    PyArrayObject* coords;
    PyArrayObject* data;
    if ( !PyArg_ParseTuple(args, "sdsOO", &file_name, &time, &cnames, &coords, &data) )
        return NULL;
    
    // Determine shape
    int rank = data->nd;
    vector<int> shape(rank,0);
    for ( int i=0; i<rank; i++ ) shape[i] = data->dimensions[i];
    
    // Write data
    int res = gft_out_full(file_name, time, &shape[0], cnames, rank, (double*)coords->data, (double*)data->data);
    
    // Return result
    return Py_BuildValue("i", res);
}

static PyMethodDef pysdfMethods[] = {
    {"read",  pysdf_read, METH_VARARGS, "Read SDF file."},
    {"write",  pysdf_write, METH_VARARGS, "Write SDF file."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initpysdf(void) {
    (void) Py_InitModule("pysdf", pysdfMethods);
    import_array();
}

int main (int argc, const char * argv[])
{

    // insert code here...
    printf("Hello, World!\n");
    return 0;
}

