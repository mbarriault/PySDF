//
//  pysdfmodule.c
//  pydsfmodule
//
//  Created by Michael Barriault on 11-04-27.
//  Copyright 2011 MikBarr Studios. All rights reserved.
//

#include "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/Python.h"
#include "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/oldnumeric.h"
#define NUMERICS_PACKAGE "NumPy"
#include "/usr/local/include/bbhutil.h"
#include <stdio.h>
#include <sdf.h>

static PyObject* pysdf_read(PyObject* self, PyObject* args) {
    int i;
    const char* file_name;
    char* gf_name;
    int level;
    int* shape;
    char* cnames;
    int rank;
    double time = 0.;
    double* coords;
    double* data;
    if ( !PyArg_ParseTuple(args, "si", &file_name, &level) )
        return NULL;
    gft_read_name(file_name, level, gf_name);
    gft_read_rank(gf_name, level, &rank);
    shape = calloc(rank,sizeof(int));
    gft_read_shape(gf_name, level, shape);
    int cnames_count = 0;
    int coords_count = 0;
    int data_count = 1;
    for ( i=0; i<rank; i++ ) {
        cnames_count += 2;
        coords_count += shape[i];
        data_count *= shape[i];
    }
    cnames = calloc(cnames_count,sizeof(char));
    coords = calloc(coords_count,sizeof(double));
    data = malloc(data_count*sizeof(double));
    gft_read_full(gf_name, level, shape, cnames, rank, &time, coords, data);
    PyArrayObject* pcoords = PyArray_FromDimsAndData(1, coords_count, PyArray_DOUBLE, coords);
    Py_IncRef(pcoords); free(coords);
    PyArrayObject* pdata = PyArray_FromDimsAndData(rank, shape, PyArray_DOUBLE, data);
    Py_IncRef(pdata); free(data);
    return Py_BuildValue("ssdOO", gf_name, cnames, time, pcoords, pdata);
}

static PyMethodDef pysdfMethods[] = {
    {"read",  pysdf_read, METH_VARARGS, "Read SDF file."},
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

